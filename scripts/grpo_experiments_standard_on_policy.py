import argparse
import logging
import random
import os
import torch
import json
import wandb
import time
import numpy as np
from pathlib import Path
from cs336_alignment.utils import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
R1_ZERO_PROMPT = ROOT_DIR / "cs336_alignment" / "prompts" / "r1_zero.prompt"


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gsm8k(file_path: Path) -> list[dict]:
    data = []

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            data.append({"question": sample["question"], "answer": sample["answer"]})

    return data


def get_sampling_params(
    seed,
    temperature,
    max_tokens,
    n=1,
    stop=["</answer>"],
    include_stop_str_in_output=True,
):
    return {
        "n": n,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "include_stop_str_in_output": include_stop_str_in_output
    }



def evaluate(
    data: list[dict],
    template: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    server: VLLMServer,
) -> tuple[dict[str, float], list[dict]]:
    sampling_params = get_sampling_params(seed, temperature, max_tokens)

    prompts = [template.format(question=sample["question"]) for sample in data]
    ground_truths = [sample["answer"].split("####")[-1].strip().replace(",", "") for sample in data]

    completions = server.generate_completions(prompts, sampling_params=sampling_params, batch_size=batch_size)

    total_rewards, format_rewards, answer_rewards, lengths = [], [], [], []
    records = []
    for sample, prompt, ground_truth, completion in zip(data, prompts, ground_truths, completions):
        rewards = r1_zero_reward_fn(completion.text, ground_truth)

        records.append(
            {
                "question": sample["question"],
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": completion.text,
                "response_token_length": len(completion.token_ids),
                **rewards
            }
        )

        total_rewards.append(rewards["reward"])
        format_rewards.append(rewards["format_reward"])
        answer_rewards.append(rewards["answer_reward"])
        lengths.append(len(completion.token_ids))

    metrics = {
        "val/reward": float(np.mean(total_rewards)),
        "val/format_reward": float(np.mean(format_rewards)),
        "val/answer_reward": float(np.mean(answer_rewards)),
        "val/response_length": float(np.mean(lengths)),
        "val/n": len(data),
    }

    return metrics, records


def log_rollouts(records: list[dict], step: int, split: str, num_examples: int = 8):
    columns = ["question", "ground_truth", "reward", "format_reward", "answer_reward", "response_length"]
    table = wandb.Table(columns=columns)
    
    for record in records[:num_examples]:
        table.add_data(
            record["question"],
            record["ground_truth"],
            record.get("reward"),
            record.get("format_reward"),
            record.get("answer_reward"),
            record.get("response_token_length"),
        )

    wandb.log({f"{split}/rollouts": table}, step=step)


def save_rollouts(records: list[dict], output_dir: Path, name:str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"saved {len(records)} rollouts to {output_path}")


def main(args):
    # configuration
    seed_everything(args.seed)

    output_dir = args.output_dir / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    template = R1_ZERO_PROMPT.read_text()
    prompts_per_step = args.rollout_batch_size // args.group_size
    
    wandb.init(
        project=args.wandb_project,
        name=f"grpo-on-policy-seed{args.seed}",
        mode=args.wandb_mode,
        config=vars(args) | {"prompts_per_step": prompts_per_step},
    )

    # model, optimizer, vLLM server
    model, tokenizer = get_model_and_tokenizer(args.model_id, device=args.policy_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    server = VLLMServer(model_id=args.model_id, gpu=args.rollout_device, seed=args.seed)
    server.start()
    server.init_weight_sync(args.policy_device)
    logger.info(f"started VLLM server at: {server.base_url}")

    # load data
    train_data = load_gsm8k(args.train_path)
    val_data = load_gsm8k(args.val_path)[: args.n_val_examples]

    order = list(range(len(train_data)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    if args.n_train_examples:
        order = order[:args.n_train_examples]
    logger.info(f"loaded {len(order)} train and {len(val_data)} val examples")

    try:
        # initial eval
        metrics, records = evaluate(
            val_data, template, args.seed, args.sampling_temperature, args.sampling_max_tokens, args.rollout_batch_size, server
        )
        wandb.log(metrics, step=0)
        log_rollouts(records, step=0, split="val")
        save_rollouts(records, output_dir=output_dir, name="val_step0")

        # for loop train
        for step in range(1, args.num_rollout_steps + 1):
            step_start = time.monotonic()
            
            ## sync weights
            server.sync_policy_weights(model)

            ## rollout
            start = (step - 1) * prompts_per_step
            batch_indices = order[start:start + prompts_per_step]

            unique_prompts = []
            repeated_questions = []
            repeated_prompts = []
            repeated_ground_truths = []
            for idx in batch_indices:
                sample = train_data[idx]

                unique_prompts.append(template.format(question=sample["question"]))
                repeated_questions.extend([sample["question"]] * args.group_size)
                repeated_prompts.extend([template.format(question=sample["question"])] * args.group_size)
                repeated_ground_truths.extend([sample["answer"].split("####")[-1].strip().replace(",", "")] * args.group_size)

            sampling_params = get_sampling_params(args.seed + step, args.sampling_temperature, args.sampling_max_tokens, n=args.group_size)
            completions = server.generate_completions(unique_prompts, sampling_params, batch_size=args.rollout_batch_size)
            responses = [c.text for c in completions]
            response_token_counts = [len(c.token_ids) for c in completions]

            ## train
            loss, metadata = grpo_train_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
            )

            ## eval
            train_metrics = {
                "train/loss": metadata["loss"],
                "train/grad_norm": metadata["grad_norm"],
                "train/token_entropy": metadata["token_entropy"],
                "train/reward": metadata["mean_total_reward"],
                "train/format_reward": metadata["mean_format_reward"],
                "train/mean_raw_rewards": metadata["mean_raw_rewards"],
                "train/std_raw_rewards": metadata["std_raw_rewards"],
                "train/mean_advantages": metadata["mean_advantages"],
                "train/std_advantages": metadata["std_advantages"],
                "train/response_length": float(np.mean(response_token_counts)),
                "train/step_time_s": time.monotonic() - step_start,
            }
            wandb.log(train_metrics, step=step)
            logger.info(
                "[step %d/%d] - loss=%.4f grad_norm=%.3f entropy=%.3f reward=%.3f fmt=%.3f",
                step,
                args.num_rollout_steps,
                train_metrics["train/loss"],
                train_metrics["train/grad_norm"],
                train_metrics["train/token_entropy"],
                train_metrics["train/reward"],
                train_metrics["train/format_reward"],
            )

            ## save/log    
            if step % args.rollout_log_interval == 0:
                train_records = []
                for question, prompt, ground_truth, response, length in zip(repeated_questions, repeated_prompts, repeated_ground_truths, responses, response_token_counts):
                    rewards = r1_zero_reward_fn(response, ground_truth)
                    train_records.append(
                        {
                            "question": question,
                            "prompt": prompt,
                            "ground_truth": ground_truth,
                            "response": response,
                            "response_token_length": length,
                            **rewards
                        }
                    )
                log_rollouts(train_records, step=step, split='train', num_examples=16)

            if step % args.eval_interval == 0 or step == args.num_rollout_steps:
                metrics, records = evaluate(
                    val_data, template, args.seed, args.sampling_temperature, args.sampling_max_tokens, args.rollout_batch_size, server
                )
                wandb.log(metrics, step=step)
                logger.info("[eval at step %d] %s", step, metrics)
                log_rollouts(records, step=step, split="val")
                save_rollouts(records, output_dir=output_dir, name=f"val_step{step}")

    finally:
        server.stop()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train-path", default=ROOT_DIR / "data" / "gsm8k" / "train.jsonl")
    parser.add_argument("--val-path", default=ROOT_DIR / "data" / "gsm8k" / "test.jsonl")
    parser.add_argument("--output-dir", default=ROOT_DIR / "results" / "grpo_on_policy")

    # model
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")

    # devices
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--rollout-device", type=int, default=1)

    # training hyperparams
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)

    # inference hyperparams
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # eval
    parser.add_argument("--eval-interval", type=int, default=10)

    # logging
    parser.add_argument("--wandb-project", default="cs336-assignment5-grpo")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--rollout-log-interval", type=int, default=40)

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    main(args)