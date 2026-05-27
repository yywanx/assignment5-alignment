import argparse
import logging
import json
from pathlib import Path
from statistics import mean
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]

PROMPTS = {
    "question_only": "cs336_alignment/prompts/question_only.prompt",
    "r1_zero": "cs336_alignment/prompts/r1_zero.prompt",
    "r1_zero_three_shot": "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
}


def load_gsm8k(file_path: Path) -> list[dict]:
    data = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            data.append({"question": sample["question"], "answer": sample["answer"]})
    
    return data


def categorize(format_reward: float, answer_reward: float) -> int:
    if format_reward == 1.0 and answer_reward == 1.0:
        return 1
    elif format_reward == 1.0 and answer_reward == 0.0:
        return 2
    elif format_reward == 0.0 and answer_reward == 0.0:
        return 3
    else:
        raise ValueError(f"Invalid format reward {format_reward} or correctness reward {answer_reward}")


def summarize(name: str, results: list[dict]) -> dict:
    summary = {
        "name": name,
        "n": len(results),
        "format_rate": mean(row["format_reward"] for row in results),
        "answer_accuracy": mean(row["answer_reward"] for row in results),
        "category1": sum(row["category"] == 1 for row in results),
        "category2": sum(row["category"] == 2 for row in results),
        "category3": sum(row["category"] == 3 for row in results)
    }

    logger.info(
        "[%s] n=%d fmt=%.4f acc=%.4f cat1=%d cat2=%d cat3=%d",
        name,
        len(results),
        summary["format_rate"],
        summary["answer_accuracy"],
        summary["category1"],
        summary["category2"],
        summary["category3"],
    )

    return summary

def main(args):
    test_data = load_gsm8k(args.test_path)
    if args.limit is not None:
        test_data = test_data[: args.limit]
    logger.info(f"loaded {len(test_data)} GSM8K samples")

    server = VLLMServer(model_id=args.model_id, gpu=args.gpu)
    server.start()
    logger.info(f"started VLLM server at {server.base_url}")

    summaries = []
    try:
        for name, path in PROMPTS.items():
            sampling_params = {
                "n": 1,
                "seed": args.seed,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": ["</answer>"] if name != "question_only" else None,
                "include_stop_str_in_output": True if name != "question_only" else False
            }

            path = ROOT_DIR / path
            template = path.read_text()
            prompts = [template.format(question=sample["question"]) for sample in test_data]

            completions = server.generate_completions(prompts, sampling_params, batch_size=args.batch_size)
            responses = [c.text for c in completions]

            assert len(test_data) == len(prompts)
            assert len(test_data) == len(responses)

            results = []
            for sample, prompt, response in zip(test_data, prompts, responses):
                gt = sample["answer"].split("####")[-1].strip().replace(",", "")

                if name == "question_only":
                    rewards = question_only_reward_fn(response, gt)
                else:
                    rewards = r1_zero_reward_fn(response, gt)
                category = categorize(rewards["format_reward"], rewards["answer_reward"])

                results.append({
                    "question": sample["question"],
                    "prompt": prompt,
                    "ground_truth": gt,
                    "response": response,
                    **rewards,
                    "category": category
                })
            logger.info(f"finished generating responses for {name} setting")
            
            output_dir = args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{name}.jsonl"
            with output_path.open("w") as f:
                for row in results:
                    f.write(json.dumps(row) + "\n")
            logger.info(f"dumped results to {output_path} for {name} setting")

            summaries.append(summarize(name, results))

        summary_path = output_dir / "results_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2))
        logger.info(f"dumped summaries to {summary_path}")
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-path", type=Path, default=ROOT_DIR / "data" / "gsm8k" / "test.jsonl")
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "gsm8k_prompting")
    parser.add_argument("--limit", type=int, default=None, help="For quick debug/smoke test")

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    main(args)