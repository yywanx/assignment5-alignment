import torch
from typing import Callable, Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    all_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        results = reward_fn(resp, gt)
        all_rewards.append(results)

    rewards = torch.tensor([r["reward"] for r in all_rewards], dtype=torch.float32)
    format_rewards = torch.tensor([r["format_reward"] for r in all_rewards], dtype=torch.float32)
    
    metadata = {
        "mean_total_reward": torch.mean(rewards).item(),
        "mean_format_reward": torch.mean(format_rewards).item()
    }

    return rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards_by_group = raw_rewards.view(-1, group_size)
    
    if baseline == "mean":
        mu = raw_rewards_by_group.mean(dim=-1, keepdim=True)
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        std = raw_rewards_by_group.std(dim=-1, keepdim=True)
    else:
        raise NotImplementedError(f"Unsupported advantage_normalizer: {advantage_normalizer}")

    group_advantages = (raw_rewards_by_group - mu) / (std + advantage_eps)
    advantages = group_advantages.view(-1)

    metadata = {
        "mean_raw_rewards": raw_rewards.mean().item(),
        "std_raw_rewards": raw_rewards.std().item(),
        "mean_group_mean": mu.mean().item(),
        "mean_group_std": std.mean().item(),
        "mean_advantages": advantages.mean().item(),
        "std_advantages": advantages.std().item(),
    }

    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError

    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    policy_gradient = raw_rewards_or_advantages * policy_log_probs

    per_token_policy_gradient_loss = -policy_gradient
    metadata = {}

    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch_sequence(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
):
    if loss_normalization != "sequence":
        raise NotImplementedError

    mask = mask.to(per_token_policy_gradient_loss.dtype)

    sum_policy_gradient_loss = (per_token_policy_gradient_loss * mask).sum(dim=-1)
    per_sequence_loss = sum_policy_gradient_loss / mask.sum(dim=-1).clamp_min(1.0)

    return per_sequence_loss.mean()


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    model.train()
    metadata = {}
    device = next(model.parameters()).device

    raw_rewards, metadata_ = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths
    )
    metadata.update(metadata_)

    raw_rewards = raw_rewards.to(device)

    advantages, metadata_ = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer
    )
    metadata.update(metadata_)

    batch = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer
    )
    
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)
    
    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    token_count = torch.zeros((), device=device)
    microbatch_size = len(inputs) // gradient_accumulation_steps
    for i in range(0, len(inputs), microbatch_size):
        inputs_microbatch = inputs[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]
        response_mask_microbatch = response_mask[i:i+microbatch_size]
        advantages_microbatch = advantages[i:i+microbatch_size]

        out = get_response_log_probs(
            model=model,
            input_ids=inputs_microbatch,
            labels=labels_microbatch,
            return_token_entropy=True,
        )
        log_probs = out["log_probs"]

        entropy_sum = entropy_sum + (out["token_entropy"].detach() * response_mask_microbatch).sum()
        token_count = token_count + response_mask_microbatch.sum()

        per_token_loss, metadata_ = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages_microbatch,
            policy_log_probs=log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
            response_mask=response_mask_microbatch
        )
        metadata.update(metadata_)

        loss = aggregate_loss_across_microbatch_sequence(
            per_token_policy_gradient_loss=per_token_loss,
            mask=response_mask_microbatch,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant
        )

        loss = loss * (len(inputs_microbatch) / len(inputs))
        loss.backward()

        total_loss = total_loss + loss.detach()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata["loss"] = total_loss.detach().item()
    metadata["grad_norm"] = grad_norm.item()
    metadata["token_entropy"] = (entropy_sum / token_count.clamp_min(1)).item()

    return total_loss.detach(), metadata


