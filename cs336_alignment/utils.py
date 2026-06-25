import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=="cpu" else "flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer, 
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)

    pad_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is None:
        pad_id = tokenizer.eos_token_id

    token_ids = []
    masks = []

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_id = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_id = tokenizer.encode(output_str, add_special_tokens=False)

        token_id = prompt_id + output_id
        mask = [0] * (len(prompt_id) - 1) + [1] * len(output_id)

        token_ids.append(torch.tensor(token_id, dtype=torch.long))
        masks.append(torch.tensor(mask, dtype=torch.long))

    token_ids = torch.nn.utils.rnn.pad_sequence(
        [token for token in token_ids], batch_first=True, padding_value=pad_id
    )

    response_mask = torch.nn.utils.rnn.pad_sequence(
        masks, batch_first=True, padding_value=0
    )

    return {
        "input_ids": token_ids[:, :-1],
        "labels": token_ids[:, 1:],
        "response_mask": response_mask,
    }
    
    
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits

    log_probs = torch.log_softmax(logits, dim=-1)
    label_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    results = {
        "log_probs": label_log_probs
    }

    if return_token_entropy:
        probs = torch.softmax(logits, dim=-1)
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
        results["token_entropy"] = token_entropy

    return results