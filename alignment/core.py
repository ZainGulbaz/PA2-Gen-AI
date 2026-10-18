"""Response-only causal LM math. No Trainer or prebuilt DPO loop."""
import torch
import torch.nn.functional as F


def format_prompt(instruction):
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_tokens=None):
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("Need equally sized, nonempty prompt/response lists")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    pad = getattr(tokenizer, "pad_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = pad if pad is not None else (eos if eos is not None else 0)
    rows = []
    for prompt, response in zip(prompts, responses):
        p = list(tokenizer.encode(prompt, add_special_tokens=False))
        if max_prompt_tokens is not None:
            p = p[-max_prompt_tokens:]
        r = list(tokenizer.encode(response, add_special_tokens=False))
        if eos is not None:
            r.append(eos)  # Append BEFORE truncation; do not force EOS back in.
        ids = (p + r)[:max_length]
        labels = ([-100] * len(p) + r)[:max_length]
        rows.append((ids, labels))
    width = max(len(ids) for ids, _ in rows)
    return {
        "input_ids": torch.tensor([x + [pad] * (width-len(x)) for x, _ in rows], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(x) + [0] * (width-len(x)) for x, _ in rows], dtype=torch.long),
        "labels": torch.tensor([y + [-100] * (width-len(y)) for _, y in rows], dtype=torch.long),
    }


def model_device(model):
    return next(model.parameters()).device


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def response_scores(model, batch, chunk_size=32):
    """Return SUM log p(response|prompt) and scored token counts per row.

    Shift labels once. Cross entropy runs in FP32 only on response positions,
    in chunks, avoiding an FP32 [batch,sequence,vocabulary] allocation.
    """
    b = move_batch(batch, model_device(model))
    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    targets = b["labels"][:, 1:]
    mask = targets.ne(-100) & b["attention_mask"][:, 1:].bool()
    positions = mask.nonzero(as_tuple=False)
    scores = logits[:, :1, :1].float().sum(dim=(1, 2)) * 0.0
    for pos in positions.split(chunk_size):
        row, col = pos[:, 0], pos[:, 1]
        selected = logits[row, col, :].float()
        token_logps = -F.cross_entropy(selected, targets[row, col], reduction="none")
        scores = scores.scatter_add(0, row, token_logps)
    return scores, mask.sum(dim=1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return response_scores(model, batch)[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    scores, counts = response_scores(model, build_sft_batch(tokenizer, prompts, responses, max_length))
    # Token mean over the entire batch; zero differentiable loss if no targets survive.
    return -scores.sum() / counts.sum().clamp_min(1)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, max_prompt_tokens=None):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_tokens),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_tokens),
    }


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                              reference_chosen_logps, reference_rejected_logps, beta):
    ref_gap = reference_chosen_logps - reference_rejected_logps
    gap = (policy_chosen_logps-policy_rejected_logps) - ref_gap.to(policy_chosen_logps.device)
    return -F.logsigmoid(beta * gap).mean()


def dpo_batch_loss(policy, reference, batch, beta):
    pc, nc = response_scores(policy, batch["chosen"])
    pr, nr = response_scores(policy, batch["rejected"])
    with torch.no_grad():
        rc, _ = response_scores(reference, batch["chosen"])
        rr, _ = response_scores(reference, batch["rejected"])
    rc, rr = rc.to(pc.device), rr.to(pc.device)
    loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    margin = beta * ((pc-rc)-(pr-rr))
    return loss, margin.detach(), nc, nr


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses,
                     rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return dpo_batch_loss(policy_model, reference_model, batch, beta)[0]
