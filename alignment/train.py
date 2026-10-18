"""Full parameter SFT + two explicit DPO runs. One policy GPU, one reference GPU."""
import argparse
import gc
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from alignment.core import build_sft_batch, build_dpo_batch, format_prompt, response_scores, dpo_batch_loss
from alignment.utils import read_json, write_json, append_jsonl, environment, seed_all, sha

BASE = "Qwen/Qwen2.5-0.5B"


def collate(rows, tokenizer, stage, max_length, prompt_limit):
    prompts = [format_prompt(r["prompt"]) for r in rows]
    if stage == "sft":
        return build_sft_batch(tokenizer, prompts, [r["response"] for r in rows], max_length, prompt_limit)
    return build_dpo_batch(tokenizer, prompts, [r["chosen"] for r in rows],
                           [r["rejected"] for r in rows], max_length, prompt_limit)


def amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()


def load_model(path, device, trainable):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def save_checkpoint(model, tokenizer, path, config, state=None):
    """Stage a new checkpoint before replacing the old directory."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp, safe_serialization=True)
    tokenizer.save_pretrained(temp)
    write_json(temp/"run_config.json", config)
    if state is not None:
        torch.save(state, temp/"training_state.pt")
    if path.exists():
        shutil.rmtree(path)
    temp.rename(path)


@torch.no_grad()
def validate(model, reference, loader, stage, beta, device):
    model.eval()
    total_loss, total_tokens, total_pairs, correct, margin_sum = 0., 0, 0, 0, 0.
    for batch in tqdm(loader, desc="validation", leave=False):
        with amp(device):
            if stage == "sft":
                lp, count = response_scores(model, batch)
                total_loss += float(-lp.sum())
                total_tokens += int(count.sum())
            else:
                loss, margin, _, _ = dpo_batch_loss(model, reference, batch, beta)
                n = margin.numel()
                total_loss += float(loss)*n
                total_pairs += n
                correct += int((margin > 0).sum())
                margin_sum += float(margin.sum())
    model.train()
    if stage == "sft":
        if total_tokens == 0:
            raise ValueError("No scored validation response tokens")
        return {"val_loss": total_loss/total_tokens, "val_response_tokens": total_tokens}
    return {"val_loss": total_loss/total_pairs, "reward_accuracy": correct/total_pairs,
            "reward_margin": margin_sum/total_pairs, "val_pairs": total_pairs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sft", "hh", "uf"], required=True)
    p.add_argument("--data", default="outputs/prepared")
    p.add_argument("--out", default="outputs")
    p.add_argument("--sft-checkpoint", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accumulation", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reference-device", default="cuda:1")
    p.add_argument("--restart", action="store_true", help="Restart an interrupted stage from its initial model")
    p.add_argument("--resume", action="store_true", help="Resume exact state saved by --save-resume")
    p.add_argument("--save-resume", action="store_true", help="Optional large optimizer/RNG snapshots; requires extra disk")
    args = p.parse_args()
    is_sft = args.stage == "sft"
    args.batch_size = args.batch_size or (2 if is_sft else 1)
    args.accumulation = args.accumulation or (16 if is_sft else 32)
    args.lr = args.lr or (2e-5 if is_sft else 1e-6)
    if args.max_prompt_length >= args.max_length:
        raise ValueError("Reserve response space: max_prompt_length must be less than max_length")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("Enable GPU T4 x2 in Kaggle notebook settings")
    if not is_sft and args.reference_device == "cuda:1" and torch.cuda.device_count() < 2:
        raise RuntimeError("DPO expects Kaggle GPU T4 x2; select that accelerator")
    seed_all(args.seed)
    out = Path(args.out)
    run = out / ("sft" if is_sft else f"dpo_{args.stage}")
    if (run/"complete.json").exists() and not args.restart:
        print("Already complete:", run)
        return
    if args.restart and args.resume:
        raise ValueError("Choose restart OR resume")
    if args.restart and run.exists():
        shutil.rmtree(run)
    resume_path = run/"resume"
    if args.resume and not (resume_path/"training_state.pt").exists():
        raise FileNotFoundError("No exact resume snapshot; restart this stage with --restart")
    if run.exists() and (run/"metrics.jsonl").exists() and not args.resume:
        raise RuntimeError("Interrupted stage found. Use --resume if saved; otherwise --restart. Logs must not mix different runs.")
    run.mkdir(parents=True, exist_ok=True)
    initial = BASE if is_sft else (args.sft_checkpoint or str(out/"sft"/"best"))
    if not is_sft and not Path(initial).exists():
        raise FileNotFoundError("Train SFT first; both DPO runs start from the same SFT best checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(initial)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(str(resume_path) if args.resume else initial, args.device, True)
    reference = None
    if not is_sft:
        reference = load_model(initial, args.reference_device, False)
        disable_dropout(model)
        disable_dropout(reference)
    train_rows = read_json(Path(args.data)/f"{args.stage}_train.json")
    val_rows = read_json(Path(args.data)/f"{args.stage}_val.json")
    from functools import partial
    collator = partial(collate, tokenizer=tokenizer, stage=args.stage,
                       max_length=args.max_length, prompt_limit=args.max_prompt_length)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    microbatches = math.ceil(len(train_rows)/args.batch_size)
    updates_epoch = math.ceil(microbatches/args.accumulation)
    total_updates = updates_epoch*args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1 if is_sft else 0.0)
    warmup = math.ceil(total_updates*0.03) if is_sft else 0
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_updates) if is_sft else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    config = dict(vars(args), model=BASE, initial_checkpoint=initial,
                  reference_checkpoint=None if is_sft else initial,
                  train_examples=len(train_rows), validation_examples=len(val_rows),
                  trainable_strategy="full parameters", optimizer="AdamW", weight_decay=0.1 if is_sft else 0.0,
                  schedule="cosine" if is_sft else "constant", warmup_steps=warmup,
                  effective_batch_size=args.batch_size*args.accumulation,
                  trainable_precision="fp32", autocast_precision="fp16", reference_precision="fp32 + fp16 autocast",
                  sft_reduction="global response token mean per accumulated group",
                  prompt_template="### Instruction:\\n{instruction}\\n\\n### Response:\\n",
                  prompt_truncation="keep last 256 tokens (or configured prompt limit), then right-truncate full sequence",
                  gradient_clipping=1.0, environment=environment(),
                  manifest_sha256=sha(Path(args.data)/"manifest.json"),
                  planned_optimizer_steps=total_updates)
    write_json(run/"config.json", config)
    step, start_epoch, start_micro, skipped_updates = 0, 0, 0, 0
    best = math.inf if is_sft else -math.inf
    if args.resume:
        # Load only our own snapshot; not an untrusted external pickle.
        saved = torch.load(resume_path/"training_state.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config:
            # Flags may differ; all math/data settings must match.
            excluded = {"resume", "save_resume", "restart", "environment"}
            assert {k:v for k,v in saved["config"].items() if k not in excluded} == {k:v for k,v in config.items() if k not in excluded}, "Resume config changed"
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    state[k] = v.to(args.device)
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, start_epoch, start_micro, best = saved["step"], saved["epoch"], saved["next_micro"], saved["best"]
        skipped_updates = saved["skipped_updates"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        initial_metrics = validate(model, reference, val_loader, args.stage, args.beta, args.device)
        append_jsonl(run/"metrics.jsonl", {"step": 0, "kind": "validation", **initial_metrics})
        # Save only trained checkpoints as best; initial validation is diagnostic.
    started = time.time()
    model.train()
    for epoch in range(start_epoch, args.epochs):
        indices = list(range(len(train_rows)))
        random.Random(args.seed+epoch).shuffle(indices)
        loader = DataLoader([train_rows[i] for i in indices], batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed+epoch))
        iterator = iter(loader)
        cursor = start_micro if epoch == start_epoch else 0
        for _ in range(cursor):
            next(iterator)
        progress = tqdm(total=microbatches, initial=cursor, desc=f"{args.stage} epoch {epoch+1}")
        while cursor < microbatches:
            group = [next(iterator) for _ in range(min(args.accumulation, microbatches-cursor))]
            optimizer.zero_grad(set_to_none=True)
            count = sum(int(b["labels"][:,1:].ne(-100).sum()) for b in group) if is_sft else sum(b["chosen"]["input_ids"].shape[0] for b in group)
            if count == 0:
                raise ValueError("No response tokens survived training truncation")
            group_loss, group_margin, group_correct = 0., 0., 0
            for batch in group:
                with amp(args.device):
                    if is_sft:
                        lp, _ = response_scores(model, batch)
                        numerator = -lp.sum()
                        loss = numerator/count
                        group_loss += float(numerator.detach())/count
                    else:
                        mean_loss, margin, nc, nr = dpo_batch_loss(model, reference, batch, args.beta)
                        if int(nc.min()) < 1 or int(nr.min()) < 1:
                            raise ValueError("DPO side has no response targets")
                        loss = mean_loss*margin.numel()/count
                        group_loss += float(mean_loss.detach())*margin.numel()/count
                        group_margin += float(margin.sum())/count
                        group_correct += int((margin>0).sum())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; stop and inspect data/precision")
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            lr_used = optimizer.param_groups[0]["lr"]
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if skipped:
                skipped_updates += 1
            elif scheduler:
                scheduler.step()
            step += 1
            cursor += len(group)
            row = {"step": step, "kind": "train", "epoch": epoch+1, "loss": group_loss,
                   "lr": lr_used, "grad_norm": float(grad_norm), "amp_skipped_update": skipped}
            if not is_sft:
                row.update(reward_margin=group_margin, reward_accuracy=group_correct/count)
            append_jsonl(run/"metrics.jsonl", row)
            progress.update(len(group))
            progress.set_postfix(loss=f"{group_loss:.4f}")
            if step % args.eval_every == 0 or cursor == microbatches:
                metrics = validate(model, reference, val_loader, args.stage, args.beta, args.device)
                append_jsonl(run/"metrics.jsonl", {"step": step, "kind": "validation", **metrics})
                candidate = metrics["val_loss"] if is_sft else metrics["reward_accuracy"]
                improved = candidate < best if is_sft else candidate > best
                if improved:
                    best = candidate
                    save_checkpoint(model, tokenizer, run/"best", config)
                    write_json(run/"best_metrics.json", {"step": step, **metrics})
                if args.save_resume:
                    state = {"config": config, "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict() if scheduler else None,
                             "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
                             "next_micro": cursor, "best": best, "skipped_updates": skipped_updates,
                             "torch_rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
                    save_checkpoint(model, tokenizer, resume_path, config, state)
        progress.close()
    save_checkpoint(model, tokenizer, run/"final", config)
    if resume_path.exists():
        shutil.rmtree(resume_path)
    write_json(run/"complete.json", {"optimizer_groups": step, "successful_updates": step-skipped_updates,
                                   "skipped_updates": skipped_updates, "best_metric": best,
                                   "elapsed_seconds_this_session": time.time()-started})
    del optimizer, model, reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Saved best and final:", run)


if __name__ == "__main__":
    main()
