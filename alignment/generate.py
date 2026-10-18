"""Resumable saved generations; model comparison uses identical greedy settings."""
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from alignment.core import format_prompt
from alignment.utils import read_json, write_json, read_jsonl, append_jsonl, sha, seed_all

BASE = "Qwen/Qwen2.5-0.5B"
SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def generate_one(model, tokenizer, instruction, chat, max_new_tokens, prompt_limit):
    if chat:
        prompt = tokenizer.apply_chat_template([{"role":"user", "content":instruction}], tokenize=False, add_generation_prompt=True)
    else:
        prompt = format_prompt(instruction)
    ids = tokenizer.encode(prompt, add_special_tokens=False)[-prompt_limit:]
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                do_sample=False, max_new_tokens=max_new_tokens,
                                repetition_penalty=1.1, no_repeat_ngram_size=4, num_beams=1,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                use_cache=True)
    return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip()


def checkpoint_fingerprint(source):
    path = Path(source)
    if not path.exists():
        from huggingface_hub import HfApi
        return HfApi().model_info(source).sha
    digest = hashlib.sha256()
    for p in sorted(path.glob("*")):
        if p.suffix in [".json", ".safetensors", ".bin"]:
            digest.update(p.name.encode())
            # Stream hashing so large models are not copied into RAM.
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def generate_set(source, name, rows_path, dest, chat, adapter, sanity_only, max_new_tokens, prompt_limit):
    seed_all(42)
    rows = read_json(rows_path)
    tokenizer = AutoTokenizer.from_pretrained(SMOL if adapter else source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_source = SMOL if adapter else source
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, source)
    model.eval()
    model.config.use_cache = True
    if sanity_only:
        for row in rows[:3]:
            print("\nPROMPT:", row["instruction"])
            print("RESPONSE:", generate_one(model, tokenizer, row["instruction"], chat, max_new_tokens, prompt_limit))
    else:
        dest = Path(dest)
        config = {"generator": name, "source": source, "checkpoint_sha256_or_revision": checkpoint_fingerprint(source),
                  "eval_sha256": sha(rows_path), "seed":42, "do_sample":False, "num_beams":1,
                  "max_new_tokens":max_new_tokens, "max_prompt_tokens":prompt_limit,
                  "temperature":None, "top_p":None, "repetition_penalty":1.1, "no_repeat_ngram_size":4,
                  "length_penalty":1.0, "chat_template":tokenizer.chat_template if chat else "explicit Instruction/Response"}
        config_path = dest.with_suffix(".config.json")
        if config_path.exists():
            assert read_json(config_path) == config, "Generation settings or checkpoint changed; use a new output directory"
        else:
            write_json(config_path, config)
        old = read_jsonl(dest.with_suffix(".jsonl"))
        seen = {r["id"] for r in old}
        assert len(seen) == len(old), "Duplicate generation ids"
        for row in tqdm(rows, desc=name):
            if row["id"] in seen:
                continue
            result = dict(row, generator=name, output=generate_one(model, tokenizer, row["instruction"], chat, max_new_tokens, prompt_limit))
            append_jsonl(dest.with_suffix(".jsonl"), result)
        saved = {r["id"]:r for r in read_jsonl(dest.with_suffix(".jsonl"))}
        assert set(saved) == {r["id"] for r in rows}
        write_json(dest.with_suffix(".json"), [saved[r["id"]] for r in rows])
    del model
    gc.collect()
    torch.cuda.empty_cache()


def build_pairs(baseline_path, candidate_path, out_path, track):
    bc = read_json(Path(baseline_path).with_suffix(".config.json"))
    cc = read_json(Path(candidate_path).with_suffix(".config.json"))
    for key in ["eval_sha256", "seed", "do_sample", "num_beams", "max_new_tokens",
                "max_prompt_tokens", "temperature", "top_p", "repetition_penalty",
                "length_penalty", "no_repeat_ngram_size", "chat_template"]:
        assert bc[key] == cc[key], f"Unfair comparison: generation {key} differs"
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    by_id = {r["id"]:r for r in candidate}
    assert len(by_id) == len(candidate) and set(by_id) == {r["id"] for r in baseline}
    rng = random.Random(42)
    pairs = []
    for b in baseline:
        c = by_id[b["id"]]
        assert c["instruction"] == b["instruction"]
        first, second = (c,b) if rng.random() < .5 else (b,c)
        pairs.append({"id": f"{track}:{b['id']}", "instruction": b["instruction"],
                      "response_a": first["output"], "response_b": second["output"],
                      "category": b.get("category", "general"), "track": track,
                      "_hidden": {"A": first["generator"], "B": second["generator"],
                                  "candidate": c["generator"], "baseline": b["generator"]}})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in pairs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    p.add_argument("--models-root", default=None, help="Read-only attached training outputs; generations go to --out")
    p.add_argument("--data", default=None)
    p.add_argument("--stage", choices=["baseline", "aligned", "persona", "all"], default="all")
    p.add_argument("--sanity-only", action="store_true", help="Inspect 3 responses per checkpoint BEFORE full generation")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=256)
    args = p.parse_args()
    out = Path(args.out)
    models = Path(args.models_root) if args.models_root else out
    data = Path(args.data) if args.data else out/"prepared"
    ev = out/"eval"
    ev.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.stage in ["baseline", "all"]:
        jobs.extend([(BASE, "base", "alpaca", "alpaca_eval", False, False),
                     (BASE, "base", "hh", "hh_eval", False, False)])
    if args.stage in ["aligned", "all"]:
        jobs.extend([(str(models/"sft"/"best"), "sft", "alpaca", "alpaca_eval", False, False),
                     (str(models/"dpo_uf"/"best"), "dpo_uf", "alpaca", "alpaca_eval", False, False),
                     (str(models/"sft"/"best"), "sft", "hh", "hh_eval", False, False),
                     (str(models/"dpo_hh"/"best"), "dpo_hh", "hh", "hh_eval", False, False)])
    if args.stage in ["persona", "all"]:
        jobs.extend([(SMOL, "smol_base", "persona", "persona_eval", True, False),
                     (str(models/"persona"/"best"), "persona_lora", "persona", "persona_eval", True, True)])
    for source, name, tag, subset, chat, adapter in jobs:
        generate_set(source, name, data/f"{subset}.json", ev/f"{name}_{tag}.json", chat, adapter,
                     args.sanity_only, args.max_new_tokens, args.max_prompt_tokens)
    if not args.sanity_only:
        for tag, baseline, candidate in [("alpaca", "sft", "dpo_uf"), ("hh", "sft", "dpo_hh"),
                                         ("persona", "smol_base", "persona_lora")]:
            bp, cp = ev/f"{baseline}_{tag}.json", ev/f"{candidate}_{tag}.json"
            if bp.exists() and cp.exists():
                build_pairs(bp, cp, ev/f"{tag}_pairs.jsonl", tag)


if __name__ == "__main__":
    main()
