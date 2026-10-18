"""Task 5 ONLY: allowed use of TRL DPOTrainer and PEFT LoRA."""
import argparse
import gc
import shutil
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
from alignment.utils import read_json, write_json, append_jsonl, environment, seed_all

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def conversational(rows):
    return Dataset.from_list([{
        "prompt": [{"role": "user", "content": r["prompt"]}],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
    } for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/prepared")
    p.add_argument("--out", default="outputs/persona")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--restart", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    if (out/"complete.json").exists() and not args.restart:
        print("Persona already complete")
        return
    if args.resume and args.restart:
        raise ValueError("Choose resume OR restart")
    if args.restart and out.exists():
        shutil.rmtree(out)
    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints and not args.resume:
        raise RuntimeError("Persona interrupted; pass --resume or --restart")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run persona with CUDA_VISIBLE_DEVICES=0 to keep its effective batch size fixed")
    seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    train_rows = read_json(Path(args.data)/"persona_train.json")
    val_rows = read_json(Path(args.data)/"persona_val.json")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    cfg = DPOConfig(output_dir=str(out), learning_rate=5e-5, beta=0.1,
                    num_train_epochs=3, per_device_train_batch_size=2,
                    per_device_eval_batch_size=2, gradient_accumulation_steps=8,
                    max_length=512, max_prompt_length=256,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    fp16=True, bf16=False, eval_strategy="steps", eval_steps=10,
                    save_strategy="steps", save_steps=10, save_total_limit=2,
                    logging_steps=1, report_to="none", seed=42, data_seed=42,
                    optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
                    lr_scheduler_type="constant", warmup_ratio=0.0,
                    load_best_model_at_end=False, metric_for_best_model="rewards/accuracies",
                    greater_is_better=True, remove_unused_columns=False)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, peft_config=lora,
                         processing_class=tokenizer, train_dataset=conversational(train_rows),
                         eval_dataset=conversational(val_rows))
    trainer.model.print_trainable_parameters()
    run_config = {"model": MODEL, "reference": "frozen original SmolLM2 with adapters disabled",
                  "lora": lora.to_dict(), "training_arguments": cfg.to_dict(),
                  "train_examples": len(train_rows), "validation_examples": len(val_rows),
                  "environment": environment(), "prompt_format": "native tokenizer.chat_template",
                  "chat_template": tokenizer.chat_template,
                  "justification": "Rank 16/alpha 32 with q/v projections limits capacity for 198 examples; dropout 0.05 and held-out validation monitor overfit. LR 5e-5 and three epochs are starting choices, not an experimentally established optimum."}
    # to_dict can contain enums; normalize to JSON using its string representation.
    import json
    write_json(out/"config.json", json.loads(json.dumps(run_config, default=str)))
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if args.resume and checkpoints else None)
    # Save last policy before loading/copying the validation-selected adapter.
    trainer.save_model(str(out/"final"))
    tokenizer.save_pretrained(out/"final")
    trainer.evaluate()
    best = trainer.state.best_model_checkpoint
    if not best:
        raise RuntimeError("No validated persona checkpoint found; check eval/save interval")
    (out/"best").mkdir(parents=True, exist_ok=True)
    for name in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        src = Path(best)/name
        if src.exists():
            shutil.copy2(src, out/"best"/name)
    tokenizer.save_pretrained(out/"best")
    (out/"metrics.jsonl").unlink(missing_ok=True)
    for row in trainer.state.log_history:
        append_jsonl(out/"metrics.jsonl", row)
    write_json(out/"complete.json", {"best_checkpoint": best, "best_metric": trainer.state.best_metric,
                                   "global_step": trainer.state.global_step})
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("Saved persona best and final adapters", out)


if __name__ == "__main__":
    main()
