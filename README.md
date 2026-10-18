# PA2: Small-Model Alignment Experiments

This repository contains the implementation, Kaggle notebooks, completed SFT/DPO/persona runs, fixed-set generations, blinded judgments, plots, and final report for the **Instruction Tuning and Direct Preference Optimization on a Small Language Model** assignment. Large model checkpoints remain in the saved Kaggle outputs and are intentionally excluded from the lightweight repository package. See `outputs/report_final.pdf` for the measured results and limitations.

## Start here

1. Download and extract this package. Create your GitHub repository (fork the course starter or upload these files). Keep your actual repository URL for the final report.
2. Import `notebooks/01_train_kaggle.ipynb` into Kaggle. Enable **Internet** and **GPU T4 x2**. The notebook is self-contained: its first cell restores the source, starter tests, persona pairs, and fixed HH prompts into `/kaggle/working/PA2_End_to_End`.
3. Run cells in order. Inspect each 3-prompt sanity check before running its full-generation cell. Defaults are the assignment's 30,000 SFT examples, two separate 15,000-pair DPO runs, and 1,000 validation examples for each. `SMOKE=True` is only for testing the pipeline; it cannot produce a final submission.
4. Use **Save Version** once training is complete so checkpoints and logs are preserved. If you finish stages in separate sessions, save each stage's outputs before leaving.
5. Import `notebooks/02_generate_kaggle.ipynb`. Attach the saved training notebook outputs as an input dataset. Enable Internet and a GPU. The notebook finds the attached `outputs` directory (or asks you to set the path if several exist), reads checkpoints from the input, and writes new generations under `/kaggle/working`. Save Version after generation.
6. Create a Kaggle **utility benchmark/model-enabled notebook**, for example through https://www.kaggle.com/benchmarks/tasks/new. Import or paste cells from `notebooks/03_judge_report_kaggle.ipynb` while retaining benchmark access. Attach the saved generation outputs. Ordinary GPU notebooks may lack the Kaggle model proxy.
7. Select the supported judge model. The assignment suggests `google/gemini-3.6-flash`; if that ID is unavailable, explicitly select another supported model from your environment. The actual chosen name is recorded. Run all three comparisons, then make plots and the report draft.
8. Review full examples and fill your interpretation notes. Supply your real repository URL and use `alignment.report --final --notes ...` to produce a report of at most eight pages. Visually inspect its PDF before submission. Build the submission archive with the final notebook cell.

The training notebook can take an entire Kaggle session. The assignment estimates roughly 6.5-8.5 hours on T4 x2; this implementation's exact runtime is not benchmarked. Do not start all expensive stages without a session/quota plan.

## What is implemented

| Requirement | Location |
|---|---|
| Own SFT loop, response-only labels, EOS, padding, AMP, clipping | `alignment/core.py`, `alignment/train.py` |
| Own HH and UltraFeedback DPO loops, frozen SFT reference, summed response scores | Same shared core and loop; `--stage hh` / `--stage uf` |
| SFT best/final by validation loss; DPO best/final by strict reward accuracy | `outputs/{sft,dpo_hh,dpo_uf}` |
| Staff's exact adapter signatures | `submission_adapter.py` |
| Staff's original offline tests and additional edge/math checks | `tests/` |
| TRL DPOTrainer and LoRA, used only for the persona task | `alignment/persona.py` |
| Fixed subsets, source revisions, hashes, counts, seed42 | `alignment/prepare_data.py`, `outputs/prepared/manifest.json` |
| Base, SFT, HH DPO, UF DPO, Smol baseline, persona generations | `alignment/generate.py` |
| Blinded A/B comparisons, order randomization, identical decoding | `outputs/eval/*_pairs.jsonl` |
| Five-pair requests, 5s spacing, 3 attempts, batch checkpoints, resume | `alignment/judge.py` |
| W/L/T/unknown/failure rates, retry counts; separate persona dimensions | `outputs/eval/*_judged.summary.json` |
| Actual training plots, report draft/final, submission archive | `alignment/plots.py`, `alignment/report.py`, `build_submission.py` |

## Commands from the project root

Kaggle has a CUDA-compatible PyTorch build. Do not reinstall it. The pinned Transformers/TRL/PEFT stack stays below Transformers 5, as in the manual.

```bash
python -m pip uninstall -y -q torchao
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps
PYTHONPATH=. python -m pytest -q
python -m alignment.prepare_data

# Inspect these before full generation.
python -m alignment.generate --stage baseline --sanity-only
python -m alignment.generate --stage baseline
python -m alignment.train --stage sft
python -m alignment.train --stage hh
python -m alignment.train --stage uf
CUDA_VISIBLE_DEVICES=0 python -m alignment.persona
python -m alignment.plots

# Separate notebook, attached training output root may be read-only.
python -m alignment.generate --models-root /kaggle/input/REPLACE/outputs \
  --data /kaggle/input/REPLACE/outputs/prepared --out outputs --stage aligned --sanity-only
python -m alignment.generate --models-root /kaggle/input/REPLACE/outputs \
  --data /kaggle/input/REPLACE/outputs/prepared --out outputs --stage all

# Run from a Kaggle benchmark/model-enabled environment.
python -m alignment.judge --pairs outputs/eval/alpaca_pairs.jsonl \
  --output outputs/eval/alpaca_judged.jsonl --rubric general
python -m alignment.judge --pairs outputs/eval/hh_pairs.jsonl \
  --output outputs/eval/hh_judged.jsonl --rubric safety
python -m alignment.judge --pairs outputs/eval/persona_pairs.jsonl \
  --output outputs/eval/persona_judged.jsonl --rubric persona

python -m alignment.report --repo-url https://github.com/YOUR_ACCOUNT/YOUR_REPO
# After reviewing evidence and writing your notes:
python -m alignment.report --repo-url https://github.com/YOUR_ACCOUNT/YOUR_REPO \
  --notes report_notes.json --final
python build_submission.py --repo-url https://github.com/YOUR_ACCOUNT/YOUR_REPO
```

The first uninstall command can return a harmless nonzero status if torchao is absent; the notebook handles that. Use an actual repository URL in report/build commands. Placeholder URLs are rejected.

## Experimental details and explicit choices

- **SFT:** full parameters; batch2, accumulation16 (effective32); AdamW peak2e-5; 3% warmup rounded upward to optimizer groups; cosine decay; decay0.1; clip1; one epoch; sequence512.
- **HH and UF DPO:** each starts from SFT **best** independently; batch1, accumulation32; AdamW1e-6 constant; decay0; beta0.1; one epoch; sequence512. Policy is on GPU0; frozen reference on GPU1. Both have FP32 weights and FP16 autocast. Keeping the reference FP32 avoids stored-weight rounding differences at initialization and is a choice to report. Dropout is disabled for DPO.
- **Masking/math:** tokenize prompt and response separately without automatic special tokens; append response EOS before truncation; shift labels for next-token prediction; score only response tokens. SFT uses a global response-token mean over each accumulated group. DPO uses per-response **sums**, not length averages. The final incomplete accumulation group is normalized by its actual size.
- **Prompting:** Qwen uses `### Instruction:\n{instruction}\n\n### Response:\n`. Training and generation retain the final256 prompt tokens, then truncate the full training sequence from the right to512. The adapter intentionally does not impose the training-specific256 prompt cap; it follows the staff contract's `max_length`. Smol uses its native chat template.
- **Data:** SFT keeps the first user/assistant exchange of UltraChat. HH and UF preference data filter multi-turn pairs, empty rows, duplicate selected prompts, and identical chosen/rejected answers. Validation uses the official test split and selected validation prompts are excluded from training. Source revisions and exact subsets are saved. Dataset contamination outside the selected splits is not audited.
- **AlpacaEval:** direct JSON download bypasses legacy Hugging Face dataset scripts. Shuffle with `datasets.Dataset.shuffle(seed=42)` and take300; preserve that exact subset. All compared checkpoints use the same prompts and decoding settings.
- **Persona:** provided220 pairs ->198 train/22 validation, seed42; LoRA r16/alpha32/dropout0.05, q_proj/v_proj, LR5e-5, batch2, accumulation8, three epochs. These are justified starting choices for the small subset, not proven optimal. Report observed overfit/style/helpfulness/safety behavior.
- **Generation:** greedy, max256 new tokens, repetition penalty1, one beam, seed42; no temperature/top-p sampling. Change settings only consistently across a comparison and document the change. Generation configuration hashes prevent resuming into a changed checkpoint or decoding policy.
- **Judging:** judge-facing input contains only numeric item IDs, instructions, Response A, Response B. `_hidden` mappings are never sent. Overall persona judgment targets style; helpfulness and safety judgments are separate. Unknown/failed judgments stay out of valid win-rate denominators and are reported separately. The judge is not ground truth.

## Resume and storage

- Completed training stages are skipped. Default training preserves best/final weights and per-update logs without extra optimizer snapshots. An interrupted default run requires an explicit `--restart` for **that stage only**; completed stages need not rerun.
- Optional `--save-resume` writes full model/optimizer/scaler/RNG snapshots at validation boundaries. Use `--resume --save-resume` to resume them. This requires substantial extra temporary disk space while replacing snapshots; inspect Kaggle disk before enabling it. The default avoids this overhead. Persona can resume its TRL checkpoint via `--resume`.
- Generation checkpoints each prompt. Judgment checkpoints each batch. Rerunning reuses completed IDs and retries unresolved pairs, never completed calls. Reattach saved outputs if the session itself was replaced; saved files, not notebook connection state, determine what can resume.
- In the generation and judge notebooks, read-only input checkpoints stay attached. Small provenance/config/log files are copied into the new output so report/build stages can run without weights.
- GitHub should contain implementation, notebooks, source data/provenance, plots, report, and results. Keep large model checkpoints in saved Kaggle outputs or a model host. `build_submission.py` excludes weights by default; `--include-checkpoints` is optional and produces a large ZIP.

## Report notes

Write `report_notes.json` after inspecting actual results. Four nonempty string fields are required for final mode:

```json
{
  "parameter_choices": "Explain actual choices and deviations, distinguishing memory-driven changes from performance-driven changes.",
  "interpretation": "State which model won on each set, by how much, and give evidence-backed explanations rather than unsupported causes.",
  "error_analysis": "Discuss actual improvements and regressions with saved prompt IDs, including persona helpfulness and safety.",
  "limitations": "Explain judge bias, unknowns/failures, single-seed uncertainty, sample sizes, and compute constraints from your runs."
}
```

These example strings are instructions, not submission text. The code automatically supplies measured tables, configuration values, training plots, representative excerpts, and file paths; your observations complete the interpretation. Render/inspect the final PDF and verify <=8 pages. Submit the GitHub URL, final report PDF, training plots, and DPO-versus-SFT summary with the root adapter and passing public tests, as PA2 requires.

## Attribution and validation

Starter tests, fixed HH evaluation, and persona data come from the course repository at commit `7f2e6bc6a6f48f0843bda5e4e7be3b2151ce4a45`. The SFT/DPO core and loops here are newly written, not copied from a public DPO implementation. TRL is confined to the explicitly permitted persona task. Review, understand, adapt, and disclose outside/AI assistance according to course policy; the assignment says the implementation/report must be your own.

Documentation consulted:
- https://huggingface.co/docs/transformers/v4.56.2/en/index
- https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer
- https://huggingface.co/docs/peft/en/package_reference/lora
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md
- https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- https://huggingface.co/datasets/Anthropic/hh-rlhf
- https://huggingface.co/datasets/tatsu-lab/alpaca_eval

Local checks verify math, adapter behavior, judge reliability, and syntax. Full T4 training and authenticated Kaggle model calls still have to run in your account; local CPU tests do not establish final model quality or Kaggle API access.
