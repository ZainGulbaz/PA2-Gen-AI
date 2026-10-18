# PA2: Small-Model Alignment (CODE IS IN MASTER BRANCH)

This repository is the landing page for the completed SFT, two DPO tracks, persona LoRA, generation, and blinded evaluation workflow. The source code, plots, judged outputs, and seven-page report are packaged in the saved Kaggle notebook output linked below. Model checkpoints remain in Kaggle; they are intentionally omitted from the small source/results archive.

## Download the project bundle

1. Open the [saved judging notebook output](https://www.kaggle.com/code/zaingulbaz/pa2-saved-output-judging/output?scriptVersionId=359772410).
2. In its Output tab, download `PA2-submission.zip`.
3. Extract the archive to review the implementation, tests, notebooks, measured outputs, plots, and `outputs/report_final.pdf`.

The training and generation notebook is [here](https://www.kaggle.com/code/zaingulbaz/notebook177e02c542). Its saved Version 5 contains the model checkpoints and generated responses.

## Measured results

All 616 blind comparisons completed with zero unresolved pairs. Candidate means DPO policy for the first two rows and persona LoRA for the last three.

| Comparison | N | Wins | Losses | Ties | Candidate win rate (ties half) |
|---|---:|---:|---:|---:|---:|
| UltraFeedback DPO vs SFT, Alpaca prompts | 300 | 90 | 150 | 60 | 40.0% |
| HH DPO vs SFT, HH prompts | 300 | 87 | 151 | 62 | 39.3% |
| Persona LoRA vs SmolLM2 base, style | 16 | 1 | 3 | 12 | 43.8% |
| Persona LoRA vs SmolLM2 base, helpfulness | 16 | 1 | 3 | 12 | 43.8% |
| Persona LoRA vs SmolLM2 base, safety | 16 | 1 | 0 | 15 | 53.1% |

The two DPO candidates lost more often than they won on their 300-prompt comparisons. Persona results are a small diagnostic. These are single-seed results from one LLM judge, not ground truth. The report documents the shared deterministic decoding adjustment made after sanity generations repeated phrases.

## Reproducibility

The bundle records run configuration, data manifests, per-stage metrics, generations, randomized blinded pairs, judge summaries and retries, and representative examples. Read the report's limitations and review the full responses before drawing conclusions. AI assistance contributed to implementation and analysis; follow the course disclosure policy.
