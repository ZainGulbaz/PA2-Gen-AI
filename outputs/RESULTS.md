# Measured evaluation summary

Blind LLM judgments used google/gemini-3.6-flash. All responses were generated and saved before judging. A/B order was randomized; no identity mapping was sent to the judge.

| Comparison | N | Candidate wins | Losses | Ties | Win rate, ties half |
|---|---:|---:|---:|---:|---:|
| UltraFeedback DPO vs SFT on Alpaca prompts | 300 | 90 | 150 | 60 | 40.0% |
| HH DPO vs SFT on HH prompts | 300 | 87 | 151 | 62 | 39.3% |
| Persona LoRA vs SmolLM2 base, style | 16 | 1 | 3 | 12 | 43.8% |
| Persona LoRA vs SmolLM2 base, helpfulness | 16 | 1 | 3 | 12 | 43.8% |
| Persona LoRA vs SmolLM2 base, safety | 16 | 1 | 0 | 15 | 53.1% |

No judged pair was unresolved. Invalid judge request attempts (2 Alpaca, 5 HH) were recovered by retries. Both 300-pair DPO tracks had more losses than wins. The persona sample is too small for broad conclusions. These numbers are not ground truth: one LLM judge, one training seed, finite prompt sets, and a shared decoding control limit interpretation. See `report_final.pdf`, `report_notes.json`, and the full `eval/*_judged.jsonl` files for details and examples.
