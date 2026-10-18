import json
from pathlib import Path
from alignment.generate import generate_set
plan = json.loads(Path("generation_plan.json").read_text())
for source, name, tag, subset, chat, adapter in plan["jobs"]:
    print("\n=== SANITY", name, tag, "===", flush=True)
    generate_set(source, name, Path(plan["data"])/(subset+".json"),
                 Path(plan["out"])/"eval"/(name+"_"+tag+".json"),
                 chat, adapter, True,
                 plan["max_new_tokens"], plan["max_prompt_tokens"])
print("SANITY CHECKS FINISHED - REVIEW RESPONSES BEFORE FULL GENERATION", flush=True)
