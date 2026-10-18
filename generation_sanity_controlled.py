import json
from pathlib import Path
from alignment.generate import generate_set
plan = json.loads(Path("generation_plan.json").read_text())
seen = set()
for source, name, tag, subset, chat, adapter in plan["jobs"]:
    if source in seen:
        continue
    seen.add(source)
    print("\n=== CONTROLLED SANITY", name, tag, "===", flush=True)
    generate_set(source, name, Path(plan["out"])/"sanity_inputs"/(subset+".json"),
                 Path(plan["out"])/"eval"/(name+"_"+tag+".json"),
                 chat, adapter, True,
                 plan["max_new_tokens"], plan["max_prompt_tokens"])
print("CONTROLLED SANITY FINISHED", flush=True)
