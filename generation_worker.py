import json, sys
from pathlib import Path
from alignment.generate import generate_set
plan = json.loads(Path("generation_plan.json").read_text())
worker_id = int(sys.argv[1])
indices = [[0,2,3,7], [1,4,5,6]][worker_id]
for index in indices:
    source, name, tag, subset, chat, adapter = plan["jobs"][index]
    print("BEGIN MODEL", name, tag, flush=True)
    generate_set(source, name, Path(plan["data"])/(subset+".json"),
                 Path(plan["out"])/"eval"/(name+"_"+tag+".json"),
                 chat, adapter, False,
                 plan["max_new_tokens"], plan["max_prompt_tokens"])
    print("FINISHED MODEL", name, tag, flush=True)
Path(plan["out"]).joinpath("generation_worker_"+str(worker_id)+".done").write_text("complete")
print("WORKER COMPLETE", worker_id, flush=True)
