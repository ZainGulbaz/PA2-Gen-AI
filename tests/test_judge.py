import json
from alignment.judge import judge_prompt, run_batches, summarize
from alignment.utils import read_jsonl


def pairs():
    return [{"id":f"pair-{i}","instruction":"Help me", "response_a":"A answer", "response_b":"B answer",
             "_hidden":{"A":"SECRET_DPO_MODEL","B":"SECRET_SFT_MODEL", "candidate":"SECRET_DPO_MODEL","baseline":"SECRET_SFT_MODEL"}}
            for i in range(3)]


def test_judge_prompt_does_not_leak_model_names_or_real_ids():
    prompt=judge_prompt(pairs(),"general")
    assert "SECRET_" not in prompt and "pair-" not in prompt and "_hidden" not in prompt
    assert "Response A" in prompt and "Response B" in prompt


def test_retry_checkpoint_resume_and_unknown_failure_accounting(tmp_path):
    input_path, output = tmp_path/"pairs.jsonl", tmp_path/"judged.jsonl"
    input_path.write_text("".join(json.dumps(r)+"\n" for r in pairs()))
    calls=[]
    def fake(prompt):
        calls.append(prompt)
        if len(calls)==1:
            return {"decisions":[]}  # Force structural retry.
        public=json.loads(prompt.split("\n",1)[1])
        return {"decisions":[{"id":r["id"],"winner":"A", "helpfulness_winner":"tie", "safety_winner":"unknown", "reason":"fixture"} for r in public]}
    delays=[]
    summary=run_batches(input_path,output,"fixture",fake,batch_size=2,sleeper=delays.append)
    assert len(read_jsonl(output))==3
    assert summary["overall"]["win"]==3
    assert summary["safety"]["unknown"]==3
    assert summary["request_attempts"]==3 and summary["request_retry_attempts"]==1
    assert delays==[5,5,5]
    before=len(calls)
    run_batches(input_path,output,"fixture",fake,batch_size=2,sleeper=delays.append)
    assert len(calls)==before
    counts=summarize(pairs(),read_jsonl(output)[:2],[{"id":"pair-2"}])
    assert counts["overall"]["failure"]==1


def test_failed_requests_are_logged_and_retry_on_next_run(tmp_path):
    source,target=tmp_path/"pairs.jsonl",tmp_path/"judge.jsonl"
    source.write_text(json.dumps(pairs()[0])+"\n")
    def fail(prompt):
        raise RuntimeError("fixture backend unavailable")
    result=run_batches(source,target,"fixture",fail,max_retries=3,sleeper=lambda _:None)
    assert result["overall"]["failure"]==1
    assert len(read_jsonl(target.with_suffix(".attempts.jsonl")))==3
    def good(prompt):
        return {"decisions":[{"id":"0","winner":"tie", "helpfulness_winner":"tie", "safety_winner":"tie", "reason":"fixture"}]}
    result=run_batches(source,target,"fixture",good,max_retries=3,sleeper=lambda _:None)
    assert result["overall"]["failure"]==0 and result["overall"]["tie"]==1
