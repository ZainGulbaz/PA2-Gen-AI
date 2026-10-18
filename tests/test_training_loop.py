"""Exercise the real loops/save/resume on tiny offline CPU doubles."""
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
pytest.importorskip("transformers")
from alignment import train
from alignment.utils import write_json, read_json, read_jsonl
from tests.toy_lm import ToyTokenizer


class Tokenizer(ToyTokenizer):
    def __init__(self):
        super().__init__()
        for char in "#\n":
            self.vocab[char]=len(self.vocab)
    def save_pretrained(self,path):
        write_json(Path(path)/"toy_tokenizer.json",self.vocab)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(Tokenizer().vocab_size,8)
        self.head=torch.nn.Linear(8,Tokenizer().vocab_size)
        self.config=SimpleNamespace(use_cache=False)
    def forward(self,input_ids,attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
    def save_pretrained(self,path,safe_serialization=True):
        torch.save(self.state_dict(),Path(path)/"toy_weights.pt")


@pytest.fixture
def tiny_training(monkeypatch,tmp_path):
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    data=tmp_path/"data"
    write_json(data/"manifest.json",{"fixture":True})
    for stage in ["sft","hh","uf"]:
        rows=[{"prompt":f"Say {i}","response":"abc","chosen":"abc","rejected":"xyz"} for i in range(12)]
        write_json(data/f"{stage}_train.json",rows)
        write_json(data/f"{stage}_val.json",rows[:3])
    monkeypatch.setattr(train.AutoTokenizer,"from_pretrained",lambda _:Tokenizer())
    def load(path,device,trainable):
        torch.manual_seed(101)
        m=Model().to(device)
        file=Path(path)/"toy_weights.pt"
        if file.exists():
            m.load_state_dict(torch.load(file,weights_only=True))
        if not trainable:
            m.requires_grad_(False);m.eval()
        return m
    monkeypatch.setattr(train,"load_model",load)
    monkeypatch.setattr(train,"environment",lambda:{"fixture":True})
    def execute(out,stage="sft",extras=()):
        argv=["train","--stage",stage,"--data",str(data),"--out",str(out),"--device","cpu",
              "--reference-device","cpu","--batch-size","2","--accumulation","2","--lr","0.01",
              "--max-length","64","--max-prompt-length","32","--eval-every","1",*extras]
        monkeypatch.setattr(sys,"argv",argv)
        train.main()
    yield execute
    torch.set_num_threads(threads)


@pytest.mark.parametrize("stage",["sft","hh","uf"])
def test_real_training_loop_best_final_and_completed_stage_skip(tiny_training,tmp_path,stage):
    out=tmp_path/"out"
    if stage!="sft":
        tiny_training(out,"sft")
    tiny_training(out,stage)
    run=out/("sft" if stage=="sft" else "dpo_"+stage)
    assert (run/"best/toy_weights.pt").exists()
    assert (run/"final/toy_weights.pt").exists()
    assert read_json(run/"complete.json")["successful_updates"]==3
    before=read_jsonl(run/"metrics.jsonl")
    tiny_training(out,stage)
    assert read_jsonl(run/"metrics.jsonl")==before


def test_exact_training_resume_matches_uninterrupted_cpu_run(tiny_training,tmp_path,monkeypatch):
    full, interrupted=tmp_path/"full",tmp_path/"interrupted"
    tiny_training(full)
    original=train.save_checkpoint
    def crash_once(model,tokenizer,path,config,state=None):
        original(model,tokenizer,path,config,state)
        if Path(path).name=="resume":
            raise RuntimeError("simulated disconnection")
    monkeypatch.setattr(train,"save_checkpoint",crash_once)
    with pytest.raises(RuntimeError,match="simulated disconnection"):
        tiny_training(interrupted,extras=["--save-resume"])
    monkeypatch.setattr(train,"save_checkpoint",original)
    tiny_training(interrupted,extras=["--save-resume","--resume"])
    expected=torch.load(full/"sft/final/toy_weights.pt",weights_only=True)
    actual=torch.load(interrupted/"sft/final/toy_weights.pt",weights_only=True)
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
