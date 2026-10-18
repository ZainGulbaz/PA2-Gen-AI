import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from tests.toy_lm import ToyTokenizer, ToyCausalLM
from alignment import core


def test_eos_is_appended_before_truncation_and_can_be_removed():
    t = ToyTokenizer()
    b = core.build_sft_batch(t, ["abc"], ["xyz"], 5)
    assert b["input_ids"][0].tolist() == t.encode("abcxy")
    assert b["labels"][0].tolist() == [-100]*3 + t.encode("xy")
    assert t.eos_token_id not in b["labels"][0].tolist()


def test_prompt_consumes_window_and_all_masked_loss_has_valid_gradient():
    t, model = ToyTokenizer(), ToyCausalLM(ToyTokenizer().vocab_size)
    b = core.build_sft_batch(t, ["abcdef"], ["x"], 3)
    assert b["labels"].eq(-100).all()
    loss = core.compute_sft_loss(model, t, ["abcdef"], ["x"], 3)
    assert loss.item() == 0
    loss.backward()
    assert model.logit_bias.grad.eq(0).all()


def test_no_eos_no_pad_and_padding_does_not_mask_real_tokens():
    t = ToyTokenizer()
    t.eos_token_id = None
    t.pad_token_id = None
    b = core.build_sft_batch(t, ["ab", "a"], ["x", "y"], 8)
    assert b["input_ids"].tolist() == [t.encode("abx"), t.encode("ay")+[0]]
    assert b["attention_mask"].tolist() == [[1,1,1],[1,1,0]]
    assert b["labels"].tolist() == [[-100,-100,t.vocab["x"]],[-100,t.vocab["y"],-100]]


def test_pad_equals_eos_but_response_eos_remains_scored():
    t = ToyTokenizer()
    t.pad_token_id = t.eos_token_id
    b = core.build_sft_batch(t, ["a", "abc"], ["x", "y"], 10)
    assert b["labels"][0,2] == t.eos_token_id
    assert b["labels"][0,3:].eq(-100).all()
    lp, n = core.response_scores(ToyCausalLM(t.vocab_size), b)
    assert n.tolist() == [2,2]
    torch.testing.assert_close(lp, torch.full((2,), -2*math.log(t.vocab_size)))


def test_shifted_conditional_scores_and_row_order_against_independent_oracle():
    t = ToyTokenizer()
    class PositionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.arange(t.vocab_size**2, dtype=torch.float32).reshape(t.vocab_size,t.vocab_size)/300)
        def forward(self, input_ids, attention_mask=None):
            # Nonuniform input-dependent logits catch both shift and prompt-mask errors.
            return SimpleNamespace(logits=self.weights[input_ids])
    m = PositionModel()
    prompts, responses = ["a", "long: ", "z"], ["xy", "b", "ab"]
    b = core.build_sft_batch(t, prompts, responses, 12)
    logits = m(b["input_ids"]).logits
    expected = []
    for row in range(3):
        total = 0.
        for j in range(1,b["input_ids"].shape[1]):
            if b["labels"][row,j] != -100:
                total += F.log_softmax(logits[row,j-1],dim=-1)[b["labels"][row,j]]
        expected.append(total)
    actual = core.compute_response_logprobs(m,t,prompts,responses,12)
    torch.testing.assert_close(actual,torch.stack(expected))


def test_dpo_formula_direction_initial_strict_accuracy_and_optimizer_step():
    t = ToyTokenizer()
    policy, ref = ToyCausalLM(t.vocab_size), ToyCausalLM(t.vocab_size)
    batch = core.build_dpo_batch(t,["Pick: "],["a"],["b"],16)
    loss, margin, _, _ = core.dpo_batch_loss(policy,ref,batch,.1)
    assert float(loss) == float(torch.log(torch.tensor(2.)))
    assert int((margin>0).sum()) == 0
    optimizer = torch.optim.SGD(policy.parameters(),lr=.5)
    before = ref.logit_bias.detach().clone()
    loss.backward()
    optimizer.step()
    new_loss, new_margin, _, _ = core.dpo_batch_loss(policy,ref,batch,.1)
    assert new_loss < loss and new_margin > 0
    assert ref.logit_bias.grad is None
    torch.testing.assert_close(ref.logit_bias,before)


def test_positive_and_negative_dpo_gaps_match_formula():
    pc=torch.tensor([-2.,-7.],requires_grad=True)
    pr=torch.tensor([-5.,-3.],requires_grad=True)
    rc=torch.tensor([-4.,-4.])
    rr=torch.tensor([-4.,-4.])
    actual=core.compute_dpo_loss_from_logps(pc,pr,rc,rr,.2)
    expected=torch.log1p(torch.exp(-.2*torch.tensor([3.,-4.]))).mean()
    torch.testing.assert_close(actual,expected)


def test_fp16_logit_score_initializer_does_not_overflow():
    t=ToyTokenizer()
    class Big(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.full((t.vocab_size,),1000.,dtype=torch.float16))
        def forward(self,input_ids,attention_mask=None):
            return SimpleNamespace(logits=self.bias.view(1,1,-1).expand(*input_ids.shape,-1))
    loss=core.compute_sft_loss(Big(),t,["abc"],["defgh"],16)
    assert torch.isfinite(loss)
