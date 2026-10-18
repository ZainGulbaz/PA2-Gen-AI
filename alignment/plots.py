import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from alignment.utils import read_jsonl


def plot_run(run, dest):
    rows = read_jsonl(run/"metrics.jsonl")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3 if run.name.startswith("dpo") else 1,
                             figsize=(12,3.6) if run.name.startswith("dpo") else (7,3.6), squeeze=False)
    ax = axes[0,0]
    train = [r for r in rows if r.get("kind")=="train"]
    val = [r for r in rows if r.get("kind")=="validation"]
    if run.name == "persona":
        train = [r for r in rows if "loss" in r]
        val = [r for r in rows if "eval_loss" in r]
        ax.plot([r["step"] for r in train],[r["loss"] for r in train],alpha=.6,label="train, raw")
        ax.plot([r["step"] for r in val],[r["eval_loss"] for r in val],"o-",label="validation")
    else:
        ax.plot([r["step"] for r in train],[r["loss"] for r in train],alpha=.6,label="train, raw")
        ax.plot([r["step"] for r in val],[r["val_loss"] for r in val],"o-",label="validation")
    ax.set_ylabel("Response cross entropy" if run.name=="sft" else "DPO loss")
    if run.name.startswith("dpo"):
        for ax, field, title in zip(axes[0,1:], ["reward_accuracy","reward_margin"],["Reward accuracy (strict >)","Reward margin"]):
            ax.plot([r["step"] for r in train],[r[field] for r in train],alpha=.4,label="train, raw")
            ax.plot([r["step"] for r in val],[r[field] for r in val],"o-",label="validation")
            ax.set_ylabel(title)
    for ax in axes[0]:
        ax.set_xlabel("Optimizer group")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(run.name + " (no smoothing)")
    fig.tight_layout()
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f"{run.name}.png",dpi=180)
    fig.savefig(dest/f"{run.name}.pdf")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs")
    args=p.parse_args()
    out=Path(args.out)
    for stage in ["sft","dpo_hh","dpo_uf","persona"]:
        plot_run(out/stage,out/"plots")


if __name__=="__main__":
    main()
