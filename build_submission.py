"""Package source + measured artifacts after completing and reviewing the report."""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from pypdf import PdfReader


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs")
    p.add_argument("--repo-url",required=True)
    p.add_argument("--zip",default="PA2-submission.zip")
    p.add_argument("--include-checkpoints",action="store_true",help="Large optional archive; models need not be committed to GitHub")
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    out=Path(args.out).resolve()
    meta=json.loads((out/"report_metadata.json").read_text())
    if meta["is_draft"] or meta["repo_url"] != args.repo_url:
        raise ValueError("Create report_final.pdf using your interpretation notes and the actual repository URL first")
    assert len(PdfReader(out/"report_final.pdf").pages)<=8
    assert not json.loads((out/"prepared"/"manifest.json").read_text())["smoke"]
    for stage in ["sft","dpo_hh","dpo_uf","persona"]:
        assert (out/stage/"complete.json").exists(),f"Unfinished stage {stage}"
    for track in ["alpaca","hh","persona"]:
        assert (out/"eval"/f"{track}_judged.summary.json").exists(),f"Missing evaluation {track}"
    subprocess.run([sys.executable,"-m","pytest","-q","tests"],cwd=root,check=True)
    (out/"repository_url.txt").write_text(args.repo_url+"\n")
    dest=Path(args.zip).resolve()
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or out in path.parents or path==dest:
                continue
            if any(x in path.parts for x in [".git","__pycache__",".pytest_cache"]):
                continue
            if path.name.endswith(".zip") or path.suffix==".pyc":
                continue
            z.write(path,"PA2/"+str(path.relative_to(root)))
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.name=="report_draft.pdf":
                continue
            if not args.include_checkpoints and any(x in path.parts for x in ["best","final","resume"]):
                continue
            if any(x.startswith("checkpoint-") or x.endswith(".tmp") for x in path.parts):
                continue
            z.write(path,"PA2/outputs/"+str(path.relative_to(out)))
    print("Created",dest,"size MiB",round(dest.stat().st_size/2**20,2))
    print("Submit the GitHub URL and final report; preserve checkpoint outputs on Kaggle separately.")


if __name__=="__main__":
    main()
