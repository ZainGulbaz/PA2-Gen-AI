"""Build a concise evidence-based draft from actual runs; no invented results."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from pypdf import PdfReader
from alignment.utils import read_json, read_jsonl, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs")
    p.add_argument("--repo-url",required=True)
    p.add_argument("--notes",default=None,help="Optional JSON with parameter_choices, interpretation, error_analysis, limitations")
    p.add_argument("--final",action="store_true",help="Use your completed interpretation notes and produce report_final.pdf")
    args=p.parse_args()
    if "YOUR" in args.repo_url or not args.repo_url.startswith("https://github.com/"):
        raise ValueError("Supply the actual GitHub repository URL")
    out=Path(args.out)
    configs={stage:read_json(out/stage/"config.json") for stage in ["sft","dpo_hh","dpo_uf","persona"]}
    summaries={track:read_json(out/"eval"/f"{track}_judged.summary.json") for track in ["alpaca","hh","persona"]}
    manifest=read_json(out/"prepared"/"manifest.json")
    if manifest["smoke"]:
        raise ValueError("Smoke runs are not final assignment experiments")
    notes=read_json(args.notes) if args.notes else {}
    if args.final and any(not str(notes.get(k,"")).strip() for k in ["parameter_choices","interpretation","error_analysis","limitations"]):
        raise ValueError("For the final report, supply your own completed parameter_choices, interpretation, error_analysis, and limitations notes")
    styles=getSampleStyleSheet()
    # Embed portable fonts shipped with ReportLab instead of relying on PDF viewer substitution.
    import reportlab
    fonts=Path(reportlab.__file__).parent/"fonts"
    for name,file in [("Vera","Vera.ttf"),("Vera-Bold","VeraBd.ttf"),
                      ("Vera-Italic","VeraIt.ttf"),("Vera-BoldItalic","VeraBI.ttf")]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily("Vera",normal="Vera",bold="Vera-Bold",italic="Vera-Italic",boldItalic="Vera-BoldItalic")
    for style in styles.byName.values():
        style.fontName="Vera-Bold" if style.name.startswith("Heading") else "Vera"
    styles["BodyText"].fontSize=9
    styles["BodyText"].leading=12
    styles["Heading2"].fontSize=12
    story=[]
    def para(text,style="BodyText"):
        story.append(Paragraph(escape(str(text)).replace("\n","<br/>"),styles[style]))
        story.append(Spacer(1,5))
    def table(rows,widths=None):
        data=[[Paragraph(escape(str(c)),styles["BodyText"]) for c in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e9eff7")),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.4,colors.lightgrey),
                               ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(t)
        story.append(Spacer(1,8))
    para("PA2: Small-model SFT and DPO", "Title")
    para("Post-training comparison of instruction tuning, helpful/harmless preference optimization, general preference optimization, and persona LoRA." if args.final else "Evidence-based report draft. Review the generated examples and add your own experimentally grounded interpretation before submission.")
    para("Repository, models, and data", "Heading2")
    para(args.repo_url)
    para("Qwen/Qwen2.5-0.5B is the base for full-parameter SFT, HH DPO, and UltraFeedback DPO. Each DPO policy and frozen reference start from SFT best. The persona track uses HuggingFaceTB/SmolLM2-360M-Instruct with a frozen, adapter-disabled base reference. Seed: 42.")
    rows=[["Track","Train / validation","Data and fixed evaluation"]]
    for stage in ["sft","dpo_hh","dpo_uf","persona"]:
        c=configs[stage]
        data={"sft":"UltraChat first user/assistant exchange; 300 AlpacaEval prompts",
              "dpo_hh":"Single-turn HH-RLHF; provided 300 synthetic HH prompts",
              "dpo_uf":"Single-turn UltraFeedback; same 300 AlpacaEval prompts",
              "persona":"Provided 220 persona pairs, 90/10 split; 16 fixed style/help/safety prompts"}[stage]
        rows.append([stage,f"{c['train_examples']} / {c['validation_examples']}",data])
    table(rows,[75,95,335])
    para("Experimental configuration", "Heading2")
    rows=[["Track","Strategy / batch / precision","Optimizer / length / beta"]]
    for stage,c in configs.items():
        if stage=="persona":
            a=c["training_arguments"]
            rows.append([stage,"LoRA r16, alpha32, dropout0.05, q_proj/v_proj; batch2 x accumulation8; FP32 weights/FP16 AMP",
                         f"AdamW {a['learning_rate']}; constant LR; warmup0; decay0; clip1; length512; 3 epochs; beta0.1"])
        else:
            rows.append([stage,f"Full; batch{c['batch_size']} x accumulation{c['accumulation']}; FP32 weights/FP16 AMP",
                         f"AdamW {c['lr']}; {c['schedule']}; warmup {c['warmup_steps']} steps; decay{c['weight_decay']}; clip1; length{c['max_length']}; {c['epochs']} epoch; beta {c['beta'] if stage!='sft' else 'N/A'}"])
    table(rows,[65,225,215])
    for stage in ["dpo_hh","dpo_uf"]:
        para(f"{stage} exact frozen reference: {configs[stage]['reference_checkpoint']}.")
    para("Persona exact reference: "+configs["persona"]["reference"]+".")
    para("Prompt/pad labels are -100; response EOS precedes right truncation. Qwen retains the last256 prompt tokens consistently. SFT uses a response-token mean over each accumulated group; DPO sums response log probabilities. Best checkpoints minimize SFT validation loss or maximize strict DPO reward accuracy.")
    para("Parameter choices and deviations", "Heading2")
    para(notes.get("parameter_choices","Main tracks use the guide recipe. AdamW is selected; the reference stays FP32 on GPU1 under FP16 autocast to avoid stored-weight rounding differences. Qwen uses an explicit prompt; LoRA uses rank16 q/v targets and a 22-pair holdout. These are starting settings; no hyperparameter search is claimed."))
    story.append(PageBreak())
    para("Training behavior", "Heading2")
    for stage in ["sft","dpo_hh","dpo_uf","persona"]:
        block_start=len(story)
        complete=read_json(out/stage/"complete.json")
        best_path=out/stage/"best_metrics.json"
        best=read_json(best_path) if best_path.exists() else {"best_metric":complete.get("best_metric")}
        para(f"{stage}: selected validation metrics {best}. Skipped AMP updates: {complete.get('skipped_updates','see TRL logs')}.")
        image=out/"plots"/f"{stage}.png"
        if image.exists():
            from PIL import Image as PILImage
            with PILImage.open(image) as im:
                width,height=im.size
            story.append(Image(str(image),width=480,height=480*height/width))
        block=story[block_start:]
        story[block_start:]=[KeepTogether(block)]
    para("Plots show raw measurements without smoothing. Training loss alone does not establish generation quality. Preference reward accuracy is based on chosen reward > rejected reward; exact initial ties count as incorrect, not as 0.5.")
    story.append(PageBreak())
    para("DPO versus SFT results", "Heading2")
    rows=[["Evaluation","N / W / T / L","Win rate: no ties / half ties","Unknown / failed"]]
    for track in ["alpaca","hh"]:
        s=summaries[track]; r=s["overall"]
        def percent(x): return "N/A" if x is None else f"{100*x:.1f}%"
        rows.append([track,f"{s['total_pairs']} / {r['win']} / {r['tie']} / {r['loss']}",
                     f"{percent(r['win_rate_excluding_ties'])} / {percent(r['win_rate_ties_half'])}",f"{r['unknown']} / {r['failure']}"])
        para(f"{track}: judge {s['judge_model']}, request attempts {s.get('request_attempts',0)}, retried attempts {s.get('request_retry_attempts',0)}, invalid/failed requests {s.get('invalid_or_failed_request_attempts',0)}.")
    table(rows,[75,135,175,120])
    para("Model outputs were saved before judging. Response order was randomized with seed42, and A/B identities were withheld from the judge. Each request contained at most five pairs, with at least five seconds between attempts and up to three attempts. Unknown judgments and unresolved failures are excluded from win-rate denominators and reported separately over all prompts. Ties receive half credit only in the explicitly labeled half-tie rate.")
    para("Persona LoRA results (separate target)","Heading2")
    s=summaries["persona"]
    rows=[["Dimension","Win / tie / loss","Unknown / failed","Half-tie win rate"]]
    for dim in ["overall","helpfulness","safety"]:
        r=s[dim]
        rows.append(["persona style" if dim=="overall" else dim, f"{r['win']} / {r['tie']} / {r['loss']}",f"{r['unknown']} / {r['failure']}",percent(r['win_rate_ties_half'])])
    table(rows,[100,135,130,140])
    para("The persona baseline is unmodified SmolLM2-Instruct, not Qwen SFT. Style, helpfulness, and safety are judged separately on the same 16 prompts. This small diagnostic is not combined with the AlpacaEval or HH headline results.")
    para("Interpretation and error analysis","Heading2")
    for track in ["alpaca","hh"]:
        r=summaries[track]["overall"]
        difference=r["win"]-r["loss"]
        para(f"Measured {track} outcome: DPO {'won more' if difference>0 else 'lost more' if difference<0 else 'won and lost equally'}; wins minus losses = {difference}. This observation does not identify the cause of the difference.")
    para(notes.get("interpretation","Inspect the examples and category-specific failures before attributing gains or regressions to dataset coverage, safety refusals, verbosity, factuality, or over-optimization. Connect any explanation to saved generations and reward diagnostics. These experiments use a single seed and do not establish statistical significance."))
    story.append(PageBreak())
    para("Representative examples", "Heading2")
    def excerpt(text,limit):
        return " ".join(str(text).split())[:limit]
    for track in ["alpaca","hh","persona"]:
        judged=read_jsonl(out/"eval"/f"{track}_judged.jsonl")
        selected=[]
        for wanted in ["win","loss","tie"]:
            match=next((r for r in judged if r["candidate_outcome"]==wanted),None)
            if match:
                selected.append(match)
            if len(selected)==2:
                break
        for r in selected:
            block_start=len(story)
            hidden=r["_hidden"]
            cand="response_a" if hidden["A"]==hidden["candidate"] else "response_b"
            base="response_b" if cand=="response_a" else "response_a"
            para(f"{track} / {r['candidate_outcome']} / {r['id']}","Heading3")
            para("Prompt: "+excerpt(r["instruction"],160))
            para("Baseline excerpt: "+excerpt(r[base],220))
            para("Candidate excerpt: "+excerpt(r[cand],220))
            para("Judge rationale: "+excerpt(r["reason"],180))
            block=story[block_start:]
            story[block_start:]=[KeepTogether(block)]
    para(notes.get("error_analysis","Excerpts are truncated for space; use the full saved judged JSONL files for analysis. A judge rationale is an assessment, not independently verified ground truth."))
    story.append(PageBreak())
    para("Limitations and reproducibility", "Heading2")
    para(notes.get("limitations","Limitations include judge bias and position sensitivity, a single training seed, synthetic HH prompts, 300-prompt general/safety samples, a 16-prompt persona diagnostic, and any unresolved judge calls. Results are specific to the prompt template and greedy decoding. This draft must be visually checked and supplemented with your interpretation of actual outputs."))
    para("Exact dataset revisions and selected-row hashes are in outputs/prepared/manifest.json. Actual hyperparameters and package/GPU versions are in outputs/{sft,dpo_hh,dpo_uf,persona}/config.json. Training metrics and checkpoint selection are in metrics.jsonl and best_metrics.json. Generation settings are in outputs/eval/*.config.json; pair files retain hidden identity metadata locally; judgments, attempts, failures, and summaries use *_judged.*. Checkpoints are saved with best and final identities.")
    for tag, baseline in [("alpaca","sft"),("hh","sft"),("persona","smol_base")]:
        config_path=out/"eval"/f"{baseline}_{tag}.config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing actual generation settings: {config_path}")
        g=read_json(config_path)
        fields=["seed","do_sample","num_beams","max_new_tokens","max_prompt_tokens",
                "temperature","top_p","repetition_penalty","no_repeat_ngram_size","length_penalty"]
        para(f"Actual {tag} decoding: "+", ".join(f"{k}={g[k]}" for k in fields)+". Template: "+("native SmolLM2 chat template" if tag=="persona" else "explicit Instruction/Response"))
    para("References", "Heading2")
    for text in [
        "Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. https://arxiv.org/abs/2305.18290",
        "Bai et al. (2022). Training a Helpful and Harmless Assistant with RLHF. https://arxiv.org/abs/2204.05862",
        "Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.56.2/en/index",
        "TRL DPOTrainer 0.23.1: https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer",
        "PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora",
        "Kaggle Benchmarks: https://github.com/Kaggle/kaggle-benchmarks",
        "Course starter data/tests: https://github.com/ahmad903501/small-model-alignment-starter",
    ]:
        para(text)
    para("Disclose external references and any AI assistance as required by the course. Understand and adapt this implementation before submitting it as your work.")
    output=out/("report_final.pdf" if args.final else "report_draft.pdf")
    def footer(canvas,doc):
        canvas.setFont("Vera",8)
        canvas.drawRightString(A4[0]-40,24,f"PA2 | {doc.page}")
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(output).pages)
    if pages>8:
        raise RuntimeError(f"Draft is {pages} pages; shorten notes/examples to meet the 8-page limit")
    write_json(out/"report_metadata.json",{"pages":pages,"repo_url":args.repo_url,"is_draft":not args.final})
    print(f"Created {output} ({pages} pages). Render/inspect it before final submission.")


if __name__=="__main__":
    main()
