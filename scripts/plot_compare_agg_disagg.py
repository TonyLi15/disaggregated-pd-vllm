#!/usr/bin/env python3
# scripts/plot_compare_agg_disagg.py
# Compare agg vs disagg by fixing ANY TWO of (concurrency, prompt_tokens, max_tokens)
# and sweeping the remaining dimension.
#
# Examples:
#  - Fix conc & pt, sweep mt:
#      python3 scripts/plot_compare_agg_disagg.py --conc 32 --pt 1024
#  - Fix conc & mt, sweep pt:
#      python3 scripts/plot_compare_agg_disagg.py --conc 32 --mt 8192
#  - Fix pt & mt, sweep conc:
#      python3 scripts/plot_compare_agg_disagg.py --pt 256 --mt 2048
#
# Optional:
#  - --metric mean_tps|p50_ttft  (default: mean_tps)
#  - --model "Qwen/Qwen2.5-7B-Instruct"
#  - --input results/bench_runs  --output results/figures

import argparse, os, sys, glob, re
import pandas as pd
import matplotlib.pyplot as plt

try:
    from bench_utils import metadata_from_filename  # shared helper if present
except Exception:
    FNAME_META = re.compile(
        r"run_(?P<mode>agg|disagg)_model_(?P<modeltag>.+?)_conc(?P<conc>\d+)_pt(?P<pt>\d+)_mt(?P<mt>\d+)\.csv$"
    )
    def metadata_from_filename(path: str):
        base = os.path.basename(path)
        m = FNAME_META.search(base)
        if not m: return {}
        out = m.groupdict()
        out["conc"] = int(out.pop("conc"))
        out["pt"]   = int(out.pop("pt"))
        out["mt"]   = int(out.pop("mt"))
        out["modeltag"] = out.pop("modeltag")
        return {"mode": out["mode"], "model_tag": out["modeltag"],
                "concurrency": out["conc"], "prompt_tokens": out["pt"], "max_tokens": out["mt"]}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def parse_args():
    p = argparse.ArgumentParser(description="Compare agg vs disagg by fixing any two of (conc, pt, mt).")
    p.add_argument("--conc", type=int, help="Fixed concurrency")
    p.add_argument("--pt",   type=int, help="Fixed prompt_tokens")
    p.add_argument("--mt",   type=int, help="Fixed max_tokens")
    p.add_argument("--metric", choices=["mean_tps", "p50_ttft"], default="mean_tps",
                   help="Metric to plot (default: mean_tps)")
    p.add_argument("--model", type=str, help="Optional model_tag filter")
    p.add_argument("--input", default="results/bench_runs", help="Input dir or glob for CSVs")
    p.add_argument("--output", default="results/figures", help="Output dir for figures")
    return p.parse_args()

def validate_mode(a):
    provided = {k:v for k,v in {"concurrency":a.conc, "prompt_tokens":a.pt, "max_tokens":a.mt}.items() if v is not None}
    if len(provided) != 2:
        sys.exit("Please provide exactly TWO among --conc, --pt, --mt (the missing one will be swept).")
    missing = {"concurrency","prompt_tokens","max_tokens"} - set(provided.keys())
    sweep_dim = missing.pop()
    return provided, sweep_dim

def load_all(csv_glob):
    paths = sorted(glob.glob(csv_glob))
    if not paths: sys.exit(f"No CSV files found in {csv_glob}")
    frames=[]
    for pth in paths:
        try:
            df = pd.read_csv(pth)
        except Exception:
            continue
        meta = metadata_from_filename(pth)
        for k,v in meta.items():
            df[k] = v
        if "mean_tps" not in df.columns and "mean_tokens_per_sec" in df.columns:
            df["mean_tps"] = df["mean_tokens_per_sec"]
        frames.append(df)
    if not frames:
        sys.exit("No readable CSVs.")
    df = pd.concat(frames, ignore_index=True)
    for c in ["concurrency","prompt_tokens","max_tokens","mean_tps","p50_ttft"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["mode","model_tag","concurrency","prompt_tokens","max_tokens"], keep="last")
    return df

def main():
    a = parse_args()
    fixed, sweep_dim = validate_mode(a)

    os.makedirs(a.output, exist_ok=True)
    csv_glob = a.input if not os.path.isdir(a.input) else os.path.join(a.input, "run_*.csv")
    df = load_all(csv_glob)
    df = df[df["mode"].isin(["agg","disagg"])]

    # Apply fixed filters
    for k,v in fixed.items():
        df = df[df[k] == v]
    if a.model:
        df = df[df["model_tag"] == a.model]

    if df.empty:
        sys.exit(f"No data matching filters: fixed={fixed}, sweep={sweep_dim}, model={a.model or 'ANY'}")

    # Prepare plot
    plt.figure(figsize=(8,6))
    for mode in ["agg","disagg"]:
        sub = df[df["mode"] == mode]
        if sub.empty: continue
        grouped = sub.groupby(sweep_dim, as_index=False)[a.metric].mean().sort_values(sweep_dim)
        plt.plot(grouped[sweep_dim], grouped[a.metric], marker="o", label=mode)

    # Labels/titles
    axis_label = {"concurrency":"Concurrency",
                  "prompt_tokens":"Prompt tokens (pt)",
                  "max_tokens":"Max tokens (mt)"}[sweep_dim]
    ylabel = "Mean TPS" if a.metric == "mean_tps" else "p50 TTFT (s)"
    title_bits = [f"agg vs disagg", f"metric={a.metric}"]
    for k in ["concurrency","prompt_tokens","max_tokens"]:
        if k in fixed:
            title_bits.append(f"{k}={fixed[k]}")
    if a.model:
        title_bits.append(f"model={a.model}")
    plt.xlabel(axis_label); plt.ylabel(ylabel); plt.title(", ".join(title_bits))
    plt.legend(title="mode"); plt.grid(True); plt.tight_layout()

    # Output names
    tag = f"{a.metric}"
    for k in ["concurrency","prompt_tokens","max_tokens"]:
        if k in fixed: tag += f"_{k}{fixed[k]}"
    fig_path = os.path.join(a.output, f"compare_agg_disagg_sweep_{sweep_dim}_{tag}.png")
    plt.savefig(fig_path)
    print(f"[OK] Saved: {fig_path}")

    raw_path = os.path.join(a.output, f"raw_compare_sweep_{sweep_dim}_{tag}.csv")
    df.to_csv(raw_path, index=False)
    print(f"[OK] Raw data: {raw_path}")

if __name__ == "__main__":
    main()