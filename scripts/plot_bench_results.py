#!/usr/bin/env python3
# scripts/plot_bench_results.py
# --------------------------------
# Plots benchmark results from CSVs produced by collect_from_log.py.
# Works with or without CLI args:
#   - Default: read results/bench_runs/*.csv and write figures to results/figures/
#   - With args: --input "<glob or dir>"  --output "<dir>"
#
# Examples:
#   python3 scripts/plot_bench_results.py
#   python3 scripts/plot_bench_results.py --input "results/bench_runs/*.csv" --output results/figures
#   python3 scripts/plot_bench_results.py --input results/bench_runs --output results/figures
#
import argparse
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from bench_utils import metadata_from_filename  # shared helper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def parse_args():
    p = argparse.ArgumentParser(description="Plot benchmark figures from CSVs.")
    p.add_argument("--input", help="CSV glob pattern or directory containing CSVs (default: results/bench_runs)",
                   default=None)
    p.add_argument("--output", help="Output directory for figures (default: results/figures)",
                   default=None)
    return p.parse_args()

def resolve_paths(args):
    # Defaults
    input_default = os.path.join(ROOT, "results", "bench_runs")
    output_default = os.path.join(ROOT, "results", "figures")

    input_arg = args.input if args.input else input_default
    output_arg = args.output if args.output else output_default

    # If input is a directory, expand to glob
    if os.path.isdir(input_arg):
        csv_glob = os.path.join(input_arg, "run_*.csv")
    else:
        csv_glob = input_arg  # already a glob pattern

    os.makedirs(output_arg, exist_ok=True)
    return csv_glob, output_arg

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    args = parse_args()
    csv_glob, out_dir = resolve_paths(args)

    csv_paths = sorted(glob.glob(csv_glob))
    if not csv_paths:
        sys.exit(f"No CSV files matched: {csv_glob}")

    df_list = []
    for path in csv_paths:
        df = pd.read_csv(path)

        # Prefer embedded metadata; otherwise derive from filename
        meta = {
            "mode": df["mode"].iloc[0] if "mode" in df.columns and pd.notna(df["mode"].iloc[0]) else None,
            "model_tag": df["model_tag"].iloc[0] if "model_tag" in df.columns and pd.notna(df["model_tag"].iloc[0]) else None,
            "concurrency": int(df["concurrency"].iloc[0]) if "concurrency" in df.columns and pd.notna(df["concurrency"].iloc[0]) else None,
            "prompt_tokens": int(df["prompt_tokens"].iloc[0]) if "prompt_tokens" in df.columns and pd.notna(df["prompt_tokens"].iloc[0]) else None,
            "max_tokens": int(df["max_tokens"].iloc[0]) if "max_tokens" in df.columns and pd.notna(df["max_tokens"].iloc[0]) else None,
        }
        if any(v is None for v in meta.values()):
            meta2 = metadata_from_filename(path)
            for k, v in meta2.items():
                if meta.get(k) is None:
                    meta[k] = v

        meta.setdefault("mode", "unknown")
        meta.setdefault("model_tag", "unknown")
        for k, v in meta.items():
            df[k] = v

        # Backward-compat column name
        if "mean_tps" not in df.columns and "mean_tokens_per_sec" in df.columns:
            df["mean_tps"] = df["mean_tokens_per_sec"]

        df_list.append(df)

    all_df = pd.concat(df_list, ignore_index=True)

    # Coerce numeric types for grouping/plotting
    all_df = ensure_numeric(all_df, ["concurrency", "prompt_tokens", "max_tokens", "mean_tps", "p50_ttft"])

    # Drop exact duplicates on the key experiment dimensions
    key_cols = ["mode", "model_tag", "concurrency", "prompt_tokens", "max_tokens"]
    all_df = all_df.drop_duplicates(subset=key_cols, keep="last")

    # Save the entire merged table for inspection
    all_raw_path = os.path.join(out_dir, "raw_all_merged.csv")
    all_df.to_csv(all_raw_path, index=False)
    print(f"[INFO] Wrote merged raw table: {all_raw_path}")

    # Sanity report
    for mode in sorted(all_df["mode"].dropna().unique()):
        sub = all_df[all_df["mode"] == mode]
        concs = sorted([int(x) for x in sub['concurrency'].dropna().unique()])
        pts   = sorted([int(x) for x in sub['prompt_tokens'].dropna().unique()])
        mts   = sorted([int(x) for x in sub['max_tokens'].dropna().unique()])
        print(f"[SANITY] mode={mode}: conc={concs}, pt={pts}, mt={mts}")

    # Verify essential columns
    needed = ["concurrency", "prompt_tokens", "max_tokens", "mean_tps", "p50_ttft", "mode"]
    for col in needed:
        if col not in all_df.columns:
            sys.exit(f"Missing required column '{col}' in aggregated dataframe. Check your CSVs/collector.")

    # ----------------------------
    # Plot A: Throughput vs Concurrency (fixed pt; one line per max_tokens)
    # ----------------------------
    for mode in sorted(all_df["mode"].dropna().unique()):
        subm = all_df[all_df["mode"] == mode].copy()
        if subm.empty:
            continue
        for pt in sorted(subm["prompt_tokens"].dropna().unique()):
            sub = subm[subm["prompt_tokens"] == pt].copy()
            if sub.empty:
                continue

            plt.figure(figsize=(8, 6))
            for mt in sorted(sub["max_tokens"].dropna().unique()):
                line = sub[sub["max_tokens"] == mt].sort_values("concurrency")
                if line.empty:
                    continue
                plt.plot(line["concurrency"], line["mean_tps"], marker="o", label=f"mt={int(mt)}")

            plt.xlabel("Concurrency")
            plt.ylabel("Mean TPS")
            plt.title(f"Throughput vs Concurrency (mode={mode}, pt={int(pt)})")
            plt.legend(title="max_tokens")
            plt.grid(True)
            plt.tight_layout()

            out_img = os.path.join(out_dir, f"throughput_vs_concurrency_mode{mode}_pt{int(pt)}.png")
            plt.savefig(out_img)
            print(f"Saved figure {out_img}")

            raw_out = os.path.join(out_dir, f"raw_throughput_mode{mode}_pt{int(pt)}.csv")
            sub.sort_values(["max_tokens", "concurrency"])[
                ["mode", "model_tag", "prompt_tokens", "max_tokens", "concurrency", "mean_tps"]
            ].to_csv(raw_out, index=False)
            print(f"Wrote raw slice {raw_out}")

    # ----------------------------
    # Plot B: TTFT vs Prompt Tokens (fixed conc; one line per max_tokens)
    # ----------------------------
    for mode in sorted(all_df["mode"].dropna().unique()):
        subm = all_df[all_df["mode"] == mode].copy()
        if subm.empty:
            continue
        for conc in sorted(subm["concurrency"].dropna().unique()):
            sub = subm[subm["concurrency"] == conc].copy()
            if sub.empty:
                continue

            plt.figure(figsize=(8, 6))
            for mt in sorted(sub["max_tokens"].dropna().unique()):
                line = sub[sub["max_tokens"] == mt].sort_values("prompt_tokens")
                if line.empty:
                    continue
                plt.plot(line["prompt_tokens"], line["p50_ttft"], marker="o", label=f"mt={int(mt)}")

            plt.xlabel("Prompt tokens")
            plt.ylabel("p50 TTFT (s)")
            plt.title(f"TTFT vs Prompt Tokens (mode={mode}, conc={int(conc)})")
            plt.legend(title="max_tokens")
            plt.grid(True)
            plt.tight_layout()

            out_img = os.path.join(out_dir, f"ttft_vs_pt_mode{mode}_conc{int(conc)}.png")
            plt.savefig(out_img)
            print(f"Saved figure {out_img}")

            raw_out = os.path.join(out_dir, f"raw_ttft_mode{mode}_conc{int(conc)}.csv")
            sub.sort_values(["max_tokens", "prompt_tokens"])[
                ["mode", "model_tag", "concurrency", "max_tokens", "prompt_tokens", "p50_ttft"]
            ].to_csv(raw_out, index=False)
            print(f"Wrote raw slice {raw_out}")

    print("All figures written.")

if __name__ == "__main__":
    main()