#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Input and output directories
INPUT_DIR = "results/bench_runs"
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all CSV files from the input directory
csv_paths = glob.glob(os.path.join(INPUT_DIR, "run_*.csv"))
df_list = []
for path in csv_paths:
	df = pd.read_csv(path)
	# Extract parameter values (handles filenames with model tags)
	fname = os.path.basename(path)
	# Example: run_model_Qwen_Qwen2.5-7B-Instruct_conc4_pt64_mt128.csv
	fname = fname.replace("run_", "").replace(".csv", "")
	parts = fname.split("_")

	# Dynamically extract numeric parameters
	conc = next(int(p.replace("conc", "")) for p in parts if p.startswith("conc"))
	pt   = next(int(p.replace("pt", ""))   for p in parts if p.startswith("pt"))
	mt   = next(int(p.replace("mt", ""))   for p in parts if p.startswith("mt"))

	# Extract model tag if present
	model_tag = "_".join(parts[1:parts.index(f"conc{conc}")]) if "model" in parts else "unknown"

	# Add metadata columns
	df["concurrency"] = conc
	df["prompt_tokens"] = pt
	df["max_tokens"] = mt
	df["model"] = model_tag
	df_list.append(df)

# Combine all CSVs into a single DataFrame
all_df = pd.concat(df_list, ignore_index=True)

# Example Plot 1: concurrency vs mean_tokens/sec
plt.figure(figsize=(8, 6))
for pt in sorted(all_df["prompt_tokens"].unique()):
    sub = all_df[all_df["prompt_tokens"] == pt]
    plt.plot(sub["concurrency"], sub["mean_tps"], marker='o', label=f"pt={pt}")
plt.xlabel("Concurrency")
plt.ylabel("Mean TPS")
plt.title("Throughput vs Concurrency (by prompt_tokens)")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "throughput_vs_concurrency.png")
plt.savefig(out_path)
print(f"Saved figure {out_path}")

# Example Plot 2: prompt_tokens vs p50_ttft
plt.figure(figsize=(8, 6))
for conc in sorted(all_df["concurrency"].unique()):
    sub = all_df[all_df["concurrency"] == conc]
    plt.plot(sub["prompt_tokens"], sub["p50_ttft"], marker='o', label=f"conc={conc}")
plt.xlabel("Prompt tokens")
plt.ylabel("p50 TTFT (s)")
plt.title("Latency (TTFT) vs Prompt Tokens (by concurrency)")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "ttft_vs_prompt_tokens.png")
plt.savefig(out_path)
print(f"Saved figure {out_path}")

# Add more plots following the same pattern if needed