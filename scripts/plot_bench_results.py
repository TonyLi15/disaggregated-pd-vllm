#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# ----------------------------
# Input and output directories
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_DIR = os.path.join(ROOT, "results", "bench_runs")
OUTPUT_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
FNAME_META = re.compile(
    r"run_(?P<mode>agg|disagg)_model_(?P<modeltag>.+?)_conc(?P<conc>\d+)_pt(?P<pt>\d+)_mt(?P<mt>\d+)\.csv$"
)

def metadata_from_filename(path: str):
    """Fallback extractor when CSV doesn't carry meta columns."""
    base = os.path.basename(path)
    m = FNAME_META.search(base)
    if m:
        return {
            "mode": m.group("mode"),
            "model_tag": m.group("modeltag"),
            "concurrency": int(m.group("conc")),
            "prompt_tokens": int(m.group("pt")),
            "max_tokens": int(m.group("mt")),
        }

    # Legacy format fallback (no mode prefix). Try to glean conc/pt/mt.
    parts = base.replace("run_", "").replace(".csv", "").split("_")
    def _pick(prefix):
        for p in parts:
            if p.startswith(prefix):
                return int(p.replace(prefix, ""))
        return None

    conc = _pick("conc")
    pt   = _pick("pt")
    mt   = _pick("mt")

    # best-effort model tag
    try:
        if "model" in parts and conc is not None:
            model_idx = parts.index("model")
            model_tag = "_".join(parts[model_idx+1:parts.index(f"conc{conc}")])
        else:
            model_tag = "unknown"
    except Exception:
        model_tag = "unknown"

    return {
        "mode": "unknown",
        "model_tag": model_tag,
        "concurrency": conc,
        "prompt_tokens": pt,
        "max_tokens": mt,
    }

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------
# Load CSVs
# ----------------------------
csv_paths = glob.glob(os.path.join(INPUT_DIR, "run_*.csv"))
if not csv_paths:
    raise SystemExit(f"No CSV files found in {INPUT_DIR}")

df_list = []
for path in csv_paths:
    df = pd.read_csv(path)

    # Prefer metadata already inside CSV (from collector)
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

    # Fill defaults if still missing
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

# Drop exact duplicates on the key experiment dimensions to avoid overlapping points
key_cols = ["mode", "model_tag", "concurrency", "prompt_tokens", "max_tokens"]
all_df = all_df.drop_duplicates(subset=key_cols, keep="last")

# Save the entire merged table for inspection
all_raw_path = os.path.join(OUTPUT_DIR, "raw_all_merged.csv")
all_df.to_csv(all_raw_path, index=False)
print(f"[INFO] Wrote merged raw table: {all_raw_path}")

# Sanity: show what we actually have
for mode in sorted(all_df["mode"].dropna().unique()):
    sub = all_df[all_df["mode"] == mode]
    print(f"[SANITY] mode={mode}: conc={sorted(sub['concurrency'].dropna().unique())}, "
          f"pt={sorted(sub['prompt_tokens'].dropna().unique())}, "
          f"mt={sorted(sub['max_tokens'].dropna().unique())}")

# Verify essential columns
needed = ["concurrency", "prompt_tokens", "max_tokens", "mean_tps", "p50_ttft", "mode"]
for col in needed:
    if col not in all_df.columns:
        raise SystemExit(f"Missing required column '{col}' in aggregated dataframe. Check your CSVs/collector.")

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
        # lines per max_tokens
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

        out_img = os.path.join(OUTPUT_DIR, f"throughput_vs_concurrency_mode{mode}_pt{int(pt)}.png")
        plt.savefig(out_img)
        print(f"Saved figure {out_img}")

        # Dump raw used data for this figure
        raw_out = os.path.join(OUTPUT_DIR, f"raw_throughput_mode{mode}_pt{int(pt)}.csv")
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
        # lines per max_tokens
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

        out_img = os.path.join(OUTPUT_DIR, f"ttft_vs_pt_mode{mode}_conc{int(conc)}.png")
        plt.savefig(out_img)
        print(f"Saved figure {out_img}")

        # Dump raw used data for this figure
        raw_out = os.path.join(OUTPUT_DIR, f"raw_ttft_mode{mode}_conc{int(conc)}.csv")
        sub.sort_values(["max_tokens", "prompt_tokens"])[
            ["mode", "model_tag", "concurrency", "max_tokens", "prompt_tokens", "p50_ttft"]
        ].to_csv(raw_out, index=False)
        print(f"Wrote raw slice {raw_out}")

print("All figures written.")