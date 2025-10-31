# scripts/bench_utils.py
#!/usr/bin/env python3
"""
bench_utils.py
Common helpers shared by benchmark scripts (collector, plotter, runners).
"""

import os
import re

FNAME_META = re.compile(
    r"run_(?P<mode>agg|disagg)_model_(?P<modeltag>.+?)_conc(?P<conc>\d+)_pt(?P<pt>\d+)_mt(?P<mt>\d+)\.(?:csv|log)$"
)

def metadata_from_filename(path: str):
    """Extract (mode, model_tag, conc, pt, mt) from the standardized filename.
    Falls back to best-effort parsing for legacy names.
    """
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

    # Legacy fallback: try to glean conc/pt/mt tokens
    parts = base.replace("run_", "").replace(".csv", "").replace(".log", "").split("_")

    def _pick(prefix):
        for p in parts:
            if p.startswith(prefix):
                try:
                    return int(p[len(prefix):])
                except ValueError:
                    pass
        return None

    conc = _pick("conc")
    pt   = _pick("pt")
    mt   = _pick("mt")

    # Best-effort model tag
    try:
        if "model" in parts and conc is not None:
            model_idx = parts.index("model")
            model_tag = "_".join(parts[model_idx + 1: parts.index(f"conc{conc}")])
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