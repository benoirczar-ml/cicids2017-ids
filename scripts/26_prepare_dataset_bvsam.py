import glob
import json
import os
import numpy as np

USE_PANDAS = os.environ.get("USE_PANDAS", "0") == "1"

if USE_PANDAS:
    import pandas as pd
else:
    import cudf

RAW_DIR = "/srv/work/datasets/cicids2017/raw/hf_bvsam/traffic_labels"
OUT_DIR = "/srv/work/datasets/cicids2017/processed"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
if not files:
    raise SystemExit(f"No parquet files in {RAW_DIR}")

if USE_PANDAS:
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
else:
    frames = [cudf.read_parquet(f) for f in files]
    df = cudf.concat(frames, ignore_index=True)

# Drop likely leakage columns (IPs, ports, timestamps, flow ids)
leak_cols = []
for c in df.columns:
    cl = c.lower()
    if "ip" in cl or "port" in cl or "timestamp" in cl or "flow id" in cl:
        leak_cols.append(c)

if leak_cols:
    df = df.drop(columns=leak_cols)

label_col = "Label" if "Label" in df.columns else None
if label_col is None:
    raise SystemExit("Label column not found")

df["Label_Binary"] = (df[label_col] != "BENIGN").astype("int8")

# Replace inf with null, fill nulls with median (per column)
numeric_cols = [c for c in df.columns if c not in (label_col, "Label_Binary")]
medians = {}
for c in numeric_cols:
    if df[c].dtype.kind in "fi":
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        medians[c] = float(med) if med is not None else 0.0
        df[c] = df[c].fillna(med)

out_path = os.path.join(OUT_DIR, "cicids2017_clean.parquet")
df.to_parquet(out_path, index=False)

meta = {
    "source": "hf-bvsam traffic_labels parquet",
    "raw_dir": RAW_DIR,
    "rows": int(len(df)),
    "columns": list(df.columns),
    "dropped_leak_cols": leak_cols,
    "medians": medians,
}

with open(os.path.join(REPORT_DIR, "prep_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

with open(os.path.join(REPORT_DIR, "feature_list.txt"), "w") as f:
    for c in df.columns:
        if c not in (label_col, "Label_Binary"):
            f.write(c + "\n")

print(f"Saved: {out_path}")
print(f"Rows: {len(df)} | Cols: {len(df.columns)}")
