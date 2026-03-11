import glob
import json
import os
import cudf

DATA_DIR = "/srv/work/datasets/cicids2017/parquet/hf-bvk"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
if not files:
    raise SystemExit(f"No parquet files in {DATA_DIR}")

frames = []
row_counts = {}
for f in files:
    df = cudf.read_parquet(f)
    row_counts[os.path.basename(f)] = int(len(df))
    frames.append(df)

df = cudf.concat(frames, ignore_index=True)

total_rows = int(len(df))
columns = list(df.columns)
schema = {c: str(df[c].dtype) for c in columns}

# Null counts
null_counts = df.isnull().sum().to_pandas().to_dict()

# Label stats
label_col = None
for c in columns:
    if c.lower() in {"label", "class", "attack"}:
        label_col = c
        break

label_counts = None
if label_col:
    label_counts = df[label_col].value_counts().to_pandas()

# Save reports
with open(os.path.join(REPORT_DIR, "schema.json"), "w") as f:
    json.dump({"total_rows": total_rows, "row_counts": row_counts, "schema": schema}, f, indent=2)

with open(os.path.join(REPORT_DIR, "null_counts.json"), "w") as f:
    json.dump(null_counts, f, indent=2)

if label_counts is not None:
    label_counts.to_csv(os.path.join(REPORT_DIR, "label_counts.csv"))

print(f"Total rows: {total_rows}")
print(f"Columns: {len(columns)}")
print(f"Label column: {label_col}")
