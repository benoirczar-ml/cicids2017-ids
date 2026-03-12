import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV_DIR = os.environ.get("CSV_DIR", "/srv/work/datasets/cicids2017/raw/csv")
RAW_PARQUET_DIR = os.environ.get(
    "RAW_PARQUET_DIR", "/srv/work/datasets/cicids2017/raw/hf_bvsam/traffic_labels"
)
OUT_DIR = os.environ.get(
    "OUT_DIR", "/srv/work/datasets/cicids2017/processed/partitioned_raw"
)
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "500000"))

os.makedirs(OUT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
parquet_files = sorted(glob.glob(os.path.join(RAW_PARQUET_DIR, "*.parquet")))

def write_parquet_chunks(df_iter, prefix):
    part = 0
    for df in df_iter:
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        table = pa.Table.from_pandas(df, preserve_index=False)
        out_path = os.path.join(OUT_DIR, f"{prefix}-part-{part:04d}.parquet")
        pq.write_table(table, out_path)
        part += 1

if csv_files:
    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        reader = pd.read_csv(csv_path, chunksize=CHUNK_ROWS, low_memory=False)
        write_parquet_chunks(reader, base)
    print(f"Wrote partitioned parquet from CSVs to {OUT_DIR}")
elif parquet_files:
    for i, pq_path in enumerate(parquet_files):
        df = pd.read_parquet(pq_path)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        table = pa.Table.from_pandas(df, preserve_index=False)
        out_path = os.path.join(OUT_DIR, f"raw-part-{i:04d}.parquet")
        pq.write_table(table, out_path)
    print(f"Wrote partitioned parquet from parquet sources to {OUT_DIR}")
else:
    raise SystemExit("No CSV or parquet sources found for partitioning.")
