import os
import glob
import cudf

RAW_DIR = "/srv/work/datasets/cicids2017/parquet/hf-bvk"
OUT_DIR = "/srv/work/datasets/cicids2017/processed/windows_1s"

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SEC = 1.0

COLS = [
    "Src IP dec",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "SYN Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "URG Flag Count",
    "Packet Length Mean",
    "Packet Length Min",
    "Down/Up Ratio",
    "Label",
]

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))

for i, path in enumerate(files):
    df = cudf.read_parquet(path, columns=COLS)

    # Timestamp like "MM:SS.s"
    ts_split = df["Timestamp"].str.split(":", expand=True)
    minutes = ts_split[0].astype("float32")
    seconds = ts_split[1].astype("float32")
    t_seconds = minutes * 60.0 + seconds

    window_id = (t_seconds / WINDOW_SEC).astype("int32")
    df = df.assign(window_id=window_id)

    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    df["attack_label"] = df["Label"].where(df["Label"] != "BENIGN", None)

    # normalize labels
    df["attack_label"] = df["attack_label"].str.replace("Botnet - Attempted", "Botnet")
    df["attack_label"] = df["attack_label"].str.replace("Web Attack - Brute Force - Attempted", "Web Attack - Brute Force")

    group_cols = ["Src IP dec", "window_id"]
    agg = {
        "Flow Duration": "mean",
        "Flow Bytes/s": "mean",
        "Flow Packets/s": "mean",
        "Fwd Packets/s": "mean",
        "Bwd Packets/s": "mean",
        "SYN Flag Count": "sum",
        "FIN Flag Count": "sum",
        "RST Flag Count": "sum",
        "URG Flag Count": "sum",
        "Packet Length Mean": "mean",
        "Packet Length Min": "min",
        "Down/Up Ratio": "mean",
        "is_attack": "max",
        "attack_label": "first",
        "Protocol": "first",
    }

    out = df.groupby(group_cols).agg(agg).reset_index()
    out["attack_label"] = out["attack_label"].fillna("BENIGN")
    out["Label_Binary"] = out["is_attack"].astype("int32")
    out = out.drop(columns=["is_attack"])

    out = out.rename(columns={
        "Flow Duration": "mean_flow_duration",
        "Flow Bytes/s": "mean_flow_bytes_s",
        "Flow Packets/s": "mean_flow_packets_s",
        "Fwd Packets/s": "mean_fwd_packets_s",
        "Bwd Packets/s": "mean_bwd_packets_s",
        "SYN Flag Count": "sum_syn",
        "FIN Flag Count": "sum_fin",
        "RST Flag Count": "sum_rst",
        "URG Flag Count": "sum_urg",
        "Packet Length Mean": "mean_pkt_len_mean",
        "Packet Length Min": "min_pkt_len_min",
        "Down/Up Ratio": "mean_down_up_ratio",
        "Protocol": "protocol",
    })

    out_path = os.path.join(OUT_DIR, f"part-{i:04d}.parquet")
    out.to_parquet(out_path)

print(f"Wrote windowed aggregates to {OUT_DIR}")
