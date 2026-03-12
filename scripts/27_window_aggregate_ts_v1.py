import os
import glob
import cudf

RAW_DIR = "/srv/work/datasets/cicids2017/raw/hf_bvsam/traffic_labels"
OUT_DIR = "/srv/work/datasets/cicids2017/processed/windows_1s_ts_v1"

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SEC = 1

COLS = [
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
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
    "Min Packet Length",
    "Down/Up Ratio",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Label",
]

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
if not files:
    raise SystemExit(f"No parquet files in {RAW_DIR}")

for i, path in enumerate(files):
    df = cudf.read_parquet(path, columns=COLS)

    # Floor timestamps to window size
    if WINDOW_SEC == 1:
        df["window_ts"] = df["Timestamp"].astype("datetime64[s]")
    else:
        ts_int = df["Timestamp"].astype("int64")
        window_us = WINDOW_SEC * 1_000_000
        df["window_ts"] = (ts_int // window_us) * window_us
        df["window_ts"] = df["window_ts"].astype("datetime64[us]")

    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    df["attack_label"] = df["Label"].where(df["Label"] != "BENIGN", None)

    group_cols = ["Source IP", "window_ts"]

    agg = {
        "Flow Duration": ["mean", "count"],
        "Flow Bytes/s": "mean",
        "Flow Packets/s": "mean",
        "Fwd Packets/s": "mean",
        "Bwd Packets/s": "mean",
        "SYN Flag Count": "sum",
        "FIN Flag Count": "sum",
        "RST Flag Count": "sum",
        "URG Flag Count": "sum",
        "Packet Length Mean": "mean",
        "Min Packet Length": "min",
        "Down/Up Ratio": "mean",
        "Total Fwd Packets": "sum",
        "Total Backward Packets": "sum",
        "Total Length of Fwd Packets": "sum",
        "Total Length of Bwd Packets": "sum",
        "Destination IP": "nunique",
        "Destination Port": "nunique",
        "Source Port": "nunique",
        "Protocol": "first",
        "is_attack": "max",
        "attack_label": "first",
    }

    out = df.groupby(group_cols).agg(agg).reset_index()

    out.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in out.columns
    ]

    out = out.rename(columns={
        "Source IP": "src_ip",
        "Flow Duration_mean": "mean_flow_duration",
        "Flow Duration_count": "flow_count",
        "Flow Bytes/s_mean": "mean_flow_bytes_s",
        "Flow Packets/s_mean": "mean_flow_packets_s",
        "Fwd Packets/s_mean": "mean_fwd_packets_s",
        "Bwd Packets/s_mean": "mean_bwd_packets_s",
        "SYN Flag Count_sum": "sum_syn",
        "FIN Flag Count_sum": "sum_fin",
        "RST Flag Count_sum": "sum_rst",
        "URG Flag Count_sum": "sum_urg",
        "Packet Length Mean_mean": "mean_pkt_len_mean",
        "Min Packet Length_min": "min_pkt_len_min",
        "Down/Up Ratio_mean": "mean_down_up_ratio",
        "Total Fwd Packets_sum": "sum_total_fwd_packets",
        "Total Backward Packets_sum": "sum_total_bwd_packets",
        "Total Length of Fwd Packets_sum": "sum_total_fwd_bytes",
        "Total Length of Bwd Packets_sum": "sum_total_bwd_bytes",
        "Destination IP_nunique": "nunique_dst_ip",
        "Destination Port_nunique": "nunique_dst_port",
        "Source Port_nunique": "nunique_src_port",
        "Protocol_first": "protocol",
        "is_attack_max": "is_attack",
        "attack_label_first": "attack_label",
    })

    out["attack_label"] = out["attack_label"].fillna("BENIGN")
    out["Label_Binary"] = out["is_attack"].astype("int32")
    out = out.drop(columns=["is_attack"])

    out_path = os.path.join(OUT_DIR, f"part-{i:04d}.parquet")
    out.to_parquet(out_path)

print(f"Wrote windowed aggregates to {OUT_DIR}")
