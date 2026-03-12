import os
import json
import numpy as np
import pandas as pd
import faiss

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

label_col = "Label"
label_bin = "Label_Binary"

holdout_attacks = ["Botnet", "Web Attack - Brute Force", "Infiltration - Portscan"]

FEATURES = [
    "Protocol",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Down/Up Ratio",
    "SYN Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "URG Flag Count",
    "Bwd RST Flags",
    "Fwd RST Flags",
    "Packet Length Min",
    "Packet Length Mean",
    "Fwd Packet Length Min",
    "Bwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
]


def confusion_matrix_binary(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn, fp, fn, tp


def precision_recall_f1(tn, fp, fn, tp):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def roc_auc(y_true, y_score):
    y_true = y_true.astype(np.float32)
    y_score = y_score.astype(np.float32)
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    idx = np.argsort(-y_score)
    y_sorted = y_true[idx]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / pos
    fpr = fps / neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapezoid(tpr, fpr))


def average_precision(y_true, y_score):
    y_true = y_true.astype(np.float32)
    y_score = y_score.astype(np.float32)
    pos = y_true.sum()
    if pos == 0:
        return float("nan")
    idx = np.argsort(-y_score)
    y_sorted = y_true[idx]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    precision = tps / (tps + fps + 1e-12)
    ap = (precision * y_sorted).sum() / pos
    return float(ap)


def threshold_by_benign_fpr(benign_scores, fpr_target):
    q = 1.0 - fpr_target
    return float(np.quantile(benign_scores, q))


def eval_metrics(y_true, scores, threshold):
    preds = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix_binary(y_true, preds)
    precision, recall, f1 = precision_recall_f1(tn, fp, fn, tp)
    roc = roc_auc(y_true, scores)
    ap = average_precision(y_true, scores)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": roc, "pr_auc": ap,
    }


# Load parquet via pandas/pyarrow
_df = pd.read_parquet(DATA_PATH, columns=FEATURES + [label_col, label_bin])

# Benign-only train/val
benign_df = _df[_df[label_col] == "BENIGN"]

train_b = benign_df.sample(n=min(len(benign_df), 200_000), random_state=42)
val_b = benign_df.sample(n=min(len(benign_df), 50_000), random_state=7)

X_train = train_b[FEATURES].to_numpy(dtype=np.float32)
X_val = val_b[FEATURES].to_numpy(dtype=np.float32)

# Standardize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# FAISS GPU index
res = faiss.StandardGpuResources()
d = X_train.shape[1]
index_cpu = faiss.IndexFlatL2(d)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(X_train)

k = 10
val_D, _ = index.search(X_val, k)
val_scores = val_D[:, -1]

thr_1e3 = threshold_by_benign_fpr(val_scores, 1e-3)
thr_1e4 = threshold_by_benign_fpr(val_scores, 1e-4)

results = []
for holdout_attack in holdout_attacks:
    attack_df = _df[_df[label_col] == holdout_attack]
    if len(attack_df) == 0:
        continue

    target_attack = min(int(len(attack_df)), 50_000)
    attack_s = attack_df.sample(n=target_attack, random_state=42)
    benign_s = benign_df.sample(n=target_attack, random_state=42)

    test_df = pd.concat([attack_s, benign_s], ignore_index=True)
    y_test = test_df[label_bin].to_numpy(dtype=np.int32)

    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
    X_test = (X_test - mean) / std

    test_D, _ = index.search(X_test, k)
    test_scores = test_D[:, -1]

    res = {
        "holdout_attack": holdout_attack,
        "holdout_rows": int(len(attack_s)),
        "test_rows": int(len(test_df)),
        "benign_train_rows": int(len(train_b)),
        "benign_val_rows": int(len(val_b)),
        "thresholds": {"fpr_1e-3": thr_1e3, "fpr_1e-4": thr_1e4},
        "test_metrics@fpr_1e-3": eval_metrics(y_test, test_scores, thr_1e3),
        "test_metrics@fpr_1e-4": eval_metrics(y_test, test_scores, thr_1e4),
    }
    results.append(res)

out_path = os.path.join(REPORT_DIR, "anomaly_faiss_knn_botnet_focus.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
