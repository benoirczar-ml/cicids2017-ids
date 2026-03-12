import os
import json
import cudf
import cupy as cp
import torch
from cuml.ensemble import IsolationForest
from cuml.model_selection import train_test_split

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

_df = cudf.read_parquet(DATA_PATH)
label_col = "Label"
label_bin = "Label_Binary"

holdout_attacks = ["Botnet", "Web Attack - Brute Force", "Infiltration - Portscan"]

# Botnet-focused feature subset
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


def cupy_to_torch(x):
    return torch.utils.dlpack.from_dlpack(x)


def confusion_matrix_binary_torch(y_true, y_pred):
    y_true = y_true.int()
    y_pred = y_pred.int()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tn, fp, fn, tp


def precision_recall_f1_torch(tn, fp, fn, tp):
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def roc_auc_torch(y_true, y_score):
    y_true = y_true.float()
    y_score = y_score.float()
    pos = y_true.sum()
    neg = y_true.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    idx = torch.argsort(y_score, descending=True)
    y_sorted = y_true[idx]
    tps = torch.cumsum(y_sorted, 0)
    fps = torch.cumsum(1 - y_sorted, 0)
    tpr = tps / pos
    fpr = fps / neg
    tpr = torch.cat([torch.zeros(1, device=y_true.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=y_true.device), fpr])
    auc = torch.trapz(tpr, fpr)
    return auc.item()


def average_precision_torch(y_true, y_score):
    y_true = y_true.float()
    y_score = y_score.float()
    pos = y_true.sum()
    if pos == 0:
        return float("nan")
    idx = torch.argsort(y_score, descending=True)
    y_sorted = y_true[idx]
    tps = torch.cumsum(y_sorted, 0)
    fps = torch.cumsum(1 - y_sorted, 0)
    precision = tps / (tps + fps + 1e-12)
    ap = (precision * y_sorted).sum() / pos
    return ap.item()


def threshold_by_benign_fpr(benign_scores, fpr_target):
    q = 1.0 - fpr_target
    return float(torch.quantile(benign_scores, q).item())


def eval_metrics(y_true, scores, threshold):
    preds = (scores >= threshold).int()
    tn, fp, fn, tp = confusion_matrix_binary_torch(y_true, preds)
    precision, recall, f1 = precision_recall_f1_torch(tn, fp, fn, tp)
    roc = roc_auc_torch(y_true, scores)
    ap = average_precision_torch(y_true, scores)
    return {
        "tn": int(tn.item()), "fp": int(fp.item()), "fn": int(fn.item()), "tp": int(tp.item()),
        "precision": precision.item(), "recall": recall.item(), "f1": f1.item(),
        "roc_auc": roc, "pr_auc": ap,
    }


results = []

# benign-only data for IsolationForest
benign_full = _df[_df[label_col] == "BENIGN"][FEATURES]

# train/val split
X_train_b, X_val_b = train_test_split(benign_full, test_size=0.25, random_state=42)

if len(X_train_b) > 200_000:
    X_train_b = X_train_b.sample(n=200_000, random_state=42)
if len(X_val_b) > 100_000:
    X_val_b = X_val_b.sample(n=100_000, random_state=42)

# fit IsolationForest (benign-only)
iso = IsolationForest(
    n_estimators=200,
    max_samples=1.0,
    max_features=1.0,
    contamination=0.001,
    random_state=42,
)
iso.fit(X_train_b)

# benign scores for thresholds
val_scores = iso.score_samples(X_val_b)
val_scores_t = cupy_to_torch(val_scores).float()
# higher score => more normal; we invert to anomaly score
val_anom = -val_scores_t

thr_1e3 = threshold_by_benign_fpr(val_anom, 1e-3)
thr_1e4 = threshold_by_benign_fpr(val_anom, 1e-4)

for holdout_attack in holdout_attacks:
    attack_df = _df[_df[label_col] == holdout_attack]
    if len(attack_df) == 0:
        continue

    target_attack = min(int(len(attack_df)), 50_000)
    attack_s = attack_df.sample(n=target_attack, random_state=42)[FEATURES]
    benign_s = _df[_df[label_col] == "BENIGN"].sample(n=target_attack, random_state=42)[FEATURES]

    test_df = cudf.concat([attack_s, benign_s], ignore_index=True)
    y_test = cudf.Series(cp.concatenate([
        cp.ones(len(attack_s), dtype=cp.int32),
        cp.zeros(len(benign_s), dtype=cp.int32),
    ]))

    test_scores = iso.score_samples(test_df)
    test_anom = -cupy_to_torch(test_scores).float()
    y_test_t = cupy_to_torch(y_test.to_cupy()).int()

    res = {
        "holdout_attack": holdout_attack,
        "holdout_rows": int(len(attack_s)),
        "test_rows": int(len(test_df)),
        "benign_train_rows": int(len(X_train_b)),
        "benign_val_rows": int(len(X_val_b)),
        "thresholds": {"fpr_1e-3": thr_1e3, "fpr_1e-4": thr_1e4},
        "test_metrics@fpr_1e-3": eval_metrics(y_test_t, test_anom, thr_1e3),
        "test_metrics@fpr_1e-4": eval_metrics(y_test_t, test_anom, thr_1e4),
    }

    results.append(res)

out_path = os.path.join(REPORT_DIR, "anomaly_isoforest_botnet_focus.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
