import os
import json
import cudf
import cupy as cp
import xgboost as xgb
import torch
from cuml.model_selection import train_test_split

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

_df = cudf.read_parquet(DATA_PATH)
label_col = "Label"
label_bin = "Label_Binary"

holdout_attacks = ["Botnet", "Web Attack - Brute Force", "Infiltration - Portscan"]

# Selected stable features (appear in >=2 holdout shifts)
FEATURES = [
    "Fwd Seg Size Min",
    "Bwd Packet Length Mean",
    "Bwd Segment Size Avg",
    "Protocol",
    "SYN Flag Count",
    "Total Length of Fwd Packet",
    "Subflow Fwd Bytes",
    "Fwd Packet Length Mean",
    "Fwd Segment Size Avg",
    "Down/Up Ratio",
    "Packet Length Mean",
    "Average Packet Size",
    "FIN Flag Count",
    "Packet Length Min",
    "Bwd Packet Length Min",
    "Fwd Packet Length Min",
    "Subflow Bwd Bytes",
]


def stratified_sample(X, y, target_rows, seed=42):
    if target_rows >= len(y):
        return X, y
    cp.random.seed(seed)
    y_cp = y.to_cupy()
    idx_all = cp.arange(len(y_cp))
    pos_idx = idx_all[y_cp == 1]
    neg_idx = idx_all[y_cp == 0]
    pos_count = int(len(pos_idx))
    neg_count = int(len(neg_idx))
    total = pos_count + neg_count
    if total == 0:
        return X, y
    target_pos = int(target_rows * (pos_count / total))
    target_neg = target_rows - target_pos
    target_pos = min(target_pos, pos_count)
    target_neg = min(target_neg, neg_count)
    pos_sample = cp.random.choice(pos_idx, size=target_pos, replace=False) if target_pos > 0 else cp.array([], dtype=cp.int64)
    neg_sample = cp.random.choice(neg_idx, size=target_neg, replace=False) if target_neg > 0 else cp.array([], dtype=cp.int64)
    sample_idx = cp.concatenate([pos_sample, neg_sample])
    sample_idx = cp.random.permutation(sample_idx)
    return X.iloc[sample_idx], y.iloc[sample_idx]


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


def threshold_by_benign_fpr(benign_probs, fpr_target):
    q = 1.0 - fpr_target
    return float(torch.quantile(benign_probs, q).item())


def eval_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).int()
    tn, fp, fn, tp = confusion_matrix_binary_torch(y_true, preds)
    precision, recall, f1 = precision_recall_f1_torch(tn, fp, fn, tp)
    roc = roc_auc_torch(y_true, probs)
    ap = average_precision_torch(y_true, probs)
    return {
        "tn": int(tn.item()), "fp": int(fp.item()), "fn": int(fn.item()), "tp": int(tp.item()),
        "precision": precision.item(), "recall": recall.item(), "f1": f1.item(),
        "roc_auc": roc, "pr_auc": ap,
    }


results = []
for holdout_attack in holdout_attacks:
    mask_holdout = _df[label_col] == holdout_attack
    train_df = _df[~mask_holdout]
    attack_df = _df[mask_holdout]

    if len(attack_df) == 0:
        continue

    X_all = train_df[FEATURES]
    y_all = train_df[label_bin].astype("int32")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )

    X_train, y_train = stratified_sample(X_train, y_train, target_rows=400_000)
    X_val, y_val = stratified_sample(X_val, y_val, target_rows=150_000)

    val_df = train_df.loc[X_val.index]
    val_benign_df = val_df[val_df[label_col] == "BENIGN"]
    if len(val_benign_df) > 200_000:
        val_benign_df = val_benign_df.sample(n=200_000, random_state=42)

    X_val_benign = val_benign_df[FEATURES]

    benign_df = _df[_df[label_col] == "BENIGN"]
    target_attack = min(int(len(attack_df)), 50_000)
    attack_df = attack_df.sample(n=target_attack, random_state=42)
    benign_df = benign_df.sample(n=target_attack, random_state=42)
    test_df = cudf.concat([attack_df, benign_df], ignore_index=True)

    X_test = test_df[FEATURES]
    y_test = test_df[label_bin].astype("int32")

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg / (pos + 1e-9))

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 256,
        "scale_pos_weight": scale_pos_weight,
    }

    train_dm = xgb.DMatrix(X_train.values, label=y_train.to_cupy())
    val_dm = xgb.DMatrix(X_val.values, label=y_val.to_cupy())
    val_benign_dm = xgb.DMatrix(X_val_benign.values)
    test_dm = xgb.DMatrix(X_test.values, label=y_test.to_cupy())

    bst = xgb.train(
        params,
        train_dm,
        num_boost_round=300,
        evals=[(val_dm, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    val_benign_probs = cupy_to_torch(cp.asarray(bst.predict(val_benign_dm))).float()
    test_probs = cupy_to_torch(cp.asarray(bst.predict(test_dm))).float()
    y_test_t = cupy_to_torch(y_test.to_cupy()).int()

    thresholds = {
        "fpr_1e-3": threshold_by_benign_fpr(val_benign_probs, 1e-3),
        "fpr_1e-4": threshold_by_benign_fpr(val_benign_probs, 1e-4),
    }

    res = {
        "holdout_attack": holdout_attack,
        "holdout_rows": int(len(attack_df)),
        "test_rows": int(len(test_df)),
        "train_rows": int(len(y_train)),
        "val_benign_rows": int(len(val_benign_df)),
        "best_iteration": int(bst.best_iteration),
        "thresholds": thresholds,
        "test_metrics@0.5": eval_metrics(y_test_t, test_probs, 0.5),
        "test_metrics@fpr_1e-3": eval_metrics(y_test_t, test_probs, thresholds["fpr_1e-3"]),
        "test_metrics@fpr_1e-4": eval_metrics(y_test_t, test_probs, thresholds["fpr_1e-4"]),
    }

    results.append(res)

out_path = os.path.join(REPORT_DIR, "drift_holdout_xgb_benign_thresholds_feat_subset.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
