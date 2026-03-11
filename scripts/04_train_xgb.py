import os
import sys
import json
import cudf
import cupy as cp
import xgboost as xgb
from cuml.model_selection import train_test_split

sys.path.append("/srv/work/projects/cicids2017-ids")
from src.metrics_gpu import confusion_matrix_binary, precision_recall_f1, roc_auc, average_precision

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

_df = cudf.read_parquet(DATA_PATH)
label_col = "Label_Binary"
X = _df.drop(columns=["Label", label_col])
y = _df[label_col].astype("int32")

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)


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

# Reduce sizes to fit VRAM
X_train, y_train = stratified_sample(X_train, y_train, target_rows=500_000)
X_val, y_val = stratified_sample(X_val, y_val, target_rows=200_000)
X_test, y_test = stratified_sample(X_test, y_test, target_rows=200_000)

# Compute scale_pos_weight
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

# DMatrix on GPU with cupy arrays
train_dm = xgb.DMatrix(X_train.values, label=y_train.to_cupy())
val_dm = xgb.DMatrix(X_val.values, label=y_val.to_cupy())
test_dm = xgb.DMatrix(X_test.values, label=y_test.to_cupy())

bst = xgb.train(
    params,
    train_dm,
    num_boost_round=300,
    evals=[(val_dm, "val")],
    early_stopping_rounds=30,
    verbose_eval=False,
)

probs = bst.predict(test_dm)
probs = cp.asarray(probs)
preds = (probs >= 0.5).astype(cp.int32)

y_true = y_test.to_cupy()

# Metrics

tn, fp, fn, tp = confusion_matrix_binary(y_true, preds)
precision, recall, f1 = precision_recall_f1(tn, fp, fn, tp)
roc = roc_auc(y_true, probs)
ap = average_precision(y_true, probs)

report = {
    "best_iteration": int(bst.best_iteration),
    "train_rows": int(len(y_train)),
    "val_rows": int(len(y_val)),
    "test_rows": int(len(y_test)),
    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc,
    "pr_auc": ap,
}

with open(os.path.join(REPORT_DIR, "xgb_metrics.json"), "w") as f:
    json.dump(report, f, indent=2)

print(report)
