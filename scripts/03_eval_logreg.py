import os
import sys
import json

USE_CPU = os.environ.get("USE_CPU", "0") == "1"
if USE_CPU:
    import pandas as pd
    from sklearn.model_selection import train_test_split as sk_split
    from sklearn.preprocessing import StandardScaler as SkScaler
    from sklearn.linear_model import LogisticRegression as SkLogReg
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
else:
    import cudf
    import cupy as cp
    from cuml.model_selection import train_test_split
    from cuml.preprocessing import StandardScaler
    from cuml.linear_model import LogisticRegression

if not USE_CPU:
    sys.path.append("/srv/work/projects/cicids2017-ids")
    from src.metrics_gpu import confusion_matrix_binary, precision_recall_f1, roc_auc, average_precision

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

label_col = "Label_Binary"
if USE_CPU:
    _df = pd.read_parquet(DATA_PATH)
    X = _df.drop(columns=["Label", label_col])
    y = _df[label_col].astype("int32")
else:
    _df = cudf.read_parquet(DATA_PATH)
    X = _df.drop(columns=["Label", label_col])
    y = _df[label_col].astype("int32")

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

SAMPLE_ROWS = int(os.environ.get("SAMPLE_ROWS", "800000"))
if USE_CPU:
    if SAMPLE_ROWS < len(y):
        X, _, y, _ = sk_split(X, y, train_size=SAMPLE_ROWS, stratify=y, random_state=42)
else:
    X, y = stratified_sample(X, y, SAMPLE_ROWS)

if USE_CPU:
    X_train, X_tmp, y_train, y_tmp = sk_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = sk_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    scaler = SkScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = SkLogReg(max_iter=400, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    tn = int(((y_test == 0) & (preds == 0)).sum())
    fp = int(((y_test == 0) & (preds == 1)).sum())
    fn = int(((y_test == 1) & (preds == 0)).sum())
    tp = int(((y_test == 1) & (preds == 1)).sum())
else:
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=400)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    try:
        probs = clf.predict_proba(X_test_s)[:, 1]
    except Exception:
        scores = clf.decision_function(X_test_s)
        probs = 1 / (1 + cp.exp(-scores))
    if isinstance(preds, cudf.Series):
        preds = preds.to_cupy()
    if isinstance(probs, cudf.Series):
        probs = probs.to_cupy()
    y_true = y_test.to_cupy()
    tn, fp, fn, tp = confusion_matrix_binary(y_true, preds)
    precision, recall, f1 = precision_recall_f1(tn, fp, fn, tp)
    roc = roc_auc(y_true, probs)
    ap = average_precision(y_true, probs)

report = {
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc),
    "pr_auc": float(ap),
}

with open(os.path.join(REPORT_DIR, "baseline_logreg_metrics.json"), "w") as f:
    json.dump(report, f, indent=2)

print(report)
