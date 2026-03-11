import os
import sys
import json
import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression

sys.path.append("/srv/work/projects/cicids2017-ids")
from src.metrics_gpu import confusion_matrix_binary, precision_recall_f1, roc_auc, average_precision

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

_df = cudf.read_parquet(DATA_PATH)
label_col = "Label"
label_bin = "Label_Binary"

# Holdout attacks list
holdout_attacks = ["Botnet", "Web Attack - Brute Force", "Infiltration - Portscan"]

results = []


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

for holdout_attack in holdout_attacks:
    mask_holdout = _df[label_col] == holdout_attack
    train_df = _df[~mask_holdout]
    attack_df = _df[mask_holdout]

    if len(attack_df) == 0:
        continue

    # Build test = holdout attack + benign sample
    benign_df = _df[_df[label_col] == "BENIGN"]
    # sample benign equal to holdout size (cap at 50k for VRAM)
    target_attack = min(int(len(attack_df)), 50_000)
    attack_df = attack_df.sample(n=target_attack, random_state=42)
    benign_df = benign_df.sample(n=target_attack, random_state=42)
    test_df = cudf.concat([attack_df, benign_df], ignore_index=True)

    X_train = train_df.drop(columns=[label_col, label_bin])
    y_train = train_df[label_bin].astype("int32")

    X_test = test_df.drop(columns=[label_col, label_bin])
    y_test = test_df[label_bin].astype("int32")

    # Sample train to fit VRAM
    X_train, y_train = stratified_sample(X_train, y_train, target_rows=300_000)

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

    results.append({
        "holdout_attack": holdout_attack,
        "holdout_rows": int(len(attack_df)),
        "test_rows": int(len(test_df)),
        "train_rows": int(len(y_train)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": ap,
    })

out_path = os.path.join(REPORT_DIR, "drift_holdout_logreg.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
