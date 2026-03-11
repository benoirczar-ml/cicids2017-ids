import os
import sys
import json
import cudf
import cupy as cp
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression

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

scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=400)
clf.fit(X_train_s, y_train)

# Predict
preds = clf.predict(X_test_s)

# For probabilities, fallback to decision_function if predict_proba fails
try:
    probs = clf.predict_proba(X_test_s)[:, 1]
except Exception:
    # decision_function gives raw scores; convert with sigmoid
    scores = clf.decision_function(X_test_s)
    probs = 1 / (1 + cp.exp(-scores))

# Ensure cupy arrays
if isinstance(preds, cudf.Series):
    preds = preds.to_cupy()
if isinstance(probs, cudf.Series):
    probs = probs.to_cupy()

y_true = y_test.to_cupy()

# Metrics

tn, fp, fn, tp = confusion_matrix_binary(y_true, preds)
precision, recall, f1 = precision_recall_f1(tn, fp, fn, tp)
roc = roc_auc(y_true, probs)
ap = average_precision(y_true, probs)

report = {
    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc,
    "pr_auc": ap,
}

with open(os.path.join(REPORT_DIR, "baseline_logreg_metrics.json"), "w") as f:
    json.dump(report, f, indent=2)

print(report)
