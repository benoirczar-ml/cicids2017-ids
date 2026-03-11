import os
import json
import cudf
import cupy as cp
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# Load
_df = cudf.read_parquet(DATA_PATH)

label_col = "Label_Binary"

X = _df.drop(columns=["Label", label_col])
y = _df[label_col].astype("int32")

# Train/val/test split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

# Scale
scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Model
clf = LogisticRegression(max_iter=200, verbose=0)
clf.fit(X_train_s, y_train)

# Eval
val_pred = clf.predict(X_val_s)
test_pred = clf.predict(X_test_s)

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

report = {
    "val_accuracy": float(val_acc),
    "test_accuracy": float(test_acc),
    "rows": int(len(_df)),
    "features": int(X.shape[1]),
}

with open(os.path.join(REPORT_DIR, "baseline_logreg.json"), "w") as f:
    json.dump(report, f, indent=2)

print(report)
