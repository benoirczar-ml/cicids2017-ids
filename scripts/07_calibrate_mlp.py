import os
import json
import cudf
import cupy as cp
import torch
from cuml.model_selection import train_test_split

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"
MODEL_PATH = os.path.join(REPORT_DIR, "torch_mlp_state.pt")

os.makedirs(REPORT_DIR, exist_ok=True)


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


def best_threshold_f1(y_true, y_score):
    thresholds = torch.linspace(0.01, 0.99, 99, device=y_true.device)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        preds = (y_score >= t).int()
        tn, fp, fn, tp = confusion_matrix_binary_torch(y_true, preds)
        precision, recall, f1 = precision_recall_f1_torch(tn, fp, fn, tp)
        f1_val = f1.item()
        if f1_val > best["f1"]:
            best = {
                "threshold": float(t.item()),
                "f1": f1_val,
                "precision": precision.item(),
                "recall": recall.item(),
            }
    return best


# Load data
_df = cudf.read_parquet(DATA_PATH)
label_col = "Label_Binary"
X = _df.drop(columns=["Label", label_col])
y = _df[label_col].astype("int32")

# Split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

# Sample to fit VRAM (match training)
X_train, y_train = stratified_sample(X_train, y_train, target_rows=500_000)
X_val, y_val = stratified_sample(X_val, y_val, target_rows=200_000)
X_test, y_test = stratified_sample(X_test, y_test, target_rows=200_000)

# Standardize (GPU)
X_train_cp = X_train.values
X_val_cp = X_val.values
X_test_cp = X_test.values

mean = cp.mean(X_train_cp, axis=0)
std = cp.std(X_train_cp, axis=0)
std = cp.where(std == 0, 1.0, std)

X_val_cp = (X_val_cp - mean) / std
X_test_cp = (X_test_cp - mean) / std

X_val_t = cupy_to_torch(X_val_cp).float()
X_test_t = cupy_to_torch(X_test_cp).float()
y_val_t = cupy_to_torch(y_val.to_cupy()).float()
y_test_t = cupy_to_torch(y_test.to_cupy()).float()

in_dim = X_val_t.shape[1]
model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(128, 1),
).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    val_logits = model(X_val_t).squeeze(1)
    test_logits = model(X_test_t).squeeze(1)

# Temperature scaling on val
temperature = torch.nn.Parameter(torch.ones(1, device=val_logits.device))
optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

def _loss():
    logits = val_logits / temperature
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y_val_t)

def _closure():
    optimizer.zero_grad()
    loss = _loss()
    loss.backward()
    return loss

optimizer.step(_closure)

with torch.no_grad():
    val_probs = torch.sigmoid(val_logits / temperature)
    test_probs = torch.sigmoid(test_logits / temperature)

val_best = best_threshold_f1(y_val_t.int(), val_probs)

def eval_with_threshold(y_true, y_score, threshold):
    preds = (y_score >= threshold).int()
    tn, fp, fn, tp = confusion_matrix_binary_torch(y_true, preds)
    precision, recall, f1 = precision_recall_f1_torch(tn, fp, fn, tp)
    roc = roc_auc_torch(y_true, y_score)
    ap = average_precision_torch(y_true, y_score)
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "roc_auc": roc,
        "pr_auc": ap,
    }

val_metrics = eval_with_threshold(y_val_t.int(), val_probs, 0.5)
test_metrics_05 = eval_with_threshold(y_test_t.int(), test_probs, 0.5)
test_metrics_best = eval_with_threshold(y_test_t.int(), test_probs, val_best["threshold"])

report = {
    "temperature": float(temperature.item()),
    "val_metrics@0.5": val_metrics,
    "val_best_f1": val_best,
    "test_metrics@0.5": test_metrics_05,
    "test_metrics@val_best_f1_threshold": test_metrics_best,
    "threshold_used": val_best["threshold"],
}

with open(os.path.join(REPORT_DIR, "torch_mlp_calibrated.json"), "w") as f:
    json.dump(report, f, indent=2)

print(report)
