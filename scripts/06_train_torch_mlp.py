import os
import json
import cudf
import cupy as cp
import torch
from cuml.model_selection import train_test_split

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

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

# Load data
_df = cudf.read_parquet(DATA_PATH)
label_col = "Label_Binary"
X = _df.drop(columns=["Label", label_col])
y = _df[label_col].astype("int32")

# Split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

# Sample to fit VRAM
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

X_train_cp = (X_train_cp - mean) / std
X_val_cp = (X_val_cp - mean) / std
X_test_cp = (X_test_cp - mean) / std

# Convert to torch (GPU) via DLPack

def cupy_to_torch(x):
    # cupy implements __dlpack__; this avoids deprecated toDlpack()
    return torch.utils.dlpack.from_dlpack(x)

X_train_t = cupy_to_torch(X_train_cp).float()
X_val_t = cupy_to_torch(X_val_cp).float()
X_test_t = cupy_to_torch(X_test_cp).float()

y_train_t = cupy_to_torch(y_train.to_cupy()).float()
y_val_t = cupy_to_torch(y_val.to_cupy()).float()
y_test_t = cupy_to_torch(y_test.to_cupy()).float()

# DataLoader
batch_size = 4096
train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)
test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
in_dim = X_train_t.shape[1]
model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(128, 1),
).cuda()

# Loss with class imbalance
pos = y_train_t.sum()
neg = len(y_train_t) - pos
pos_weight = neg / (pos + 1e-9)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Train
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / max(1, len(train_loader))
    # simple val loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            val_loss += loss.item()
    val_loss = val_loss / max(1, len(val_loader))
    print(f"epoch={epoch} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")

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


# Eval on test
model.eval()
probs_list = []
y_list = []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb).squeeze(1)
        probs = torch.sigmoid(logits)
        probs_list.append(probs)
        y_list.append(yb)

y_score = torch.cat(probs_list)
y_true = torch.cat(y_list).int()

preds = (y_score >= 0.5).int()

tn, fp, fn, tp = confusion_matrix_binary_torch(y_true, preds)
precision, recall, f1 = precision_recall_f1_torch(tn, fp, fn, tp)
roc = roc_auc_torch(y_true, y_score)
ap = average_precision_torch(y_true, y_score)

report = {
    "train_rows": int(len(y_train_t)),
    "val_rows": int(len(y_val_t)),
    "test_rows": int(len(y_test_t)),
    "precision": precision.item(),
    "recall": recall.item(),
    "f1": f1.item(),
    "roc_auc": roc,
    "pr_auc": ap,
}

with open(os.path.join(REPORT_DIR, "torch_mlp_metrics.json"), "w") as f:
    json.dump(report, f, indent=2)

# Save model
model_path = os.path.join(REPORT_DIR, "torch_mlp_state.pt")
torch.save(model.state_dict(), model_path)

print(report)
