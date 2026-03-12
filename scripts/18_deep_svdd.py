import os
import json
import cudf
import cupy as cp
import torch
from cuml.model_selection import train_test_split

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

_df = cudf.read_parquet(DATA_PATH)
label_col = "Label"
label_bin = "Label_Binary"

holdout_attacks = ["Botnet", "Web Attack - Brute Force", "Infiltration - Portscan"]


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


# Benign-only dataset
benign_df = _df[_df[label_col] == "BENIGN"].drop(columns=[label_col, label_bin])
X_train, X_val = train_test_split(benign_df, test_size=0.25, random_state=42)

if len(X_train) > 300_000:
    X_train = X_train.sample(n=300_000, random_state=42)
if len(X_val) > 150_000:
    X_val = X_val.sample(n=150_000, random_state=42)

# Standardize
X_train_cp = X_train.values
mean = cp.mean(X_train_cp, axis=0)
std = cp.std(X_train_cp, axis=0)
std = cp.where(std == 0, 1.0, std)

X_train_cp = (X_train_cp - mean) / std
X_val_cp = (X_val.values - mean) / std

X_train_t = cupy_to_torch(X_train_cp).float()
X_val_t = cupy_to_torch(X_val_cp).float()

# Deep SVDD encoder
in_dim = X_train_t.shape[1]
encoder = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
).cuda()

# Initialize center c
with torch.no_grad():
    c = encoder(X_train_t[:50000]).mean(dim=0)

optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

batch_size = 4096
train_loader = torch.utils.data.DataLoader(X_train_t, batch_size=batch_size, shuffle=True, num_workers=0)

# Train
epochs = 6
for epoch in range(1, epochs + 1):
    encoder.train()
    total_loss = 0.0
    for xb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        z = encoder(xb)
        loss = torch.mean((z - c) ** 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / max(1, len(train_loader))
    print(f"epoch={epoch} loss={avg_loss:.6f}")

# Validation scores
with torch.no_grad():
    z_val = encoder(X_val_t)
    val_scores = torch.mean((z_val - c) ** 2, dim=1)

thr_1e3 = threshold_by_benign_fpr(val_scores, 1e-3)
thr_1e4 = threshold_by_benign_fpr(val_scores, 1e-4)

# Evaluate on holdouts
results = []
for holdout_attack in holdout_attacks:
    attack_df = _df[_df[label_col] == holdout_attack]
    if len(attack_df) == 0:
        continue

    target_attack = min(int(len(attack_df)), 50_000)
    attack_s = attack_df.sample(n=target_attack, random_state=42)
    benign_s = _df[_df[label_col] == "BENIGN"].sample(n=target_attack, random_state=42)

    test_df = cudf.concat([attack_s, benign_s], ignore_index=True)
    X_test = test_df.drop(columns=[label_col, label_bin])
    y_test = test_df[label_bin].astype("int32")

    X_test_cp = (X_test.values - mean) / std
    X_test_t = cupy_to_torch(X_test_cp).float()
    with torch.no_grad():
        z_test = encoder(X_test_t)
        test_scores = torch.mean((z_test - c) ** 2, dim=1)

    y_test_t = cupy_to_torch(y_test.to_cupy()).int()

    res = {
        "holdout_attack": holdout_attack,
        "holdout_rows": int(len(attack_s)),
        "test_rows": int(len(test_df)),
        "benign_train_rows": int(len(X_train)),
        "benign_val_rows": int(len(X_val)),
        "thresholds": {"fpr_1e-3": thr_1e3, "fpr_1e-4": thr_1e4},
        "test_metrics@fpr_1e-3": eval_metrics(y_test_t, test_scores, thr_1e3),
        "test_metrics@fpr_1e-4": eval_metrics(y_test_t, test_scores, thr_1e4),
    }

    results.append(res)

out_path = os.path.join(REPORT_DIR, "anomaly_deep_svdd.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
