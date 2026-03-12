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


# Train benign-only autoencoder once
benign_full = _df[_df[label_col] == "BENIGN"]
benign_feat = benign_full.drop(columns=[label_col, label_bin])
X_train_b, X_val_b = train_test_split(benign_feat, test_size=0.25, random_state=42)
if len(X_train_b) > 300_000:
    X_train_b = X_train_b.sample(n=300_000, random_state=42)
if len(X_val_b) > 150_000:
    X_val_b = X_val_b.sample(n=150_000, random_state=42)

X_train_cp = X_train_b.values
mean = cp.mean(X_train_cp, axis=0)
std = cp.std(X_train_cp, axis=0)
std = cp.where(std == 0, 1.0, std)

X_train_cp = (X_train_cp - mean) / std
X_val_cp = (X_val_b.values - mean) / std

X_train_t = cupy_to_torch(X_train_cp).float()
X_val_t = cupy_to_torch(X_val_cp).float()

in_dim = X_train_t.shape[1]
ae = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, in_dim),
).cuda()

opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)
crit = torch.nn.MSELoss()

train_loader = torch.utils.data.DataLoader(X_train_t, batch_size=4096, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(X_val_t, batch_size=4096, shuffle=False, num_workers=0)

for epoch in range(1, 6):
    ae.train()
    total_loss = 0.0
    for xb in train_loader:
        opt.zero_grad(set_to_none=True)
        recon = ae(xb)
        loss = crit(recon, xb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / max(1, len(train_loader))

    ae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb in val_loader:
            recon = ae(xb)
            loss = crit(recon, xb)
            val_loss += loss.item()
    val_loss = val_loss / max(1, len(val_loader))
    print(f"ae_epoch={epoch} train_loss={avg_loss:.6f} val_loss={val_loss:.6f}")

# AE benign validation scores
with torch.no_grad():
    recon_val = ae(X_val_t)
    ae_val_scores = torch.mean((X_val_t - recon_val) ** 2, dim=1)

# free training tensors to reduce GPU pressure for XGB
del X_train_t, X_val_t, X_train_cp, X_val_cp
torch.cuda.empty_cache()

results = []
for holdout_attack in holdout_attacks:
    mask_holdout = _df[label_col] == holdout_attack
    attack_df = _df[mask_holdout]
    # avoid large boolean mask on full df; sample first, then filter
    sample_df = _df.sample(n=min(len(_df), 400_000), random_state=42)
    train_df = sample_df[sample_df[label_col] != holdout_attack]
    if len(attack_df) == 0:
        continue

    # Reduce before split to avoid GPU OOM
    X_all = train_df.drop(columns=[label_col, label_bin])
    y_all = train_df[label_bin].astype("int32")
    if len(X_all) > 300_000:
        X_all = X_all.sample(n=300_000, random_state=42)
        y_all = y_all.loc[X_all.index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )

    # XGB model
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
        "max_bin": 128,
        "scale_pos_weight": scale_pos_weight,
    }

    train_dm = xgb.DMatrix(X_train.values, label=y_train.to_cupy())
    val_dm = xgb.DMatrix(X_val.values, label=y_val.to_cupy())

    bst = xgb.train(
        params,
        train_dm,
        num_boost_round=300,
        evals=[(val_dm, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # Benign val for fusion thresholds (no leakage)
    val_df = train_df.loc[X_val.index]
    val_benign_df = val_df[val_df[label_col] == "BENIGN"]
    if len(val_benign_df) > 200_000:
        val_benign_df = val_benign_df.sample(n=200_000, random_state=42)

    # XGB benign scores
    xgb_val_probs = cupy_to_torch(cp.asarray(bst.predict(xgb.DMatrix(val_benign_df.drop(columns=[label_col, label_bin]).values)))).float()

    # AE benign scores (use same mean/std)
    val_benign_cp = (val_benign_df.drop(columns=[label_col, label_bin]).values - mean) / std
    val_benign_t = cupy_to_torch(val_benign_cp).float()
    with torch.no_grad():
        recon = ae(val_benign_t)
        ae_val_benign_scores = torch.mean((val_benign_t - recon) ** 2, dim=1)

    # Normalize scores using benign stats
    xgb_mu, xgb_std = xgb_val_probs.mean(), xgb_val_probs.std() + 1e-6
    ae_mu, ae_std = ae_val_benign_scores.mean(), ae_val_benign_scores.std() + 1e-6

    xgb_val_z = (xgb_val_probs - xgb_mu) / xgb_std
    ae_val_z = (ae_val_benign_scores - ae_mu) / ae_std
    fusion_val = torch.maximum(xgb_val_z, ae_val_z)

    thr_fpr_1e3 = threshold_by_benign_fpr(fusion_val, 1e-3)
    thr_fpr_1e4 = threshold_by_benign_fpr(fusion_val, 1e-4)

    # Test set
    target_attack = min(int(len(attack_df)), 50_000)
    attack_s = attack_df.sample(n=target_attack, random_state=42)
    benign_s = benign_full.sample(n=target_attack, random_state=42)
    test_df = cudf.concat([attack_s, benign_s], ignore_index=True)

    X_test = test_df.drop(columns=[label_col, label_bin])
    y_test = test_df[label_bin].astype("int32")

    xgb_test_probs = cupy_to_torch(cp.asarray(bst.predict(xgb.DMatrix(X_test.values)))).float()

    X_test_cp = (X_test.values - mean) / std
    X_test_t = cupy_to_torch(X_test_cp).float()
    with torch.no_grad():
        recon_test = ae(X_test_t)
        ae_test_scores = torch.mean((X_test_t - recon_test) ** 2, dim=1)

    xgb_test_z = (xgb_test_probs - xgb_mu) / xgb_std
    ae_test_z = (ae_test_scores - ae_mu) / ae_std
    fusion_test = torch.maximum(xgb_test_z, ae_test_z)

    y_test_t = cupy_to_torch(y_test.to_cupy()).int()

    res = {
        "holdout_attack": holdout_attack,
        "holdout_rows": int(len(attack_s)),
        "test_rows": int(len(test_df)),
        "train_rows": int(len(y_train)),
        "val_benign_rows": int(len(val_benign_df)),
        "thresholds": {"fpr_1e-3": float(thr_fpr_1e3), "fpr_1e-4": float(thr_fpr_1e4)},
        "test_metrics@fpr_1e-3": eval_metrics(y_test_t, fusion_test, thr_fpr_1e3),
        "test_metrics@fpr_1e-4": eval_metrics(y_test_t, fusion_test, thr_fpr_1e4),
        "xgb_test_metrics@0.5": eval_metrics(y_test_t, xgb_test_probs, 0.5),
    }

    results.append(res)

out_path = os.path.join(REPORT_DIR, "fusion_ae_xgb_benign_thresholds.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
