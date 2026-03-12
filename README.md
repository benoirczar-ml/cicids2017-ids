# CICIDS2017 IDS (GPU + Parquet)

Recruiter‑grade IDS project using CICIDS2017 with a **full GPU + Parquet** pipeline.

## Dataset
We use the CICIDS2017 dataset (UNB/CIC). The current run uses the Hugging Face parquet conversion (`bvk/CICIDS-2017`) for fast GPU‑first iteration. This is a convenient subset/packaging of the original CSVs, so results are not directly comparable to the full competition setup.

Paths:
- Raw parquet: `/srv/work/datasets/cicids2017/parquet/hf-bvk`
- Processed parquet: `/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet`

## Rules
See [PROJECT_RULES.md](PROJECT_RULES.md).

## Environment
RAPIDS env (GPU ETL):
```bash
micromamba activate /srv/work/envs/cicids2017-rapids
```

## Pipeline (current)
1. `scripts/00_inspect_parquet.py` → schema + label counts
2. `scripts/01_prepare_dataset.py` → drop leakage cols, fill NaNs, save clean parquet
3. `scripts/02_train_baseline.py` + `scripts/03_eval_logreg.py` → LogReg baseline
4. `scripts/04_train_xgb.py` → XGBoost GPU baseline
5. `scripts/05_drift_holdout.py` → attack holdout drift check
6. `scripts/06_train_torch_mlp.py` → torch MLP on GPU (sampled for VRAM)

## Results (current)
LogReg (full dataset):
- precision 0.9823, recall 0.9867, f1 0.9845, roc_auc 0.9986, pr_auc 0.9962

XGBoost (sampled 500k train / 200k val / 200k test):
- precision 0.99984, recall 0.99976, f1 0.99980, roc_auc 0.9999999, pr_auc 0.9999998

Torch MLP (sampled 500k / 200k / 200k):
- precision 0.99619, recall 0.99852, f1 0.99735, roc_auc 0.9998868, pr_auc 0.9997905
- calibrated (temperature 0.9249, threshold 0.72): precision 0.99813, recall 0.99785, f1 0.99799

Drift holdout (LogReg, mixed test with benign + held‑out attack):
- Botnet: ROC‑AUC 0.815, PR‑AUC 0.707 (precision/recall 0.0)
- Web Attack Brute Force: ROC‑AUC 0.039, PR‑AUC 0.317 (precision/recall 0.0)
- Infiltration Portscan: ROC‑AUC 0.899, PR‑AUC 0.899 (precision 0.992, recall 0.556)

Drift robustness (XGBoost + threshold tuning on in‑distribution val):
- Botnet: ROC‑AUC 0.996, PR‑AUC 0.996 but recall 0.0 at thresholds tuned on val
- Web Attack Brute Force: precision 1.0, recall 0.329 at threshold 0.5
- Infiltration Portscan: precision ~0.9999, recall 0.551 at best‑F1 threshold (val‑tuned)

## Next steps
- Improve drift robustness (feature filters + calibration + class‑wise thresholds)
- Time‑aware split (when using full CSVs with true timestamps)
- Report + plots
