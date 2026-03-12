# CICIDS2017 IDS (GPU + Parquet)

Recruiter‑grade IDS project using CICIDS2017 with a **full GPU + Parquet** pipeline.

## Dataset
We use the CICIDS2017 dataset (UNB/CIC). The current run uses the Hugging Face parquet conversion
`bvsam/cic-ids-2017` (traffic_labels) because it includes true timestamps and IPs for time‑aware
windowing. We keep the earlier `bvk/CICIDS-2017` conversion for fast iteration, but it only stores
`MM:SS.s` in `Timestamp`, so it cannot support real temporal splits.

Paths:
- Raw parquet (timestamped): `/srv/work/datasets/cicids2017/raw/hf_bvsam/traffic_labels`
- Partitioned raw parquet: `/srv/work/datasets/cicids2017/processed/partitioned_raw`
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
7. `scripts/26_prepare_dataset_bvsam.py` → rebuild clean parquet from timestamped HF source
8. `scripts/27_window_aggregate_ts_v1.py` → 1s windows using real timestamps
9. `scripts/28_faiss_knn_windows_ts_v1.py` → FAISS kNN on true‑timestamp windows
10. `scripts/29_build_partitioned_raw_parquet.py` → partition raw parquet for GPU‑friendly ingestion

## Results (current)
LogReg (CPU baseline, 800k stratified sample; GPU blocked by NVRTC fp8 headers):
- precision 0.8993, recall 0.8633, f1 0.8809, roc_auc 0.9862, pr_auc 0.9684

XGBoost (sampled 500k train / 200k val / 200k test):
- precision 0.9966, recall 0.9992, f1 0.9979, roc_auc 0.999961, pr_auc 0.999917

Torch MLP (sampled 500k / 200k / 200k):
- precision 0.8821, recall 0.9866, f1 0.9315, roc_auc 0.99623, pr_auc 0.99064

Drift holdout (LogReg, mixed test with benign + held‑out attack):
- Botnet: ROC‑AUC 0.815, PR‑AUC 0.707 (precision/recall 0.0)
- Web Attack Brute Force: ROC‑AUC 0.039, PR‑AUC 0.317 (precision/recall 0.0)
- Infiltration Portscan: ROC‑AUC 0.899, PR‑AUC 0.899 (precision 0.992, recall 0.556)

Drift robustness (XGBoost + threshold tuning on in‑distribution val):
- Botnet: ROC‑AUC 0.996, PR‑AUC 0.996 but recall 0.0 at thresholds tuned on val
- Web Attack Brute Force: precision 1.0, recall 0.329 at threshold 0.5
- Infiltration Portscan: precision ~0.9999, recall 0.551 at best‑F1 threshold (val‑tuned)

Benign‑only thresholding (no attack leakage, thresholds set by benign FPR):
- Web Attack Brute Force: recall 0.986 at FPR 1e‑3 (precision 1.0)
- Infiltration Portscan: recall 0.983 at FPR 1e‑3 (precision ~0.9992)
- Botnet: recall still low at FPR 1e‑3 (drift in score scale)

Feature stability lab:
- Computed benign vs attack z‑shift per feature (report saved).
- A “stable subset” of 17 shared‑shift features **hurt** drift detection (Botnet/WebBF recall dropped to 0). We do not keep this subset.

Protocol‑segmented benign thresholds:
- Web Attack Brute Force and Portscan stayed strong.
- Botnet still collapsed (protocol thresholds did not help). Indicates need for open‑set/anomaly handling for this class.

Anomaly (benign‑only) baselines:
- PCA reconstruction error: very low recall across holdout attacks (not sufficient).
- Mahalanobis distance: similarly low recall (not sufficient).
Autoencoder (benign‑only):
- Still low recall on Botnet (≈0.02–0.04 at FPR 1e‑3).
- Weak on WebBF unless fused.

Fusion (AE + XGB, benign FPR thresholds):
- WebBF: recall ≈0.986 at FPR 1e‑3 (precision 1.0).
- Portscan: recall ≈0.982 at FPR 1e‑3 (precision ~0.9991).
- Botnet: recall ≈0.023 at FPR 1e‑3 (still poor).

Deep SVDD (benign‑only):
- Botnet recall ~0.033 at FPR 1e‑3 (still poor).
- WebBF/Portscan still near‑zero recall at strict FPR.

Notes:
- cuML IsolationForest not available in this build.
- cuML kNN failed due to CUDA NVRTC compile errors (fp8 headers). We need a CUDA/toolkit fix to use it.

FAISS GPU kNN anomaly (separate env `/srv/work/envs/faiss-gpu-py310`):
- Botnet recall ~0.058 at FPR 1e‑3 (better, still low).
- Portscan recall ~0.144 at FPR 1e‑3.
- WebBF still 0 at strict FPR.

Windowed aggregation (true timestamps from `bvsam/cic-ids-2017`):
- 1s windows by `Source IP` and real `Timestamp`.
- FAISS kNN (benign‑only):
  - Bot: recall 0.003 @ FPR 1e‑3 (ROC‑AUC 0.812, PR‑AUC 0.772)
  - Web Attack Brute Force: recall 0.0 @ FPR 1e‑3 (ROC‑AUC 0.952, PR‑AUC 0.900)
  - PortScan: recall 0.407 @ FPR 1e‑3 (precision ~1.0)
  - Infiltration: recall 0.714 @ FPR 1e‑3 (precision ~1.0)

## Next steps
- Improve drift robustness (feature filters + calibration + class‑wise thresholds)
- Time‑aware split (when using full CSVs with true timestamps)
- Report + plots
