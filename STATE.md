# Project State Snapshot

Date: 2026-03-12

## Git
- Repo: `https://github.com/benoirczar-ml/cicids2017-ids.git`
- Branch: `main`
- Commit: `ffe16f75eed460265b71828012de8bab75851aba`

## Datasets (current sources)
- Raw timestamped parquet (HF bvsam, traffic_labels):
  - `/srv/work/datasets/cicids2017/raw/hf_bvsam/traffic_labels` (≈292M)
- Partitioned raw parquet (GPU‑friendly ingestion):
  - `/srv/work/datasets/cicids2017/processed/partitioned_raw` (≈365M)
- Clean parquet (timestamped source, leakage dropped):
  - `/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet` (≈318M)
- Windowed aggregates (true timestamps, 1s):
  - `/srv/work/datasets/cicids2017/processed/windows_1s_ts_v1` (≈12M)

## Key scripts
- Build partitioned raw parquet:
  - `scripts/29_build_partitioned_raw_parquet.py`
- Rebuild clean dataset:
  - `scripts/26_prepare_dataset_bvsam.py`
- Windowed aggregates (true timestamps):
  - `scripts/27_window_aggregate_ts_v1.py`
- FAISS kNN on windows:
  - `scripts/28_faiss_knn_windows_ts_v1.py`

## Reports
- Baselines:
  - `reports/baseline_logreg.json`
  - `reports/baseline_logreg_metrics.json`
  - `reports/xgb_metrics.json`
  - `reports/torch_mlp_metrics.json`
- FAISS windowed anomaly:
  - `reports/anomaly_faiss_knn_windows_1s_ts_v1.json`

## Environments
- RAPIDS: `/srv/work/envs/cicids2017-rapids`
- FAISS: `/srv/work/envs/faiss-gpu-py310`

## Known issues
- cuML/LogReg on GPU blocked by NVRTC fp8 header compile error in the RAPIDS env.
  CPU fallback used for LogReg baseline.
