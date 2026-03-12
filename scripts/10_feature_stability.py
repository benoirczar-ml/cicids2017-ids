import os
import json
import cudf
import cupy as cp

DATA_PATH = "/srv/work/datasets/cicids2017/processed/cicids2017_clean.parquet"
REPORT_DIR = "/srv/work/projects/cicids2017-ids/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

_df = cudf.read_parquet(DATA_PATH)
label_col = "Label"
label_bin = "Label_Binary"

holdout_attacks = ["Botnet", "Web Attack - Brute Force", "Infiltration - Portscan"]

feature_cols = [c for c in _df.columns if c not in [label_col, label_bin]]

benign_df = _df[_df[label_col] == "BENIGN"][feature_cols]

results = []
for attack in holdout_attacks:
    attack_df = _df[_df[label_col] == attack][feature_cols]
    if len(attack_df) == 0:
        continue

    # sample for speed/VRAM
    if len(benign_df) > 300_000:
        benign_s = benign_df.sample(n=300_000, random_state=42)
    else:
        benign_s = benign_df
    if len(attack_df) > 300_000:
        attack_s = attack_df.sample(n=300_000, random_state=42)
    else:
        attack_s = attack_df

    b = benign_s.values
    a = attack_s.values

    # per-feature mean and std
    b_mean = cp.mean(b, axis=0)
    b_std = cp.std(b, axis=0)
    b_std = cp.where(b_std == 0, 1.0, b_std)

    a_mean = cp.mean(a, axis=0)

    # standardized mean shift (approx effect size)
    z = cp.abs(a_mean - b_mean) / b_std

    # collect
    z_host = cp.asnumpy(z)
    idx = cp.asnumpy(cp.argsort(-z))
    top = []
    for i in idx[:20]:
        top.append({"feature": feature_cols[int(i)], "zshift": float(z_host[int(i)])})

    results.append({
        "attack": attack,
        "benign_rows": int(len(benign_s)),
        "attack_rows": int(len(attack_s)),
        "top_shifted_features": top,
    })

out_path = os.path.join(REPORT_DIR, "feature_stability_zshift.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(results)
