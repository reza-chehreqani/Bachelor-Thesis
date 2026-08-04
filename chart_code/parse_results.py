"""Parse all results JSON files into consolidated pandas dataframes and pickle them."""
import json
import os
import re
import pandas as pd

RESULTS_DIR = "../results"

def parse_layer_setting(dirname: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", dirname)
    return tuple(int(n) for n in nums)

def parse_run_id(run_id: str):
    if run_id == "baseline":
        return dict(condition="baseline", lam=None)
    m = re.match(r"pythia-70m-deduped__(C\d_[a-z_]+)__lam([\d.]+)", run_id)
    assert m, run_id
    return dict(condition=m.group(1), lam=float(m.group(2)))

eval_rows = []
train_rows = []

for layer_setting_dir in sorted(os.listdir(RESULTS_DIR)):
    inv_layers = parse_layer_setting(layer_setting_dir)
    full_ls_path = os.path.join(RESULTS_DIR, layer_setting_dir)
    for run_id in sorted(os.listdir(full_ls_path)):
        run_path = os.path.join(full_ls_path, run_id)
        meta = parse_run_id(run_id)
        # eval
        eval_path = os.path.join(run_path, "eval", "pythia-70m-deduped_layer_stats.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                stats = json.load(f)
            for s in stats:
                row = dict(layer_setting=layer_setting_dir, inv_layers=inv_layers, run_id=run_id, **meta, **s)
                row["is_target_layer"] = s["layer"] in inv_layers
                eval_rows.append(row)
        # training log
        tlog_path = os.path.join(run_path, "training_log.json")
        if os.path.exists(tlog_path):
            with open(tlog_path) as f:
                tlog = json.load(f)
            for entry in tlog:
                row = dict(layer_setting=layer_setting_dir, inv_layers=inv_layers, run_id=run_id, **meta)
                row.update(entry)
                train_rows.append(row)

eval_df = pd.DataFrame(eval_rows)
train_df = pd.DataFrame(train_rows)

print("eval_df shape:", eval_df.shape)
print("train_df shape:", train_df.shape)
print(eval_df['condition'].unique())
print(eval_df['layer_setting'].unique())

eval_df.to_pickle("./figures/eval_df.pkl")
train_df.to_pickle("./figures/train_df.pkl")
eval_df.to_csv("./figures/eval_df.csv", index=False)
