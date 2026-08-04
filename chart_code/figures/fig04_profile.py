import pandas as pd
from plot_style import *

eval_df = pd.read_pickle("eval_df.pkl")
baseline = eval_df[eval_df.condition=="baseline"][["layer","auroc_sae","auroc_raw"]].drop_duplicates().sort_values("layer")
best = eval_df[(eval_df.layer_setting=="layers 3, 4, 5") & (eval_df.condition=="C3_sae_magnitude") & (eval_df.lam==1.0)].sort_values("layer")

fig, ax = plt.subplots(figsize=(7, 4.3))
x = baseline["layer"].values
ax.plot(x, baseline["auroc_sae"], marker="o", color="#7f8c8d", linewidth=2, label="baseline (pretrained)")
ax.plot(x, best["auroc_sae"], marker="s", color="#3d7dcb", linewidth=2, label="trained (C3, layers={3,4,5}, $\\lambda$=1.0)")
ax.axvspan(2.5, 5.5, color="#3d7dcb", alpha=0.10, label="invariance-target layer")
ax.axhline(0.5, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("AUROC")
ax.set_xticks(x)
ax.set_ylim(0.5, 0.7)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("Full-depth profile: baseline vs. recommended configuration")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/profile_best_vs_baseline.pdf")
print("saved, overwriting previous version")
