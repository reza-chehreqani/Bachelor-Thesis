import pandas as pd
from plot_style import *

eval_df = pd.read_pickle("eval_df.pkl")
baseline = eval_df[eval_df.condition == "baseline"][["layer","auroc_raw","auroc_sae"]].drop_duplicates().set_index("layer")
trained = eval_df[eval_df.condition != "baseline"].copy()
trained = trained.merge(baseline.rename(columns=lambda c: c+"_base"), left_on="layer", right_index=True)
trained["auroc_sae_gain"] = trained["auroc_sae"] - trained["auroc_sae_base"]
target_only = trained[trained.is_target_layer]

by_lambda = target_only.groupby(["condition","lam"])["auroc_sae_gain"].mean().reset_index()

fig, ax = plt.subplots(figsize=(6.8, 4.3))
for cond in ["C2_raw_invariance","C3_sae_magnitude","C4_sae_magnitude_support"]:
    sub = by_lambda[by_lambda.condition==cond].sort_values("lam")
    ax.plot(sub["lam"], sub["auroc_sae_gain"], marker="o", label=COND_LABELS[cond], color=COND_COLORS[cond], linewidth=2)
# C1 as a flat reference line (lam=0, single point) -- show as horizontal dashed line
c1_gain = by_lambda[by_lambda.condition=="C1_lm_only"]["auroc_sae_gain"].values[0]
ax.axhline(c1_gain, color=COND_COLORS["C1_lm_only"], linestyle="--", linewidth=1.5, label=COND_LABELS["C1_lm_only"])
ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (invariance loss weight)")
ax.set_ylabel("mean $\\Delta$AUROC at target layer(s)")
ax.set_title("Effect of invariance-loss weight on paraphrase separation")
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gain_vs_lambda.pdf")
print("saved")
