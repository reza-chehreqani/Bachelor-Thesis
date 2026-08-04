import pandas as pd
from plot_style import *

eval_df = pd.read_pickle("eval_df.pkl")
train_df = pd.read_pickle("train_df.pkl")

baseline = eval_df[eval_df.condition == "baseline"][["layer","auroc_sae"]].drop_duplicates().set_index("layer")
trained = eval_df[eval_df.condition != "baseline"].copy()
trained = trained.merge(baseline.rename(columns={"auroc_sae":"auroc_sae_base"}), left_on="layer", right_index=True)
trained["auroc_sae_gain"] = trained["auroc_sae"] - trained["auroc_sae_base"]
target_only = trained[trained.is_target_layer]
run_level_gain = target_only.groupby(["layer_setting","condition","lam"])["auroc_sae_gain"].mean().reset_index()

final_step = train_df[train_df.step == 499][["layer_setting","condition","lam","perplexity","sae_variance_explained"]]

merged = run_level_gain.merge(final_step, on=["layer_setting","condition","lam"], how="left")
print(merged.shape)
print(merged[["perplexity"]].describe())

fig, ax = plt.subplots(figsize=(7.5, 5.2))

for cond in ["C2_raw_invariance", "C3_sae_magnitude", "C4_sae_magnitude_support", "C1_lm_only"]:
    sub = merged[merged.condition == cond]
    ax.scatter(sub["perplexity"], sub["auroc_sae_gain"], label=COND_LABELS[cond],
               color=COND_COLORS[cond], s=45, alpha=0.8, edgecolors="white", linewidths=0.5)

# highlight recommended config
rec = merged[(merged.layer_setting=="layer 5") & (merged.condition=="C4_sae_magnitude_support") & (merged.lam==0.3)]
ax.scatter(rec["perplexity"], rec["auroc_sae_gain"], s=280, facecolors='none', edgecolors='#1a1a1a', linewidths=2.2, zorder=5)
ax.annotate("recommended, stable\n(layer 5, C4, $\\lambda$=0.3)", xy=(rec["perplexity"].values[0], rec["auroc_sae_gain"].values[0]),
            xytext=(rec["perplexity"].values[0]+4, rec["auroc_sae_gain"].values[0]-0.0028),
            fontsize=8.5, ha="left", arrowprops=dict(arrowstyle="->", lw=1))

# highlight aggressive config
agg = merged[(merged.layer_setting=="layers 3, 4, 5") & (merged.condition=="C3_sae_magnitude") & (merged.lam==1.0)]
ax.scatter(agg["perplexity"], agg["auroc_sae_gain"], s=280, facecolors='none', edgecolors='#c0392b', linewidths=2.2, zorder=5)
ax.annotate("highest raw gain, unstable\n(3,4,5 / C3 / $\\lambda$=1.0)", xy=(agg["perplexity"].values[0], agg["auroc_sae_gain"].values[0]),
            xytext=(agg["perplexity"].values[0]-8, agg["auroc_sae_gain"].values[0]-0.003),
            fontsize=8.5, ha="right", arrowprops=dict(arrowstyle="->", lw=1))

ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("final-step WikiText perplexity (log scale) — lower is healthier")
ax.set_ylabel("mean $\\Delta$AUROC at target layer(s)")
ax.set_title("Trade-off: separation gain vs. language-modeling cost")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/tradeoff_gain_vs_perplexity.pdf")
print("saved tradeoff_gain_vs_perplexity")
