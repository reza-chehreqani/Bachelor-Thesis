import pandas as pd
from plot_style import *

train_df = pd.read_pickle("train_df.pkl")
final_step = train_df[train_df.step == 499]

conds = ["C2_raw_invariance","C3_sae_magnitude","C4_sae_magnitude_support"]
lambdas = [0.01, 0.03, 0.1, 0.3, 1.0]

fig, ax = plt.subplots(figsize=(6.8,4.3))
for c in conds:
    vals = [final_step[(final_step.condition==c)&(final_step.lam==l)]["perplexity"].mean() for l in lambdas]
    ax.plot(lambdas, vals, marker="o", label=COND_LABELS[c], color=COND_COLORS[c], linewidth=2)
c1_val = final_step[final_step.condition=="C1_lm_only"]["perplexity"].mean()
ax.axhline(c1_val, color=COND_COLORS["C1_lm_only"], linestyle="--", linewidth=1.5, label=COND_LABELS["C1_lm_only"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda$ (invariance loss weight)")
ax.set_ylabel("final-step WikiText validation perplexity (log scale)")
ax.set_title("Language-modeling cost of the invariance loss")
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/perplexity_vs_lambda.pdf")
print("saved perplexity_vs_lambda")
