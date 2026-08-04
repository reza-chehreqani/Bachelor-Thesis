import pandas as pd
from plot_style import *

eval_df = pd.read_pickle("eval_df.pkl")
baseline = eval_df[eval_df.condition=="baseline"][["layer","top5pct_latent_share"]].drop_duplicates().sort_values("layer")
best = eval_df[(eval_df.layer_setting=="layer 5") & (eval_df.condition=="C4_sae_magnitude_support") & (eval_df.lam==0.3)].sort_values("layer")

fig, ax = plt.subplots(figsize=(6.8,4.2))
x = baseline["layer"].values
w=0.35
ax.bar(x-w/2, baseline["top5pct_latent_share"], w, color="#7f8c8d", label="baseline (pretrained)")
ax.bar(x+w/2, best["top5pct_latent_share"], w, color="#3d7dcb", label="trained (C4, target={5}, $\\lambda$=0.3)")
ax.axvspan(4.5, 5.5, color="#3d7dcb", alpha=0.08)
ax.set_xlabel("layer")
ax.set_ylabel("share of latent \"flips\" in top 5\\% of features")
ax.set_xticks(x)
ax.set_title("Concentration of the same/diff-meaning signal across SAE latents")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/latent_concentration.pdf")
print("saved, overwriting previous version")

# print exact values for text
print(baseline.set_index("layer")["top5pct_latent_share"].round(4))
print(best.set_index("layer")["top5pct_latent_share"].round(4))
