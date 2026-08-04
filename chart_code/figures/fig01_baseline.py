import pandas as pd
from plot_style import *

eval_df = pd.read_pickle("eval_df.pkl")
baseline = eval_df[eval_df.condition=="baseline"][["layer","auroc_raw","auroc_sae"]].drop_duplicates().sort_values("layer")

fig, ax = plt.subplots(figsize=(6.5, 4))
x = baseline["layer"].values
w = 0.35
ax.bar(x - w/2, baseline["auroc_raw"], w, label="raw residual-stream space", color="#7f8c8d")
ax.bar(x + w/2, baseline["auroc_sae"], w, label="SAE latent space", color="#3d7dcb")
ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6, label="chance level")
ax.set_xlabel("layer")
ax.set_ylabel("AUROC (same- vs. diff-meaning discrimination)")
ax.set_xticks(x)
ax.set_ylim(0.45, 0.7)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Pretrained baseline: paraphrase discrimination by layer")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/baseline_auroc_by_layer.pdf")
print("saved")
