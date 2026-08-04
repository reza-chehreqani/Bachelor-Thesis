import pandas as pd
import numpy as np
from plot_style import *

train_df = pd.read_pickle("train_df.pkl")
final_step = train_df[train_df.step == 499]

# heatmap: mean final sae_variance_explained, condition x lambda
conds = ["C2_raw_invariance","C3_sae_magnitude","C4_sae_magnitude_support"]
lambdas = [0.01, 0.03, 0.1, 0.3, 1.0]
mat = np.zeros((len(conds), len(lambdas)))
for i, c in enumerate(conds):
    for j, l in enumerate(lambdas):
        mat[i,j] = final_step[(final_step.condition==c)&(final_step.lam==l)]["sae_variance_explained"].mean()

fig, ax = plt.subplots(figsize=(6.5,3.6))
im = ax.imshow(mat, cmap="RdYlGn", vmin=0.85, vmax=0.98, aspect="auto")
ax.set_xticks(range(len(lambdas))); ax.set_xticklabels([str(l) for l in lambdas])
ax.set_yticks(range(len(conds))); ax.set_yticklabels(["C2 (raw)","C3 (SAE)","C4 (SAE+support)"])
ax.set_xlabel("$\\lambda$")
ax.set_title("Final SAE reconstruction fidelity (mean over layer-choices)")
for i in range(len(conds)):
    for j in range(len(lambdas)):
        ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=9,
                 color="white" if mat[i,j]<0.91 else "black")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("sae variance explained")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fidelity_heatmap.pdf")
print("saved")
