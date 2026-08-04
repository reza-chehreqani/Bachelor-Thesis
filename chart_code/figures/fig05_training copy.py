import pandas as pd
from plot_style import *

train_df = pd.read_pickle("train_df.pkl")

def get_run(layer_setting, condition, lam):
    sub = train_df[(train_df.layer_setting==layer_setting)&(train_df.condition==condition)&(train_df.lam==lam)].sort_values("step")
    return sub

stable = get_run("layer 5", "C4_sae_magnitude_support", 0.3)
unstable = get_run("layers 3, 4, 5", "C3_sae_magnitude", 1.0)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.75))

# panel 1: perplexity (log scale)
# ax = axes[0,0]
# ax.plot(stable["step"], stable["perplexity"], marker="o", color="#3d7dcb", label="layer{5}, C4, $\\lambda$=0.3 (stable)")
# ax.plot(unstable["step"], unstable["perplexity"], marker="s", color="#c0392b", label="layers{3,4,5}, C3, $\\lambda$=1.0 (unstable)")
# ax.set_yscale("log")
# ax.set_xlabel("training step")
# ax.set_ylabel("WikiText validation perplexity (log scale)")
# ax.set_title("(a) Language-modeling stability")
# ax.legend(fontsize=8, frameon=False)

# panel 2: auroc_sae (validation, PAWS)
ax = axes[0]
ax.plot(stable["step"], stable["auroc_sae"], marker="o", color="#3d7dcb", label="layer{5}, C4, $\\lambda$=0.3 (stable)")
ax.plot(unstable["step"], unstable["auroc_sae"], marker="s", color="#c0392b", label="layers{3,4,5}, C3, $\\lambda$=1.0 (unstable)")
ax.set_xlabel("training step")
ax.set_ylabel("validation AUROC")
ax.set_title("Paraphrase-separation progress")
ax.legend(fontsize=8, frameon=False)

# panel 3: sae_variance_explained
ax = axes[1]
ax.plot(stable["step"], stable["sae_variance_explained"], marker="o", color="#3d7dcb")
ax.plot(unstable["step"], unstable["sae_variance_explained"], marker="s", color="#c0392b")
ax.axhline(0.9, color="black", linestyle="--", linewidth=1, alpha=0.6, label="fidelity warning threshold")
ax.set_xlabel("training step")
ax.set_ylabel("SAE reconstruction variance explained")
ax.set_title("Frozen-SAE fidelity")
ax.legend(fontsize=8, frameon=False)

# panel 4: mean cosine distance, same vs diff meaning (SAE space) -- stable run
# ax = axes[1,1]
# ax.plot(stable["step"], stable["mean_same_sae_cos"], marker="o", color="#27ae60", label="same-meaning pairs")
# ax.plot(stable["step"], stable["mean_diff_sae_cos"], marker="o", color="#e67e22", label="diff-meaning pairs")
# ax.plot(unstable["step"], unstable["mean_same_sae_cos"], marker="s", linestyle="--", color="#27ae60", alpha=0.6)
# ax.plot(unstable["step"], unstable["mean_diff_sae_cos"], marker="s", linestyle="--", color="#e67e22", alpha=0.6)
# ax.set_xlabel("training step")
# ax.set_ylabel("mean cosine distance (SAE space)")
# ax.set_title("(d) Attractive/repulsive separation (solid=stable, dashed=unstable)")
# ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/training_dynamics.pdf")
print("saved")
