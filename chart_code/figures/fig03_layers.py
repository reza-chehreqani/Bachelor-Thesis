import pandas as pd
from plot_style import *

eval_df = pd.read_pickle("eval_df.pkl")
baseline = eval_df[eval_df.condition == "baseline"][["layer","auroc_sae"]].drop_duplicates().set_index("layer")
trained = eval_df[eval_df.condition != "baseline"].copy()
trained = trained.merge(baseline.rename(columns={"auroc_sae":"auroc_sae_base"}), left_on="layer", right_index=True)
trained["auroc_sae_gain"] = trained["auroc_sae"] - trained["auroc_sae_base"]
target_only = trained[trained.is_target_layer].copy()
target_only["n_inv_layers"] = target_only["inv_layers"].apply(len)

# --- Figure 3a: gain by number of simultaneous invariance layers (C3/C4 combined) ---
sub = target_only[target_only.condition.isin(["C3_sae_magnitude","C4_sae_magnitude_support"])]
by_n = sub.groupby("n_inv_layers")["auroc_sae_gain"].agg(["mean","sem"]).reset_index()

fig, ax = plt.subplots(figsize=(5.6, 4))
ax.bar(by_n["n_inv_layers"].astype(str), by_n["mean"], yerr=by_n["sem"], capsize=4, color="#3d7dcb", width=0.55)
ax.set_xlabel("number of layers carrying the invariance loss simultaneously")
ax.set_ylabel("mean $\\Delta$AUROC at target layer(s)")
ax.set_title("Stacking the invariance loss across layers")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gain_by_n_layers.pdf")

# --- Figure 3b: gain by WHICH single layer was chosen (C3/C4 combined) ---
single_layer_settings = [ls for ls in trained.layer_setting.unique() if len(trained[trained.layer_setting==ls].inv_layers.iloc[0])==1]
sl = target_only[(target_only.layer_setting.isin(single_layer_settings)) & (target_only.condition.isin(["C3_sae_magnitude","C4_sae_magnitude_support"]))]
by_layer = sl.groupby("layer")["auroc_sae_gain"].agg(["mean","min","max"]).reset_index()

fig, ax = plt.subplots(figsize=(6.5, 4))
colors = ["#c0392b" if m < 0 else "#3d7dcb" for m in by_layer["mean"]]
ax.bar(by_layer["layer"], by_layer["mean"], color=colors, width=0.6)
ax.errorbar(by_layer["layer"], by_layer["mean"],
            yerr=[by_layer["mean"]-by_layer["min"], by_layer["max"]-by_layer["mean"]],
            fmt="none", ecolor="black", capsize=3, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("single invariance-target layer")
ax.set_ylabel("$\\Delta$AUROC at that layer  [mean, min–max over $\\lambda$]")
ax.set_title("Which layer benefits most from the invariance loss")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gain_by_target_layer.pdf")
print("saved both")
