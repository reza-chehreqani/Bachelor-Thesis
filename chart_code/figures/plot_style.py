import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

COND_COLORS = {
    "C1_lm_only": "#9aa0a6",
    "C2_raw_invariance": "#f2a541",
    "C3_sae_magnitude": "#3d7dcb",
    "C4_sae_magnitude_support": "#c0392b",
    "baseline": "#555555",
}
COND_LABELS = {
    "C1_lm_only": "C1: LM-only (no invariance)",
    "C2_raw_invariance": "C2: raw-space invariance",
    "C3_sae_magnitude": "C3: SAE-space invariance",
    "C4_sae_magnitude_support": "C4: SAE-space + support term",
    "baseline": "pretrained baseline",
}
OUT_DIR = "../../thesis-final/img"
