"""Compact per-section figures for the 'How the clustering is computed' block on
the cosmetics-catalogs explorer page. One small figure per section (1-7), reusing
the exact data pipeline from cluster_portfolio.py so the illustrations match the
real numbers.

Usage:
    python analysis/method_figures.py

Writes images/analysis/method/method_*.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_portfolio as CP  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

VIOLET, INK, MUTED = "#7c3aed", "#1e1b4b", "#64748b"
CLUSTER_COLORS = CP.CLUSTER_COLORS
OUT = CP.OUT / "method"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "font.family": "sans-serif",
})

FEAT_LABEL = ["log_price", "promo_rate", "med_discount", "log_appear.", "price_cv"]


def title(fig, t):
    fig.text(0.015, 0.99, t, fontsize=13, fontweight="bold", color=INK,
             ha="left", va="top")


# --------------------------------------------------------------- data (same as CP)
df = CP.load()
agg = CP.aggregate(df)
agg["log_price"] = np.log(agg["med_price"])
agg["log_appearances"] = np.log(agg["appearances"])
Xraw = agg[CP.FEATURES].values
scaler = StandardScaler().fit(Xraw)
X = scaler.transform(Xraw)

km = KMeans(n_clusters=4, n_init=25, random_state=42).fit(X)
lab = km.labels_
order = pd.Series(agg["med_price"].values).groupby(lab).median().sort_values().index
remap = {old: new for new, old in enumerate(order)}
lab = np.array([remap[l] for l in lab])
cent = np.vstack([X[lab == c].mean(0) for c in range(4)])

pca = PCA(n_components=2, random_state=42).fit(X)
P = pca.transform(X)
C = pca.transform(cent)

rng = np.random.default_rng(42)
samp = rng.choice(len(X), size=min(6000, len(X)), replace=False)


# --------------------------------------------------------------- 1 · funnel
fig, ax = plt.subplots(figsize=(7.6, 2.3))
fig.subplots_adjust(top=0.72, left=0.30, right=0.97, bottom=0.08)
stages = ["Catalog product rows", "Unique products", "Archetypes (k-means)"]
vals = [int(len(df)), int(len(agg)), 4]
colors = [MUTED, VIOLET, INK]
y = np.arange(len(stages))[::-1]
ax.barh(y, [np.log10(v) for v in vals], color=colors, height=0.6)
for yi, v in zip(y, vals):
    ax.text(np.log10(v) + 0.07, yi, f"{v:,}", va="center", ha="left",
            fontsize=12, fontweight="bold", color=INK)
ax.set_yticks(y)
ax.set_yticklabels(stages, fontsize=10.5)
ax.set_xticks([])
ax.set_xlim(0, np.log10(max(vals)) + 1.2)
ax.spines.left.set_visible(False)
ax.tick_params(length=0)
title(fig, "From catalog rows to archetypes")
fig.savefig(OUT / "method_funnel.png", dpi=130, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------- 2 · feature distributions
fig, axes = plt.subplots(1, 5, figsize=(10.2, 2.2))
fig.subplots_adjust(top=0.70, bottom=0.14, wspace=0.35, left=0.02, right=0.99)
for ax, f, name in zip(axes, CP.FEATURES, FEAT_LABEL):
    ax.hist(agg[f].values, bins=30, color=VIOLET, alpha=0.85)
    ax.set_title(name, fontsize=10, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines.left.set_visible(False)
title(fig, "Distribution of each of the five features")
fig.savefig(OUT / "method_features.png", dpi=130, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------- 3 · standardization
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.0))
fig.subplots_adjust(top=0.80, bottom=0.22, wspace=0.25)
bp1 = a1.boxplot(Xraw, showfliers=False, patch_artist=True, widths=0.6)
bp2 = a2.boxplot(X, showfliers=False, patch_artist=True, widths=0.6)
for bp in (bp1, bp2):
    for box in bp["boxes"]:
        box.set(facecolor=VIOLET, alpha=0.55, edgecolor=INK)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.6)
a1.set_title("Raw features (incomparable scales)", fontsize=11, loc="left", color=MUTED)
a2.set_title("After z-score (mean 0, sd 1)", fontsize=11, loc="left", color=MUTED)
a2.axhline(0, color=MUTED, ls="--", lw=1, alpha=0.6)
for ax in (a1, a2):
    ax.set_xticklabels(FEAT_LABEL, rotation=30, ha="right", fontsize=8.5)
title(fig, "What standardization does")
fig.savefig(OUT / "method_standardize.png", dpi=130, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------- 4 · k-means + centroids
fig, ax = plt.subplots(figsize=(7.6, 4.0))
fig.subplots_adjust(top=0.84)
for c in range(4):
    m = lab[samp] == c
    ax.scatter(P[samp][m, 0], P[samp][m, 1], s=7, alpha=0.3, color=CLUSTER_COLORS[c])
ax.scatter(C[:, 0], C[:, 1], marker="*", s=420, color="white",
           edgecolor=INK, linewidth=1.6, zorder=5)
for c in range(4):
    ax.annotate(f"C{c}", (C[c, 0], C[c, 1]), fontsize=11, fontweight="bold",
                color=INK, ha="center", va="center", zorder=6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
title(fig, "k-means assigns each product to its nearest centroid (★)")
fig.savefig(OUT / "method_kmeans.png", dpi=130, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------- 6 · PCA loadings
fig, ax = plt.subplots(figsize=(7.6, 3.2))
fig.subplots_adjust(top=0.82, bottom=0.20)
x = np.arange(5)
w = 0.38
ax.bar(x - w / 2, pca.components_[0], width=w, color=VIOLET, label="PC1 (34%)")
ax.bar(x + w / 2, pca.components_[1], width=w, color=MUTED, label="PC2 (28%)")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x)
ax.set_xticklabels(FEAT_LABEL, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("loading")
ax.legend(frameon=False, fontsize=9.5)
title(fig, "What the two principal components mean")
fig.savefig(OUT / "method_pca_loadings.png", dpi=130, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------- 7 · appearances spike
fig, ax = plt.subplots(figsize=(7.6, 3.0))
fig.subplots_adjust(top=0.82, bottom=0.16)
app = np.clip(agg["appearances"].values, 1, 21)
ax.hist(app, bins=np.arange(1, 23) - 0.5, color=VIOLET, alpha=0.85)
pct_once = (agg["appearances"].values == 1).mean() * 100
ax.set_xlabel("Catalog appearances (21 = 21+)")
ax.set_ylabel("Unique products")
ax.set_xticks([1, 5, 10, 15, 21])
ax.annotate(f"{pct_once:.0f}% appear exactly once",
            xy=(1, (agg["appearances"].values == 1).sum()),
            xytext=(6, (agg["appearances"].values == 1).sum() * 0.8),
            fontsize=10.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED))
title(fig, "Most products appear only once (single-appearance inflation)")
fig.savefig(OUT / "method_appearances.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("wrote method figures to", OUT)
for p in sorted(OUT.glob("method_*.png")):
    print("  ", p.name)
print(f"pct_once={pct_once:.1f}  n_rows={len(df)}  n_products={len(agg)}")
