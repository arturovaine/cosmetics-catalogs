"""
Portfolio clustering over the scraped Natura / O Boticario catalog data.

Unit of analysis = unique product (brand + normalized name), aggregated across
every catalog appearance: persistence, promo exposure, discount depth, price.
KMeans on standardized features -> portfolio archetypes.

Usage:
    python analysis/cluster_portfolio.py

Writes charts to images/analysis/ and prints a JSON blob of key stats.
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images" / "analysis"

VIOLET = "#7c3aed"
INK = "#1e1b4b"
MUTED = "#64748b"
NATURA = "#15803d"
BOTICARIO = "#b45309"
CLUSTER_COLORS = ["#7c3aed", "#15803d", "#b45309", "#0ea5e9", "#e11d48", "#64748b"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "font.family": "sans-serif",
})


def styled_title(fig, title, subtitle):
    fig.text(0.02, 0.965, title, fontsize=17, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.925, subtitle, fontsize=12, style="italic", color=MUTED, ha="left")


def norm_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("quem disse, berenice?", "quem disse berenice?")
    return s


def load():
    frames = []
    # read the full 5-brand per-catalog CSVs published to the portfolio site
    DATA = Path.home() / "Documents" / "Portfolio" / "arturovaine.github.io" / "cosmetics-catalogs-data"
    LABELS = {"natura": "Natura", "boticario": "O Boticário", "eudora": "Eudora",
              "avon": "Avon", "jequiti": "Jequiti"}
    for f in glob.glob(str(DATA / "*-c*.csv")):
        slug = Path(f).stem  # e.g. natura-c12-2026
        m = re.match(r"(natura|boticario|eudora|avon|jequiti)-c(\d+)-(\d{4})", slug)
        if not m:
            continue
        df = pd.read_csv(f)
        df["brand"] = LABELS[m.group(1)]
        df["cat_cycle"] = int(m.group(2))
        df["cat_year"] = int(m.group(3))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["regular_price"] = pd.to_numeric(df["regular_price"], errors="coerce")
    df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
    df["promotion_active"] = df["promotion_active"].astype(str).str.lower().eq("true")
    df = df.dropna(subset=["regular_price", "name"])
    df = df[(df["regular_price"] >= 3) & (df["regular_price"] <= 2000)]
    df["pname"] = df["name"].map(norm_name)
    return df


def aggregate(df):
    n_catalogs = df.groupby("brand")[["cat_year", "cat_cycle"]].nunique()
    g = df.groupby(["brand", "pname"])
    agg = pd.DataFrame({
        "appearances": g.size(),
        "promo_rate": g["promotion_active"].mean(),
        "med_price": g["regular_price"].median(),
        "price_cv": g["regular_price"].std().fillna(0) / g["regular_price"].median(),
        "med_discount": g.apply(
            lambda x: x.loc[x["promotion_active"], "discount_percentage"].median(),
            include_groups=False),
        "category": g["normalized_category"].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else "Outros"),
        "line": g["product_line"].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else ""),
    }).reset_index()
    agg["med_discount"] = agg["med_discount"].fillna(0)
    agg["price_cv"] = agg["price_cv"].clip(0, 1)
    return agg


FEATURES = ["log_price", "promo_rate", "med_discount", "log_appearances", "price_cv"]


def run(agg, stats):
    agg["log_price"] = np.log(agg["med_price"])
    agg["log_appearances"] = np.log(agg["appearances"])
    X = StandardScaler().fit_transform(agg[FEATURES].values)

    ks = range(3, 8)
    sils, inertias = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sils.append(silhouette_score(X, km.labels_, sample_size=4000, random_state=42))
        inertias.append(km.inertia_)
    best_k = list(ks)[int(np.argmax(sils))]
    stats["k_scores"] = {str(k): round(s, 4) for k, s in zip(ks, sils)}
    stats["chosen_k"] = best_k

    km = KMeans(n_clusters=best_k, n_init=25, random_state=42).fit(X)
    agg["cluster"] = km.labels_
    order = agg.groupby("cluster")["med_price"].median().sort_values().index
    agg["cluster"] = agg["cluster"].map({old: new for new, old in enumerate(order)})

    # --- k selection ---
    fig, ax1 = plt.subplots(figsize=(9.5, 4.4))
    fig.subplots_adjust(top=0.80)
    ax1.plot(list(ks), sils, "o-", color=VIOLET, lw=2.2, ms=7)
    ax1.set_xticks(list(ks))
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Silhouette score", color=VIOLET)
    ax1.tick_params(axis="y", labelcolor=VIOLET)
    ax1.axvline(best_k, color=VIOLET, ls="--", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(list(ks), inertias, "s--", color=MUTED, lw=1.6, ms=6, alpha=0.8)
    ax2.set_ylabel("Inertia", color=MUTED)
    ax2.tick_params(axis="y", labelcolor=MUTED)
    ax2.spines.right.set_visible(True)
    styled_title(fig, f"Choosing k: silhouette peaks at k = {best_k}",
                 "Silhouette (violet, higher is better) and inertia elbow (gray) over k = 3–7.")
    fig.savefig(OUT / "k_selection.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- PCA map ---
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(X)
    stats["pca_var"] = [round(float(v), 3) for v in pca.explained_variance_ratio_]
    stats["pca_loadings"] = {f"PC{i+1}": {f: round(float(c), 2) for f, c in
                             zip(FEATURES, comp)} for i, comp in enumerate(pca.components_)}
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.subplots_adjust(top=0.84)
    for c in sorted(agg["cluster"].unique()):
        m = agg["cluster"].values == c
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.35, color=CLUSTER_COLORS[c])
        ax.annotate(f"C{c}", P[m].mean(axis=0), fontsize=15, fontweight="bold", color=INK,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white",
                              ec=CLUSTER_COLORS[c], lw=2))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    styled_title(fig, "The portfolio in two dimensions",
                 "PCA projection of the 5 standardized features; colors are KMeans clusters.")
    fig.savefig(OUT / "pca_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- profiles ---
    prof = []
    for c in sorted(agg["cluster"].unique()):
        g = agg[agg["cluster"] == c]
        prof.append({
            "cluster": int(c),
            "n": int(len(g)),
            "share": round(len(g) / len(agg) * 100, 1),
            "med_price": round(float(g["med_price"].median()), 1),
            "promo_rate": round(float(g["promo_rate"].mean() * 100), 1),
            "med_discount": round(float(g["med_discount"].median()), 1),
            "med_appearances": float(g["appearances"].median()),
            "price_cv": round(float(g["price_cv"].median()), 3),
            "pct_natura": round(float((g["brand"] == "Natura").mean() * 100), 1),
            "brand_mix": {b: round(float((g["brand"] == b).mean() * 100), 1)
                          for b in ["O Boticário", "Natura", "Eudora", "Avon", "Jequiti"]},
            "top_categories": g["category"].value_counts().head(3).index.tolist(),
            "top_lines": g["line"].value_counts().head(3).index.tolist(),
            "examples": g.nlargest(3, "appearances")["pname"].str.title().tolist(),
        })
    stats["profiles"] = prof

    labels = [f"C{p['cluster']}" for p in prof]
    colors = [CLUSTER_COLORS[p["cluster"]] for p in prof]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    fig.subplots_adjust(top=0.86, hspace=0.5, wspace=0.3)
    panels = [
        ("med_price", "Median price (R$)"),
        ("promo_rate", "% of appearances on promotion"),
        ("med_discount", "Median discount when on promo (%)"),
        ("med_appearances", "Median catalog appearances"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        vals = [p[key] for p in prof]
        ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title, fontsize=12.5, loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=10.5)
        ax.set_yticks([])
        ax.spines.left.set_visible(False)
    styled_title(fig, "Cluster profiles at a glance",
                 "Each bar is one archetype of the combined five-brand portfolio.")
    fig.savefig(OUT / "cluster_profiles.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- portfolio architecture map: persistence vs promo exposure ---
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.subplots_adjust(top=0.84)
    rng = np.random.default_rng(42)
    jx = rng.uniform(-0.18, 0.18, len(agg))
    jy = rng.uniform(-0.035, 0.035, len(agg))
    for c in sorted(agg["cluster"].unique()):
        m = (agg["cluster"] == c).values
        ax.scatter(agg.loc[m, "appearances"] + jx[m] * agg.loc[m, "appearances"],
                   agg.loc[m, "promo_rate"] + jy[m], s=9, alpha=0.3,
                   color=CLUSTER_COLORS[c], label=f"C{c}")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 30])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("Catalog appearances (log scale)")
    ax.set_ylabel("Share of appearances on promotion")
    ax.legend(frameon=False, loc="center right", markerscale=2)
    styled_title(fig, "The portfolio architecture map",
                 "Each dot is one unique product; position = persistence × promo exposure "
                 "(jittered for visibility).")
    fig.savefig(OUT / "architecture_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- category composition ---
    cats = df_top_categories = agg["category"].value_counts().head(7).index.tolist()
    comp = np.zeros((len(prof), len(cats) + 1))
    for i, p in enumerate(prof):
        g = agg[agg["cluster"] == p["cluster"]]
        vc = g["category"].value_counts(normalize=True)
        for j, cat in enumerate(cats):
            comp[i, j] = vc.get(cat, 0) * 100
        comp[i, -1] = max(0, 100 - comp[i, :-1].sum())
    cat_colors = ["#7c3aed", "#15803d", "#b45309", "#0ea5e9", "#e11d48",
                  "#eab308", "#14b8a6", "#94a3b8"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.subplots_adjust(top=0.82)
    left = np.zeros(len(prof))
    for j, cat in enumerate(cats + ["Other"]):
        ax.barh(labels, comp[:, j], left=left, color=cat_colors[j], label=cat, height=0.6)
        left += comp[:, j]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of cluster")
    ax.invert_yaxis()
    ax.legend(frameon=False, ncols=4, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    styled_title(fig, "What each archetype is made of",
                 "Category mix per cluster (top 7 categories; remainder grouped as Other).")
    fig.savefig(OUT / "category_mix.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- brand mix (all five brands) ---
    BRANDS = ["O Boticário", "Natura", "Eudora", "Avon", "Jequiti"]
    BRAND_HEX = {"O Boticário": "#b45309", "Natura": "#15803d", "Eudora": "#be185d",
                 "Avon": "#6d28d9", "Jequiti": "#0e7490"}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.subplots_adjust(top=0.80)
    left = np.zeros(len(prof))
    for b in BRANDS:
        vals = np.array([p["brand_mix"][b] for p in prof])
        ax.barh(labels, vals, left=left, color=BRAND_HEX[b], label=b, height=0.6)
        for i, v in enumerate(vals):
            if v >= 7:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=9.5)
        left = left + vals
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines.bottom.set_visible(False)
    ax.legend(frameon=False, ncols=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), fontsize=9.5)
    styled_title(fig, "Every archetype draws from all five brands",
                 "Brand composition per cluster; the portfolio architecture is brand-agnostic.")
    fig.savefig(OUT / "brand_mix.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    df = load()
    stats["rows"] = int(len(df))
    stats["catalogs"] = int(df.groupby(["brand", "cat_year", "cat_cycle"]).ngroups)
    agg = aggregate(df)
    stats["unique_products"] = int(len(agg))
    stats["med_appearances_overall"] = float(agg["appearances"].median())
    agg = run(agg, stats)
    agg.to_csv(ROOT / "data" / "portfolio_clusters.csv", index=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
