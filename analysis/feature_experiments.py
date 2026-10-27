"""Does the clustering change if we define features differently?

Re-runs k-means on several feature sets and compares each to the production
baseline (the 5 behavioural features) via:
  - silhouette at k=4 (quality)
  - Adjusted Rand Index vs baseline labels (how much assignments change; 1 = same)
  - best k by silhouette (does the structure prefer a different number of groups)
  - for +category / +brand: how category/brand-dominated the new clusters get.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_portfolio as CP  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

df = CP.load()
agg = CP.aggregate(df)
agg["log_price"] = np.log(agg["med_price"])
agg["log_appearances"] = np.log(agg["appearances"])
base_X = agg[CP.FEATURES].values


def fit(Xmat, k):
    Xs = StandardScaler().fit_transform(Xmat)
    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(Xs)
    return km.labels_, Xs


def sweep(Xmat):
    Xs = StandardScaler().fit_transform(Xmat)
    out = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        out[k] = round(silhouette_score(Xs, km.labels_, sample_size=4000, random_state=42), 3)
    return out


base_labels, _ = fit(base_X, 4)

topcats = agg["category"].value_counts().head(8).index.tolist()
cat_oh = np.column_stack([(agg["category"] == c).astype(float).values for c in topcats])
brands = ["O Boticário", "Natura", "Eudora", "Avon", "Jequiti"]
brand_oh = np.column_stack([(agg["brand"] == b).astype(float).values for b in brands])
is_kit = agg["pname"].str.contains(r"kit|presente|combo", case=False, regex=True).astype(float).values.reshape(-1, 1)

experiments = {
    "E0 baseline (5 behavioural)": base_X,
    "E1 + category (top-8 one-hot)": np.column_stack([base_X, cat_oh]),
    "E2 + brand (5 one-hot)": np.column_stack([base_X, brand_oh]),
    "E3 + is_kit/gift flag": np.column_stack([base_X, is_kit]),
    "E4 core: promo+discount+appearances (3)": agg[["promo_rate", "med_discount", "log_appearances"]].values,
    "E5 price+persistence only (no promo/disc)": agg[["log_price", "log_appearances", "price_cv"]].values,
}

print(f"{'experiment':<42} {'k=4 sil':>8} {'ARI vs base':>12} {'best k':>7}  sizes(k=4)")
print("-" * 100)
for name, Xmat in experiments.items():
    labels, Xs = fit(Xmat, 4)
    sil = silhouette_score(Xs, labels, sample_size=4000, random_state=42)
    ari = adjusted_rand_score(base_labels, labels)
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    sc = sweep(Xmat)
    bk = max(sc, key=sc.get)
    print(f"{name:<42} {sil:>8.3f} {ari:>12.3f} {bk:>7}  {sizes}")

# how category/brand-aligned do the +category / +brand clusterings get?
def purity(labels, series):
    d = pd.DataFrame({"c": labels, "v": series.values})
    shares = d.groupby("c")["v"].apply(lambda s: s.value_counts(normalize=True).iloc[0])
    return [round(float(x), 2) for x in sorted(shares.tolist(), reverse=True)]


lab_cat, _ = fit(np.column_stack([base_X, cat_oh]), 4)
lab_brand, _ = fit(np.column_stack([base_X, brand_oh]), 4)
print("\n+category clusters: top single-category share per cluster:", purity(lab_cat, agg["category"]))
print("+brand    clusters: top single-brand    share per cluster:", purity(lab_brand, agg["brand"]))
print("(baseline for reference) top single-category share per cluster:", purity(base_labels, agg["category"]))
print("(baseline for reference) top single-brand    share per cluster:", purity(base_labels, agg["brand"]))
