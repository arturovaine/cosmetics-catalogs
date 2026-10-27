"""Lloyd-iteration animation data for the 'k-means partitioning' section.

Runs k-means (k=4) step by step on the real 5-feature standardized space,
recording the cluster assignment and centroids at each iteration, projected to
2-D PCA for display. Cluster colours follow the production order (C0..C3 by
ascending median price), so the final iteration matches the static map.

Point positions are fixed across iterations (only colours change); centroids
move. A compact JSON is expanded into frames client-side.

Usage:
    python analysis/kmeans_anim_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib  # noqa: F401  (cluster_portfolio sets Agg; keep import order sane)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_portfolio as CP  # noqa: E402
from sklearn.cluster import kmeans_plusplus  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEST = (Path.home() / "Documents" / "Portfolio" / "arturovaine.github.io"
        / "cosmetics-catalogs-data" / "kmeans_anim.json")
K = 4
MAX_ITERS = 16
DISPLAY = 4000

df = CP.load()
agg = CP.aggregate(df)
agg["log_price"] = np.log(agg["med_price"])
agg["log_appearances"] = np.log(agg["appearances"])
X = StandardScaler().fit_transform(agg[CP.FEATURES].values)
pca = PCA(n_components=2, random_state=42).fit(X)
P = pca.transform(X)

centers, _ = kmeans_plusplus(X, n_clusters=K, random_state=42)
iters = []
prev = None
for _ in range(MAX_ITERS):
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    assign = d.argmin(1)
    iters.append((assign.copy(), centers.copy()))
    if prev is not None and np.array_equal(assign, prev):
        break
    prev = assign
    centers = np.array([X[assign == c].mean(0) if (assign == c).any() else centers[c]
                        for c in range(K)])

# colour order: centroid index -> C-order (ascending final median price).
final = iters[-1][0]
med = [np.median(agg["med_price"].values[final == c]) if (final == c).any() else 1e9
       for c in range(K)]
order = np.argsort(med)
remap = {old: new for new, old in enumerate(order)}

rng = np.random.default_rng(42)
sel = rng.choice(len(P), size=min(DISPLAY, len(P)), replace=False)
Pd = P[sel]

frames = []
for assign, centers in iters:
    cen2d = pca.transform(centers)
    frames.append({
        "col": [remap[int(v)] for v in assign[sel]],
        "cx": [round(float(v), 3) for v in cen2d[:, 0]],
        "cy": [round(float(v), 3) for v in cen2d[:, 1]],
        "cc": [remap[c] for c in range(K)],
    })

out = {
    "px": [round(float(v), 3) for v in Pd[:, 0]],
    "py": [round(float(v), 3) for v in Pd[:, 1]],
    "colors": CP.CLUSTER_COLORS[:4],
    "pcVar": [round(float(v), 3) for v in pca.explained_variance_ratio_],
    "frames": frames,
}
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_text(json.dumps(out, separators=(",", ":")))

sizes = [int((final == order[c]).sum()) for c in range(K)]
print(f"wrote {DEST}  ({DEST.stat().st_size/1024:.0f} KB)")
print(f"iterations={len(frames)}  display_points={len(Pd)}")
print(f"final cluster sizes (C0..C3) = {sizes}")
print(f"final median prices (C0..C3) = {[round(med[order[c]],1) for c in range(K)]}")
