"""Build the cumulative data for the animated architecture map.

Each step adds one more publishing period (year, cycle); a product's position is
its appearances-so-far (x, log) vs promo-rate-so-far (y), colored by its FINAL
k-means cluster so colours stay stable across frames. The last frame reproduces
the static architecture_map.png.

Writes a compact JSON (sampled products + per-appearance period/promo lists) that
the page expands into Plotly frames client-side.

Usage:
    python analysis/arch_anim_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_portfolio as CP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEST = (Path.home() / "Documents" / "Portfolio" / "arturovaine.github.io"
        / "cosmetics-catalogs-data" / "arch_anim.json")
SAMPLE = None  # None = every unique product; an int caps the sample

df = CP.load()
df["promo01"] = df["promotion_active"].astype(int)

prof = pd.read_csv(ROOT / "data" / "portfolio_clusters.csv")
clusters = prof[["brand", "pname", "cluster"]]
df = df.merge(clusters, on=["brand", "pname"], how="inner")
prof = prof.drop_duplicates(["brand", "pname"]).set_index(["brand", "pname"])

# ordered publishing periods (year, cycle) -> index
periods = sorted({(int(y), int(c)) for y, c in zip(df["cat_year"], df["cat_cycle"])})
pidx = {p: i for i, p in enumerate(periods)}
df["pi"] = [pidx[(int(y), int(c))] for y, c in zip(df["cat_year"], df["cat_cycle"])]

# catalogs added per period (for the running count shown on each frame)
cats_per_period = (df.groupby("pi")[["brand", "cat_year", "cat_cycle"]]
                   .apply(lambda g: g.drop_duplicates().shape[0]))
cum_cats = cats_per_period.reindex(range(len(periods)), fill_value=0).cumsum().tolist()

# cumulative distinct products that have appeared by each period (full data, not the sample)
first_pi = df.groupby(["brand", "pname"])["pi"].min()
cum_products = (first_pi.value_counts().reindex(range(len(periods)), fill_value=0)
                .sort_index().cumsum().tolist())

rng = np.random.default_rng(42)
keys = list(df.groupby(["brand", "pname", "cluster"]).groups.keys())
take = (range(len(keys)) if SAMPLE is None
        else rng.choice(len(keys), size=min(SAMPLE, len(keys)), replace=False))
g = df.groupby(["brand", "pname", "cluster"])

def clean(v):
    return "" if pd.isna(v) else str(v)


pts = []
for i in take:
    key = keys[i]
    sub = g.get_group(key)
    row = prof.loc[(key[0], key[1])]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    name = clean(key[1]).strip().title()
    if len(name) > 52:
        name = name[:51] + "…"
    pts.append({
        "c": int(key[2]),
        "ap": [int(v) for v in sub["pi"].tolist()],
        "pr": [int(v) for v in sub["promo01"].tolist()],
        "jx": round(float(rng.uniform(-0.18, 0.18)), 3),
        "jy": round(float(rng.uniform(-0.035, 0.035)), 3),
        # compact array: [name, brand, category, line, price, appearances, promo%, discount%]
        "d": [name, clean(key[0]), clean(row["category"]), clean(row["line"]),
              round(float(row["med_price"]), 2), int(row["appearances"]),
              round(float(row["promo_rate"]) * 100), round(float(row["med_discount"]))],
    })

out = {
    "periods": [f"{y}·c{c:02d}" for (y, c) in periods],
    "cumCatalogs": [int(v) for v in cum_cats],
    "cumProducts": [int(v) for v in cum_products],
    "colors": CP.CLUSTER_COLORS[:4],
    "names": ["C0 · managed", "C1 · one-cycle", "C2 · backbone", "C3 · promo-born"],
    "pts": pts,
}
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {DEST}  ({DEST.stat().st_size/1024:.0f} KB)")
print(f"periods={len(periods)}  sampled_products={len(pts)}  total_products={len(keys)}")
print(f"max appearances (all)={int(prof['appearances'].max())}  "
      f"max plotted={max(p['d'][5] for p in pts)}")
print(f"first period={out['periods'][0]}  last={out['periods'][-1]}  "
      f"cum_catalogs_last={cum_cats[-1]}  cum_products_last={cum_products[-1]}")
