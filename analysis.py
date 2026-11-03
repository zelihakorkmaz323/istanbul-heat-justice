"""
Step 2: Analysis functions (used by the dashboard).
"""
import geopandas as gpd
import numpy as np
import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def load_data(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).dropna(subset=["lst", "ndvi", "pop"]).copy()

    # Area in km2 (UTM zone 35N fits Istanbul well)
    gdf["area_km2"] = gdf.to_crs(32635).area / 1e6
    gdf["pop_density"] = gdf["pop"] / gdf["area_km2"]

    # Heat risk index: hot + not green + crowded
    gdf["heat_risk"] = (
        zscore(gdf["lst"])
        + zscore(-gdf["ndvi"])
        + zscore(np.log1p(gdf["pop_density"]))
    ) / 3

    # Tree planting priority: risk score weighted by number of people
    gdf["priority"] = zscore(gdf["heat_risk"]) + zscore(np.log1p(gdf["pop"]))
    gdf["priority_rank"] = gdf["priority"].rank(ascending=False).astype(int)
    return gdf


def fit_heat_model(gdf: pd.DataFrame) -> dict:
    """Simple linear model: lst = a + b1*ndvi + b2*built_frac."""
    X = np.column_stack([np.ones(len(gdf)), gdf["ndvi"], gdf["built_frac"]])
    y = gdf["lst"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()

    # Bootstrap confidence interval for the NDVI effect
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(gdf), len(gdf))
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots.append(b[1])
    low, high = np.percentile(boots, [2.5, 97.5])

    return {
        "intercept": coef[0],
        "ndvi_coef": coef[1],
        "built_coef": coef[2],
        "r2": 1 - ss_res / ss_tot,
        "ndvi_ci": (low, high),
    }


def simulate_greening(gdf: pd.DataFrame, model: dict, delta_ndvi: float) -> pd.DataFrame:
    """Estimate temperature drop if NDVI increases by delta_ndvi."""
    out = gdf[["name", "lst", "ndvi", "pop"]].copy()
    out["lst_drop"] = -model["ndvi_coef"] * delta_ndvi
    out["new_lst"] = out["lst"] - out["lst_drop"]
    out["people_cooled"] = out["pop"]
    return out
