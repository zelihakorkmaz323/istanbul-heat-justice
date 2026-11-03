"""
Step 1: Download satellite data for Istanbul districts from Google Earth Engine.

For each district we calculate (summer average):
  - lst        : land surface temperature in Celsius (Landsat 8/9)
  - ndvi       : vegetation index, 0 to 1 (Sentinel-2)
  - built_frac : share of built-up land, 0 to 1 (ESA WorldCover)
  - pop        : total population (WorldPop 2020)

Usage:
  pip install -r requirements.txt
  python fetch_data.py --project YOUR_GEE_PROJECT_ID --year 2024

You need a free Earth Engine account: https://earthengine.google.com/
"""
import argparse
import json

import ee


def mask_landsat(img):
    """Remove clouds and convert thermal band to Celsius."""
    qa = img.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 1).eq(0)  # dilated cloud
        .And(qa.bitwiseAnd(1 << 2).eq(0))  # cirrus
        .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
    )
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
    return lst.rename("lst").updateMask(clear)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Earth Engine project id")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--out", default="data/istanbul_districts.geojson")
    args = parser.parse_args()

    try:
        ee.Initialize(project=args.project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=args.project)

    start, end = f"{args.year}-06-01", f"{args.year}-08-31"

    # District boundaries (FAO GAUL). If the name does not match, change it.
    districts = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.eq("ADM1_NAME", "Istanbul")
    )
    region = districts.geometry().bounds()

    # 1) Land surface temperature
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    lst = (
        l8.merge(l9)
        .filterDate(start, end)
        .filterBounds(region)
        .map(mask_landsat)
        .median()
        .rename("lst")
    )

    # 2) Vegetation index
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")

    # 3) Built-up fraction (class 50 = built-up)
    built = (
        ee.Image("ESA/WorldCover/v200/2021")
        .select("Map")
        .eq(50)
        .rename("built_frac")
        .toFloat()
    )

    # 4) Population
    pop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "TUR"))
        .filter(ee.Filter.eq("year", 2020))
        .mosaic()
        .rename("pop")
    )

    mean_stack = lst.addBands(ndvi).addBands(built)

    print("Calculating means (this can take a minute)...")
    with_means = mean_stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=100,
        tileScale=4,
    )

    print("Calculating population...")
    with_pop = pop.reduceRegions(
        collection=with_means,
        reducer=ee.Reducer.sum().setOutputs(["pop"]),
        scale=100,
        tileScale=4,
    )

    result = with_pop.getInfo()

    # Keep only the useful properties
    for feat in result["features"]:
        p = feat["properties"]
        feat["properties"] = {
            "name": p.get("ADM2_NAME"),
            "lst": p.get("lst"),
            "ndvi": p.get("ndvi"),
            "built_frac": p.get("built_frac"),
            "pop": p.get("pop"),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"Saved {len(result['features'])} districts to {args.out}")


if __name__ == "__main__":
    main()
