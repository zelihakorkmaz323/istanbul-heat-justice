# Istanbul Heat Justice

Which Istanbul districts get the hottest in summer, who lives there,
and where should the city plant trees first?

## Data sources (all free)
- Landsat 8/9 surface temperature
- Sentinel-2 vegetation index (NDVI)
- ESA WorldCover built-up land
- WorldPop population (2020)
- FAO GAUL district boundaries

## Run it
```bash
pip install -r requirements.txt
python fetch_data.py --project YOUR_GEE_PROJECT_ID --year 2024
streamlit run app.py
```

## How it works
1. `fetch_data.py` uses Google Earth Engine to calculate summer averages per district.
2. `analysis.py` builds a heat risk index, a tree planting priority score, and a simple
   regression with a bootstrap confidence interval.
3. `app.py` shows an interactive map, a ranking, driver analysis and a greening simulator.

## Limits
- District level has only about 39 rows. Neighborhood (mahalle) data gives stronger results.
- The model shows correlation, not proof of cause.
- Surface temperature is not the same as air temperature.

## Next steps
- Use neighborhood boundaries from the Istanbul open data portal (IBB).
- Add age data (elderly and children) to the risk index.
- Compare several summers and heat wave days.
