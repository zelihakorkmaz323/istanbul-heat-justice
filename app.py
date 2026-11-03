"""
Step 3: Streamlit dashboard.

Run with:  streamlit run app.py
"""
import folium
import streamlit as st
from streamlit_folium import st_folium

import analysis

st.set_page_config(page_title="Istanbul Heat Justice", layout="wide")
st.title("Istanbul Heat Justice")
st.caption("Which districts suffer most from summer heat, and where should we plant trees first?")

DATA_PATH = "data/istanbul_districts.geojson"


@st.cache_data
def get_data():
    return analysis.load_data(DATA_PATH)


try:
    gdf = get_data()
except Exception:
    st.error("Data file not found. Run `python fetch_data.py --project YOUR_PROJECT` first.")
    st.stop()

model = analysis.fit_heat_model(gdf)

METRICS = {
    "Surface temperature (°C)": ("lst", "YlOrRd"),
    "Vegetation index (NDVI)": ("ndvi", "YlGn"),
    "Population density (people/km²)": ("pop_density", "PuRd"),
    "Heat risk index": ("heat_risk", "OrRd"),
    "Tree planting priority": ("priority", "Reds"),
}

tab_map, tab_rank, tab_drivers, tab_sim = st.tabs(
    ["Map", "Ranking", "What drives heat?", "Planting simulator"]
)

# ---------- Map ----------
with tab_map:
    label = st.selectbox("Show on map", list(METRICS.keys()))
    col, palette = METRICS[label]

    m = folium.Map(location=[41.05, 29.0], zoom_start=9, tiles="OpenStreetMap")
    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=["name", col],
        key_on="feature.properties.name",
        fill_color=palette,
        fill_opacity=0.8,
        line_opacity=0.4,
        legend_name=label,
    ).add_to(m)
    folium.GeoJson(
        gdf.to_json(),
        style_function=lambda x: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "lst", "ndvi", "pop"],
            aliases=["District", "Temp (°C)", "NDVI", "Population"],
            localize=True,
        ),
    ).add_to(m)
    st_folium(m, height=550, width=1200)

# ---------- Ranking ----------
with tab_rank:
    st.subheader("Districts that need trees first")
    table = (
        gdf[["priority_rank", "name", "lst", "ndvi", "pop", "pop_density"]]
        .sort_values("priority_rank")
        .rename(
            columns={
                "priority_rank": "Rank",
                "name": "District",
                "lst": "Temp (°C)",
                "ndvi": "NDVI",
                "pop": "Population",
                "pop_density": "People/km²",
            }
        )
    )
    st.dataframe(table.round(2), width="stretch", hide_index=True)

# ---------- Drivers ----------
with tab_drivers:
    st.subheader("Greener districts are cooler")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model R²", f"{model['r2']:.2f}")
    c2.metric("Temp change per +0.1 NDVI", f"{model['ndvi_coef'] * 0.1:.2f} °C")
    low, high = model["ndvi_ci"]
    c3.metric("95% interval (per +0.1 NDVI)", f"{low * 0.1:.2f} to {high * 0.1:.2f} °C")
    st.scatter_chart(gdf, x="ndvi", y="lst", size="pop")
    st.caption(
        "Small sample warning: this model uses districts only, so it shows a pattern, "
        "not proof of cause. Use neighborhood data for stronger results."
    )

# ---------- Simulator ----------
with tab_sim:
    st.subheader("What if we add more green space?")
    delta = st.slider("Increase NDVI by", 0.00, 0.20, 0.05, 0.01)
    sim = analysis.simulate_greening(gdf, model, delta)
    avg_drop = (sim["lst_drop"] * sim["pop"]).sum() / sim["pop"].sum()
    st.metric("Average cooling for a resident", f"{avg_drop:.2f} °C")
    st.dataframe(
        sim[["name", "lst", "new_lst", "lst_drop"]]
        .rename(
            columns={
                "name": "District",
                "lst": "Now (°C)",
                "new_lst": "After (°C)",
                "lst_drop": "Cooling (°C)",
            }
        )
        .round(2),
        width="stretch",
        hide_index=True,
    )
