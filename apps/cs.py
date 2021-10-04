import streamlit as st
import pandas as pd
import numpy as np
import geemap
import ee
import os
import altair as alt
from millify import millify
import geopandas as gpd

from utils.utils import cloudlessNDVI

# import streamlit.components.v1 as components
# from keplergl import KeplerGl

# class Metric:
#     def __init__(self, name, population, area):
#         self.name = name
#         self.population = population
#         self.area = area


def app():
    st.image(r"./assets/header.jpg", use_column_width=True)
    st.title("City of Butuan")
    out_dir = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    lc_stats = os.path.join(out_dir, "lc.csv")

    shp = gpd.read_file(r"./assets/butuan.shp")

    Map = geemap.Map()
    brgy = geemap.shp_to_ee(r"./assets/butuan.shp", encoding="latin-1")

    lon, lat = brgy.geometry().centroid().getInfo()["coordinates"]

    if st.button("Click to load new data",):
        l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA").filterBounds(brgy)
        l8 = l8.map(cloudlessNDVI)
        l8 = l8.select("NDVI")

        geemap.zonal_statistics(l8, brgy, lc_stats, statistics_type="MEAN", scale=1000)

    @st.cache()
    def load_data():
        df = pd.read_csv(lc_stats)
        df = df.drop(
            [
                "PROVINCE_C",
                "BARANGAY_C",
                "BARANGAY_A",
                "MUNCITY",
                "MUNCITY__1",
                "MUNCITY_NA",
                "URBAN_RU_1",
                "REGION",
                "MUNCITY_AL",
                "fid_",
                "MUNCITY_CO",
                "REGION_COD",
                "PROVINCE",
                "MUNCITY_TY",
            ],
            axis=1,
        )

        df = pd.melt(
            df,
            id_vars=["system:index", "BARANGAY", "URBAN_RURA", "Area", "POPULATION"],
        )
        df.variable = df.variable.apply(lambda x: x[12:20])
        df = df.rename(
            columns={
                "value": "NDVI",
                "variable": "Timestamp",
                "system:index": "Default",
                "BARANGAY": "Barangay",
                "URBAN_RURA": "Class",
                "POPULATION": "Population",
            }
        )

        df["Area"] = df["Area"] / 10_000
        # df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
        # df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
        # df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
        # df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
        df["Timestamp"] = pd.DatetimeIndex(df["Timestamp"]).date
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        # df = df.drop(['Default'], axis=1)
        return df

    df = load_data()

    brgy_select = st.multiselect(
        "Select Barangay", df["Barangay"].unique(), df["Barangay"].unique()[:2]
    )

    if len(brgy_select) == 0:
        st.error("Error! At least 1 selection is required.")
        return

    df1 = df[df["Barangay"].isin(brgy_select)]
    ndvi_brgy = df1.groupby(["Barangay", "Default"]).mean()[["NDVI"]]
    ndvi_brgy = ndvi_brgy.reset_index()

    mean_ndvi = df.loc[df.shape[0] - 1, "NDVI"]
    diff_ndvi = mean_ndvi - df.loc[0, "NDVI"]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        if diff_ndvi > 0:
            delta_mean = f"↑ {diff_ndvi:0.3f}"
        else:
            delta_mean = f"{diff_ndvi:0.3f} ↓"

        st.metric(
            f"Mean NDVI",
            f'{ndvi_brgy["NDVI"].mean():0.3f}',
            f"Selected: {len(brgy_select)}",
            delta_color="off",
        )
        # st.metric(f'Pixel Difference', f'{positive_change:0.2%}', f'Positive Change', delta_color='off')

    # with metric2:
    #     if slope > 0:
    #         trending = 'Up'
    #         delta_slope = f'↑ {slope:0.2e}'
    #     else:
    #         trending = 'Down'
    #         delta_slope = f'{slope:0.2e} ↓'

    # st.metric('Trend', f'{trending}', delta_slope)
    # st.metric(f'Area (in has)', f'{millify(aoi.geometry().area().getInfo()/10000, precision=2)}',
    #         f'{millify(aoi.geometry().area().getInfo()/1000000, precision=2)} KM2', delta_color='off')

    # with metric3:
    #     st.metric(f'Cloud Cover', f'{(hist_df.iloc[:,0] == 0).mean():0.2%}',
    #             f'{startdate_format}', delta_color='off')
    # st.metric(f'Days between', f'{millify((enddate - startdate).days, precision=2)}',
    # f'{millify((enddate - startdate).days/365.25, precision=2)} years', delta_color='off')

    with metric4:
        # st.metric(f'Cloud Cover', f'{(hist_df.iloc[:,1] == 0).mean():0.2%}',
        #         f'{enddate_format}', delta_color='off')
        num_file = df["Timestamp"].nunique()
        st.metric(
            f"Landsat 8 Images",
            f"{num_file}",
            f"~{0.919*num_file:0.2f} GB",
            delta_color="off",
        )

    # config = {
    #         'version': 'v1',
    #         'config': {
    #             'mapState': {
    #                 'latitude': lat,
    #                 'longitude': lon,
    #                 'zoom': 10
    #             }
    #         }
    #         }

    # kepler_map = KeplerGl()
    # kepler_map.add_data(data=shp[shp['BARANGAY'].isin(brgy_select)], name='boundaries')
    # kepler_map.config = config
    # kepler_map.save_to_html()

    # htmlFile = open(r'./keplergl_map.html', 'r', encoding='utf-8')
    # source_code = htmlFile.read()
    # components.html(source_code, height=500)

    altA = (
        alt.Chart(df1)
        .mark_line(point=True)
        .properties(width=750)
        .encode(
            x=alt.X("Timestamp:T"),
            y=alt.Y("NDVI:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Barangay:N", legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("Barangay:N"),
                alt.Tooltip("Timestamp:T"),
                alt.Tooltip("NDVI:Q"),
            ],
        )
        .interactive()
    )

    st.altair_chart(altA, use_container_width=True)

    altB = (
        alt.Chart(ndvi_brgy)
        .mark_bar(point=True)
        .properties(width=750)
        .encode(
            y=alt.Y("Barangay:N", sort="-x"),
            x=alt.X("NDVI:Q", title="Mean NDVI"),
            color=alt.Color("Barangay:N", legend=None),
            tooltip=[alt.Tooltip("Barangay:N"), alt.Tooltip("NDVI:Q")],
        )
        .interactive()
    )

    st.altair_chart(altB, use_container_width=True)

    df1 = df1.merge(ndvi_brgy, how="left", on="Default")
    df1 = df1.drop(["NDVI_x", "Barangay_y"], axis=1)
    df1.columns = [
        "Default",
        "Barangay",
        "Class",
        "Area",
        "Population",
        "Timestamp",
        "NDVI",
    ]
    df1 = df1[["Barangay", "Class", "Area", "NDVI"]]
    st.table(df1.iloc[: len(brgy_select), :])

    # m1, m2, m3 = st.columns(3)
    # for i in brgy_select:
    #     Metric(m1.metric('Name', f'{i}'),
    #     m2.metric('Population', millify(int(df1[df1.Barangay == i]['Population'].values[0].squeeze()), precision=2)),
    #     m3.metric('Area (in has)', millify(df1[df1['Barangay'] == i]['Area'].values[0].squeeze(), precision=2)))
