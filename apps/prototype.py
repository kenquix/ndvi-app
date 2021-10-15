import streamlit as st
import pandas as pd
import numpy as np
import os
import ee
import json
import base64
import geemap
import imageio
from millify import millify
import geopandas as gpd
import shapely.wkt
import shapely.geometry
import ast

from PIL import Image
from PIL import ImageDraw

import altair as alt

import streamlit.components.v1 as components
from streamlit_folium import folium_static

from statsmodels.nonparametric.smoothers_lowess import lowess
from altair_saver import save
import folium
from folium import plugins
import seaborn as sns

from fpdf import FPDF

from utils.utils import *

def app():
    class PDF(FPDF):
        def header(self):
            # Logo
            self.image(r'./assets/header.jpg', 10, 8, 33)
            # Arial bold 15
            self.set_font('Arial', 'B', 15)
            # Move to the right
            self.cell(80)
            # Title
            self.cell(30, 10, 'Vegetation Assessment and Monitoring Report', 0, 0, 'C')
            # Line break
            self.ln(20)

        # Page footer
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    Map = geemap.Map()
    st.header("Vegetation Assessment and Monitoring App")

    _, ds1_col, ds2_col, ds3_col, _ = st.columns((0.15, 3, 3, 3, 0.15))

    with ds1_col:
        datasource = st.selectbox(
            label="Select AOI:",
            options=["User-Defined", "Butuan City"],
            index=1,
            help="AOI - Area of Interest",
        )
    with st.form(key="form1", clear_on_submit=False):
        if datasource == "User-Defined":
            with ds2_col:
                user_input_method = st.radio(
                    label="Select input method",
                    options=["Upload KML file", "Draw bounding box"],
                    help="Options to provide user-defined AOI",
                )

            default_region = "[[[125.4727148947,8.9012996164],\
                                [125.5990576681,8.9012996164],\
                                [125.5990576681,8.9828386907],\
                                [125.4727148947,8.9828386907],\
                                [125.4727148947,8.9012996164]]]"

            if user_input_method == "Upload KML file":
                inputFile = st.file_uploader(
                    "Upload a file",
                    type=["kml"],
                    help="Currently, only KML format AOI is accepted.",
                )
                try:
                    if inputFile is None:
                        data = json.loads(default_region)

                    else:
                        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
                        input_df = gpd.read_file(inputFile, driver="KML")
                        s = str(input_df.geometry.iloc[0])
                        g1 = shapely.wkt.loads(s)
                        g2 = shapely.geometry.mapping(g1)
                        g3 = json.dumps(g2)
                        data = ast.literal_eval(g3)["coordinates"]
                        data = [[i[:2] for i in data[0]]]

                    lon_list = []
                    lat_list = []
                    for i in data[0]:
                        lon, lat = i
                        lon_list.append(lon)
                        lat_list.append(lat)

                    ne = [[np.min(lat_list), np.min(lon_list)]]
                    sw = [[np.max(lat_list), np.max(lon_list)]]

                except:
                    st.error(
                        f"Error: Expected a different input. Make sure you selected the GEOJSON format."
                    )
                    submit_button = st.form_submit_button(label="Run selection")
                    return
            else:
                st.markdown(
                    '<p style="font-size:13px">Draw a bounding box</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <ul>
                    <li>Click OK to accept cookies.</li>
                    <li>Select user-defined AOI by utilizing the toolbars located at
                    the upper left corner of the map or using the search box.</li>
                    <li>At the lower left corner, select the <strong>GEOJSON</strong> format and copy
                    the coordinates.</li><br>
                    """,
                    unsafe_allow_html=True,
                )
                components.iframe("https://boundingbox.klokantech.com/", height=500)

                inputRegion = st.text_input(
                    f"Paste AOI coordinates here:",
                    help="Currently, only GeoJSON formatted AOI is accepted",
                )

                try:
                    if len(inputRegion) == 0:
                        data = json.loads(default_region)

                    else:
                        if inputRegion[:3] == "[[[":
                            data = json.loads(inputRegion)
                        else:
                            inputRegion = "[[" + inputRegion + "]]"
                            data = json.loads(inputRegion)

                    lon_list = []
                    lat_list = []
                    for i in data[0]:
                        lon, lat = i
                        lon_list.append(lon)
                        lat_list.append(lat)

                    ne = [[np.min(lat_list), np.min(lon_list)]]
                    sw = [[np.max(lat_list), np.max(lon_list)]]

                except:
                    st.error(
                        f"Error: Expected a different input. Make sure you selected the GEOJSON format."
                    )
                    submit_button = st.form_submit_button(label="Run selection")
                    return

        elif datasource == "Butuan City":
            control0, _, control1, _, control2 = st.columns((1.5, 0.25, 3, 0.2, 1))
            butuan_boundary_gdf = gpd.read_file("./assets/butuan.shp")

            gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
            gpd.io.file.fiona.drvsupport.supported_drivers["kml"] = "rw"

            with ds2_col:
                boundary = st.selectbox(
                    label="Select Boundary",
                    options=["Administrative (Brgy Level)", "Land Cover", "Production and Protection"],
                    help="Define the boundary to be used",
                )

            if boundary == "Administrative (Brgy Level)":
                with control0:
                    selected_brgy = st.selectbox(
                        label="Select Barangay",
                        options=["Select All"]
                        + list(butuan_boundary_gdf["NAME_3"].unique()),
                        help="List of Barangays of Butuan",
                    )
                # with b2:
                #     selected_lcov = st.selectbox(
                #         label="Select Land Cover Data",
                #         options=[],
                #         help="This option is not available with the current selection",
                #     )
                if selected_brgy != "Select All":
                    selected_boundary_gdf = butuan_boundary_gdf[
                        butuan_boundary_gdf["NAME_3"] == selected_brgy
                    ]
                    selected_boundary_gdf.to_file(
                        "./assets/selection_brgy.kml", driver="KML"
                    )
                    input_df = gpd.read_file(
                        "./assets/selection_brgy.kml", driver="KML"
                    )
                else:
                    input_df = gpd.read_file("./assets/butuan_city.kml", driver="KML")

            elif boundary == "Land Cover":
                land_cover_data_year = 2015
                butuan_lcov_gdf = gpd.read_file(
                    f"./assets/butuan_landcover_{land_cover_data_year}.shp"
                )
                # with b1:
                #     selected_brgy = st.selectbox(
                #         label="Select Barangay",
                #         options=[],
                #         help="This option is not available with the current selection",
                #     )
                with control0:
                    selected_lcov = st.selectbox(
                        label="Select Land Cover",
                        options=["Select All"]
                        + list(butuan_lcov_gdf["CLASS"].unique()),
                        help="Land Cover data as of 2015",
                    )
                if selected_lcov != "Select All":
                    selected_lcov_gdf = butuan_lcov_gdf[
                        butuan_lcov_gdf["CLASS"] == selected_lcov
                    ]
                    selected_lcov_gdf.to_file(
                        "./assets/selection_lcov.kml", driver="KML"
                    )
                    input_df = gpd.read_file(
                        "./assets/selection_lcov.kml", driver="KML"
                    )
                else:
                    input_df = gpd.read_file("./assets/butuan_city.kml", driver="KML")

            elif boundary == 'Production and Protection':
                production_protection = gpd.read_file('./assets/production_protection.shp')
                with control0:
                    selected_prod = st.selectbox('Select Region', options=['Select All'] + list(production_protection['Legend'].unique()), help='Production and Protection Region')
                if selected_prod != 'Select All':
                    selected_prod_gdf = production_protection[production_protection['Legend'] == selected_prod]
                    selected_prod_gdf.to_file('./assets/selection_prod.kml', driver='KML')
                    input_df = gpd.read_file('./assets/selection_prod.kml')
                else:
                    input_df = gpd.read_file("./assets/butuan_city.kml", driver="KML")

            bounds_to_fit = input_df.bounds
            sw = bounds_to_fit[["miny", "minx"]].values.tolist()
            ne = bounds_to_fit[["maxy", "maxx"]].values.tolist()
            s = str(input_df.geometry.iloc[0])
            g1 = shapely.wkt.loads(s)
            g2 = shapely.geometry.mapping(g1)
            g3 = json.dumps(g2)
            data = ast.literal_eval(g3)["coordinates"]
            if len(data) == 1:
                data = [[i[:2] for i in data[0]]]

        aoi = ee.FeatureCollection(ee.Geometry.MultiPolygon(data))
        datamask = (
            ee.Image("UMD/hansen/global_forest_change_2015").select("datamask").eq(1)
        )
        lon, lat = aoi.geometry().centroid().getInfo()["coordinates"]
        l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA").filterBounds(aoi)
        l8_ndvi_cloudless = l8.map(cloudlessNDVI)

        with st.spinner(text="Fetching data from GEE server..."):
            date_list = date_range(l8_ndvi_cloudless, aoi)

        with control1:
            try:
                startdate, enddate = st.select_slider(
                    "Date Slider",
                    date_list.Timestamp.unique().tolist(),
                    value=[
                        date_list.Timestamp.unique().tolist()[0],
                        date_list.Timestamp.unique().tolist()[-1],
                    ],
                    help="Use the slider to select the DOI's (start and end date)",
                )

            except IndexError as e:
                st.error(
                    "Something went wrong! There seems to be no available images for the current selection. Please modify your selections or refresh the page."
                )
                submit_button = st.form_submit_button(label="Run selection")
                return

        with control2:
            scale = st.number_input(
                label="Scale",
                min_value=30,
                max_value=500,
                value=100,
                step=10,
                help="Use to rescale images. Default is 100m.",
            )

        submit_button = st.form_submit_button(label="Run selection")

    startdate_format = startdate.strftime("%B %d, %Y")
    enddate_format = enddate.strftime("%B %d, %Y")

    with st.spinner(text="Fetching data from GEE server..."):
        (
            df,
            report_df,
            start_img,
            end_img,
            diff_img,
            diff_bin_norm,
            hist_df,
        ) = read_data(l8_ndvi_cloudless, startdate, enddate, aoi, datamask, scale)

        df["Timestamp"] = pd.to_datetime(df.Timestamp)
        df = df.round(4)
        df_annual = transform(df)

    visParams = {
        "min": 0,
        "max": 1,
        "palette": [
            "FFFFFF",
            "CE7E45",
            "DF923D",
            "F1B555",
            "FCD163",
            "99B718",
            "74A901",
            "66A000",
            "529400",
            "3E8601",
            "207401",
            "056201",
            "004C00",
            "023B01",
            "012E01",
            "011D01",
            "011301",
        ],
        "opacity": 0.8,
    }

    visParams_diff = {
        "min": -1,
        "max": 1,
        "palette": ["#FF4C56", "#FFFFFF", "#73DA6E"],
        "opacity": 0.7,
    }

    # Create a folium map object.
    my_map = folium.Map(location=[lat, lon])

    # Add custom basemaps
    basemaps["Google Satellite Hybrid"].add_to(my_map)
    # basemaps["Google Terrain"].add_to(my_map)
    # basemaps["Google Maps"].add_to(my_map)

    my_map.add_ee_layer(
        diff_img.clip(aoi.geometry()), visParams_diff, "Difference Image"
    )
    my_map.add_ee_layer(
        start_img.clip(aoi.geometry()),
        visParams,
        f'{startdate.strftime("%d %B %Y")} Image',
    )
    my_map.add_ee_layer(
        end_img.clip(aoi.geometry()),
        visParams,
        f'{enddate.strftime("%d %B %Y")} Image',
    )

    # # Add a layer control panel to the map.
    my_map.add_child(folium.LayerControl())
    my_map.add_child(folium.LatLngPopup())

    # # Add fullscreen button
    plugins.Fullscreen().add_to(my_map)
    plugins.MiniMap().add_to(my_map)

    # if datasource == 'Butuan City':
    my_map.fit_bounds([sw, ne])

    start_lower = hist_df[f"{startdate}"].quantile(0.25).min()
    start_upper = hist_df[f"{startdate}"].quantile(0.75).max()
    end_lower = hist_df[f"{enddate}"].quantile(0.25).min()
    end_upper = hist_df[f"{enddate}"].quantile(0.75).max()

    x = np.array(pd.to_datetime(df.Timestamp), dtype=float)
    y = df.NDVI_Lowess
    results = np.polyfit(x, y, deg=1)
    slope, _ = results
    _, positive_change = diff_bin_norm

    mean_ndvi = df.loc[df.shape[0] - 1, "NDVI"]
    diff_ndvi = mean_ndvi - df.loc[0, "NDVI"]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        if diff_ndvi > 0:
            delta_mean = f"↑ {diff_ndvi:0.3f}"
        else:
            delta_mean = f"{diff_ndvi:0.3f} ↓"

        st.metric(f"Mean NDVI", f"{mean_ndvi:0.3f}", delta_mean)
        st.metric(
            f"Pixel Difference",
            f"{positive_change:0.2%}",
            f"Positive Change",
            delta_color="off",
        )

    with metric2:
        if slope > 0:
            trending = "Up"
            delta_slope = f"↑ {slope:0.2e}"
        else:
            trending = "Down"
            delta_slope = f"{slope:0.2e} ↓"

        st.metric("Trend", f"{trending}", delta_slope)
        st.metric(
            f"Area (in has)",
            f"{millify(aoi.geometry().area().getInfo()/10000, precision=2)}",
            f"{millify(aoi.geometry().area().getInfo()/1000000, precision=2)} KM2",
            delta_color="off",
        )

    with metric3:
        st.metric(
            f"Cloud Cover",
            f"{(hist_df.iloc[:,0] == 0).mean():0.2%}",
            f"{startdate_format}",
            delta_color="off",
        )
        st.metric(
            f"Days between",
            f"{millify((enddate - startdate).days, precision=2)}",
            f"{millify((enddate - startdate).days/365.25, precision=2)} years",
            delta_color="off",
        )

    with metric4:
        st.metric(
            f"Cloud Cover",
            f"{(hist_df.iloc[:,1] == 0).mean():0.2%}",
            f"{enddate_format}",
            delta_color="off",
        )
        st.metric(
            f"Landsat 8 Images",
            f"{df.shape[0]}",
            f"~{0.919*df.shape[0]:0.2f} GB",
            delta_color="off",
        )

    folium_static(my_map)

    st.image(r"./assets/scale.png", use_column_width=True)

    with st.spinner(
        text="Fetching data from GEE server... Generating timelapse animation..."
    ):
        with st.expander("Timelapse of Annual NDVI Composite Images"):
            timelapse = st.checkbox("Check to generate the animation.")

            if timelapse:
                out_dir = os.path.join(os.path.expanduser("~"), "assets")

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                # out_gif = os.path.join(out_dir, 'landsat_ndvi_ts.gif')

                # add bands for DOY and Year
                def add_doy(img):
                    doy = ee.Date(img.get("system:time_start")).getRelative(
                        "day", "year"
                    )
                    return img.set("doy", doy)

                def add_year(img):
                    year = ee.Date(img.get("system:time_start")).get("year")
                    return img.set("year", year)

                l8_ndvi_cloudless = l8_ndvi_cloudless.map(add_doy)
                l8_ndvi_cloudless = l8_ndvi_cloudless.map(add_year)

                filenames = []
                images = []
                region = aoi.geometry().bounds()
                l8_ndvi_cloudless = l8_ndvi_cloudless.select("NDVI")

                zip_dir = os.path.join(os.path.expanduser("~"), "assets")

                for i in range(startdate.year, enddate.year + 1):
                    timelapse_dir = os.path.join(zip_dir, f"landsat_{i}.png")
                    fcol = (
                        l8_ndvi_cloudless.filterMetadata("year", "equals", i)
                        .reduce(ee.Reducer.median())
                        .clip(aoi)
                        .updateMask(datamask)
                    )
                    geemap.get_image_thumbnail(
                        fcol,
                        timelapse_dir,
                        visParams,
                        region=region,
                        dimensions=500,
                        format="png",
                    )
                    img = Image.open(timelapse_dir)
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0), f"Year {i}")
                    img.save(timelapse_dir)
                    filenames.append(timelapse_dir)

                for filename in filenames:
                    images.append(imageio.imread(filename))

                imageio.mimsave(
                    os.path.join(zip_dir, "landsat_ndvi_ts.gif"), images, fps=1
                )

                file_ = open(os.path.join(zip_dir, "landsat_ndvi_ts.gif"), "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()

                st.markdown(
                    f"""<center><img src="data:image/gif;base64,{data_url}" alt="timelapse gif"></center>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

    decrease_list = list(report_df[report_df.Interpretation.isin(["Decrease"])].index)
    increase_list = list(report_df[report_df.Interpretation.isin(["Increase"])].index)

    st.markdown("---")
    st.subheader("Data Analytics and Visualization")
    st.markdown(
        "Table 1. Area (in hectares) and Percent Change between Selected Dates across NDVI classes"
    )

    st.dataframe(report_df)
    st.markdown(
        f"""
        <p align="justify">A decrease in the first four (4) NDVI classes (i.e., -1 - 0.6) or an increase in the remaining NDVI classes
        (i.e., 0.6 above) is interpreted as positive and vice versa. </p>
        
        <p align="justify">Given this and inspecting the table above, we observed <font color="#2D8532"><strong>increase</strong></font> in the following NDVI landcover classes: 
        <font color="#2D8532"><strong>{', '.join(increase_list)}</strong></font> while we observed a <font color="#A42F25"><strong>decrease</strong></font> in the following classes: 
        <font color="#A42F25"><strong>{', '.join(decrease_list)}</strong></font>.</p>
        """,
        unsafe_allow_html=True,
    )

    if st.checkbox("Cumulative"):
        cumulative = True
        tooltip_title = "Proportion"
    else:
        cumulative = False
        tooltip_title = "Density"

    temp_df = hist_df[(hist_df.iloc[:, 0] != 0) & (hist_df.iloc[:, 1] != 0)].copy()
    hist_df = pd.melt(temp_df)

    s = sns.kdeplot(data=hist_df, x="value", hue="variable", cut=1)
    # s = sns.ecdfplot(data=hist_df, x="value", hue="variable")
    before = s.get_lines()[0].get_data()
    after = s.get_lines()[1].get_data()
    before_df = pd.DataFrame(
        {"NDVI": before[0], "Density": before[1], "Date": f"{startdate}"}
    )
    after_df = pd.DataFrame(
        {"NDVI": after[0], "Density": after[1], "Date": f"{enddate}"}
    )

    if cumulative:
        before_df["Density"] = (
            before_df["Density"].cumsum().div(before_df["Density"].cumsum().max())
        )
        after_df["Density"] = (
            after_df["Density"].cumsum().div(after_df["Density"].cumsum().max())
        )
    data = before_df.append(after_df).round(4)

    altC = (
        alt.Chart(data)
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X("NDVI:Q"),
            y=alt.Y("Density:Q"),
            color=alt.Color(
                "Date:N", legend=alt.Legend(title="Date", orient="top-left")
            ),
            tooltip=[
                alt.Tooltip("Date:N", title="Date"),
                alt.Tooltip("NDVI:Q", title="NDVI", format=",.4f"),
                alt.Tooltip("Density:Q", title=f"{tooltip_title}", format=",.4f"),
            ],
        )
    )

    altCa = (
        alt.Chart(data)
        .mark_line(opacity=0.3, width=2)
        .encode(
            x=alt.X("NDVI:Q"),
            y=alt.Y("Density:Q"),
            color=alt.Color(
                "Date:N"
            )
        )
    )

    nearest, selectors, rules, points, text = ruler(data, "NDVI", "Q", altC, "Density")
    fig1 = altCa + altC + rules + selectors + points + text
    fig1.save('./assets/fig1.png')
    st.altair_chart(fig1, use_container_width=True)

    st.markdown(
        f"<center>Figure 1. Distribution of NDVI values for images of selected dates</center><br>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p align="justify">Figure 1 shows the distribution of NDVI values for each pixel (100-meter resolution). The
        <strong><font color="#9198A2">blue curve</font></strong> correspond to the earliest available image between the selected dates (i.e., <strong>{startdate}</strong>), 
        while the <strong><font color="#BAA083">orange curve</font></strong> correspond to the most recent available image (i.e., <strong>{enddate}</strong>).</p>
        <p align="justify">For the image on {startdate}, the figure shows that 50% of the data lies between <strong>{start_lower:.2f} - {start_upper:.2f}</strong>
        (inclusive), while for the image on {enddate}, 50% of the data lies between <strong>{end_lower:.2f} - {end_upper:.2f}</strong>
        (inclusive).</p>
        """,
        unsafe_allow_html=True,
    )

    option3, option1, option2 = st.columns((1, 2, 2))
    with option3:
        select = st.radio(
            label="Select Wrangling",
            options=["Raw (None)", "Standardize", "Smoothen"],
            help="Options to visualize raw or transformed data",
        )

    rule_option_dict = {
        "Mean": "mean",
        "Median": "median",
        "Maximum": "max",
        "Minimum": "min",
    }
    band_option_dict = {
        "Inter-quartile Range": "iqr",
        "Confidence Interval": "ci",
        "Standard Error": "stderr",
        "Standard Deviation": "stdev",
    }

    annual = (
        alt.Chart(df_annual)
        .mark_circle(opacity=0)
        .encode(x=alt.X("Timestamp:T"), y=alt.Y("NDVI:Q"),)
    )

    if select == "Smoothen":
        with option1:
            rule_option = st.selectbox(
                label="Select Line Aggregation",
                options=[],
                help="This option is not available with the current selection",
            )

        with option2:
            band_option = option2.selectbox(
                label="Select Error Bar Extent",
                options=[],
                help="This option is not available with the current selection",
            )

        baseC = alt.Chart(df).encode(
            x=alt.X("Timestamp:T", title=None),
            y=alt.Y(
                "NDVI_Lowess:Q",
                title="NDVI",
                scale=alt.Scale(domain=[df.NDVI_Lowess.min(), df.NDVI_Lowess.max()]),
            ),
        )

        linesC = baseC.mark_line().encode(
            tooltip=[
                alt.Tooltip("Timestamp:T", title="Date"),
                alt.Tooltip("NDVI_Lowess:Q", title="NDVI", format=",.4f"),
            ]
        )

        regC = (
            baseC.transform_regression("Timestamp", "NDVI_Lowess")
            .mark_line(color="#C32622")
            .encode()
        )

        nearest, selectors, rules, points, text = ruler(
            df, "Timestamp", "T", linesC, "NDVI_Lowess:Q"
        )

        fig2 = (
            annual + linesC + regC + rules + selectors + points + text
        ).interactive(bind_y=False)
        fig2.save('./assets/fig2.png')

        st.altair_chart(fig2, use_container_width=True)
        st.markdown(
            "<center>Figure 2. Smoothed Trend of Mean NDVI Time-series</center><br>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <p align="justify">Figure 2 shows a smoothed version of the time-series plot that lessens the variations between time steps, 
            removes noise and easily visualizes the underlying trend. Given this, we can observe that the red line which corresponds to
            the best-fit line of the series is <strong>trending {trending}</strong>.</p><br>
            """,
            unsafe_allow_html=True,
        )

    elif select == "Raw (None)":
        with option1:
            rule_option = st.selectbox(
                label="Select Line Aggregation",
                options=["Mean", "Median", "Maximum", "Minimum"],
                index=1,
                help="Defines the location of the horizontal red line along the y-axis",
            )

        with option2:
            band_option = option2.selectbox(
                label="Select Error Bar Extent",
                options=[
                    "Inter-quartile Range",
                    "Confidence Interval",
                    "Standard Error",
                    "Standard Deviation",
                ],
                help="Defines the extent of the colored region",
            )

        baseA = alt.Chart(df).encode(
            x=alt.X("Timestamp:T", title=None),
            y=alt.Y(
                f"NDVI:Q", scale=alt.Scale(domain=[df["NDVI"].min(), df["NDVI"].max()])
            ),
        )

        linesA = baseA.mark_line().encode(
            tooltip=[
                alt.Tooltip("Timestamp:T", title="Date"),
                alt.Tooltip("NDVI:Q", title="NDVI", format=",.4f"),
            ]
        )

        rule = (
            alt.Chart(df)
            .mark_rule(color="red")
            .encode(
                y=alt.Y(f"{rule_option_dict[rule_option]}(NDVI):Q"),
                tooltip=[
                    alt.Tooltip(
                        f"{rule_option_dict[rule_option]}(NDVI):Q",
                        title=f"{rule_option} NDVI Line",
                        format=",.4f",
                    )
                ],
            )
        )

        bandA = (
            alt.Chart(df)
            .mark_errorband(extent=f"{band_option_dict[band_option]}")
            .encode(y=alt.Y("NDVI:Q"))
        )

        nearest, selectors, rules, points, text = ruler(
            df, "Timestamp", "T", linesA, "NDVI:Q"
        )
        fig2 = (
            annual + linesA + rule + bandA + selectors + rules + points + text
        ).interactive(bind_y=False)
        fig2.save('./assets/fig2.png')
        st.altair_chart(fig2, use_container_width=True)
        st.markdown(
            "<center>Figure 2. Mean NDVI Time-series over the AOI for imagery at every available date</center><br>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <p align="justify">Figure 2 shows a time-series that plots the average NDVI values over the AOI at each available image 
            between the selected dates. We can observe that the maximum mean NDVI (i.e., <strong>{df.NDVI.max():.2f})</strong> is observed on 
            <strong>{df.loc[df.NDVI.argmax(),'Timestamp'].strftime('%B %d, %Y')}</strong> while the minimum mean NDVI (i.e., 
            <strong>{df.NDVI.min():.2f}</strong>) is observed on <strong>{df.loc[df.NDVI.argmin(),'Timestamp'].strftime('%B %d, %Y')}</strong>. A difference of
            <strong>{df.NDVI.max()-df.NDVI.min():.2f}</strong>.</p>

            <p align="justify">The red line corresponds to the selected line aggregation of NDVI over the AOI and DOI.</p>		
            """,
            unsafe_allow_html=True,
        )

        # <p align="justify">The dots correspond to average NDVI values over the AOI aggregated per year. Maximum NDVI over a year span (i.e., <strong>{df_annual.NDVI.max():.2f})</strong>
        # is observed in <strong>{df.loc[df_annual.NDVI.argmax(),'Timestamp'].strftime('%Y')}</strong>, while minimum NDVI (i.e., <strong>{df_annual.NDVI.min():.2f})</strong>
        # is observed in <strong>{df.loc[df_annual.NDVI.argmin(),'Timestamp'].strftime('%Y')}</strong>.</p>

    elif select == "Standardize":
        with option1:
            rule_option = st.selectbox(
                label="Select Line Aggregation",
                options=[],
                help="This option is not available with the current selection",
            )

        with option2:
            band_option = option2.selectbox(
                label="Select Error Bar Extent",
                options=[],
                help="This option is not available with the current selection",
            )

        baseA = alt.Chart(df).encode(
            x=alt.X("Timestamp:T", title=None),
            y=alt.Y("Standard:Q"),
            tooltip=[
                alt.Tooltip("Timestamp:T"),
                alt.Tooltip("Standard:Q", format=",.4f"),
            ],
        )

        barA = baseA.mark_bar().encode(
            color=alt.Color(
                "Standard:Q", scale=alt.Scale(scheme="redblue"), legend=None
            ),
            tooltip=[
                alt.Tooltip("Timestamp:T", title="Date"),
                alt.Tooltip("NDVI:Q", title="NDVI", format=",.4f"),
            ],
        )

        rule = alt.Chart(df).mark_rule(color="red").encode(y=alt.Y("mean(Standard):Q"))

        nearest, selectors, rules, points, text = ruler(
            df, "Timestamp", "T", barA, "Standard:Q"
        )
        fig2 = (annual + barA + rule + selectors + rules + points + text).interactive(
            bind_y=False
        )
        fig2.save('./assets/fig2.png')
        st.altair_chart(fig2, use_container_width=True)
        st.markdown(
            "<center>Figure 2. Standardized Mean NDVI Time-series over the AOI for imagery at every available date</center><br>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <p align="justify">Figure 2 shows a time-series that plots the standardized value average NDVI values over the AOI at each available image 
            between the selected dates, setting the mean of the series to zero (0) with a
            standard deviation equal to one (1). We can observe that the maximum mean NDVI (i.e., <strong>{df.Standard.max():.2f})</strong> is observed on 
            <strong>{df.loc[df.Standard.argmax(),'Timestamp'].strftime('%B %d, %Y')}</strong> while the minimum mean NDVI (i.e., 
            <strong>{df.Standard.min():.2f}</strong>) is observed on <strong>{df.loc[df.Standard.argmin(),'Timestamp'].strftime('%B %d, %Y')}</strong>. A difference of
            <strong>{df.Standard.max()-df.Standard.min():.2f}</strong>.</p>
            """,
            unsafe_allow_html=True,
        )

    baseB = alt.Chart(df).encode(
        x=alt.X("DOY:Q", scale=alt.Scale(domain=(0, 340)), title="DOY")
    )

    lower = df.groupby("DOY")["NDVI"].quantile(0.25).min()
    upper = df.groupby("DOY")["NDVI"].quantile(0.75).max()

    lineB = baseB.mark_line().encode(
        y=alt.Y("median(NDVI):Q", scale=alt.Scale(domain=[lower, upper])),
        tooltip=[
            alt.Tooltip("DOY:Q", title="DOY"),
            alt.Tooltip("median(NDVI):Q", title="Median NDVI"),
        ],
    )

    rainy_df = pd.DataFrame({"x1": [152], "x2": [334]})

    rainy_season = (
        alt.Chart(rainy_df)
        .mark_rect(opacity=0.2, color="#A5DCFF")
        .encode(
            x=alt.X("x1", title=""),
            x2="x2",
            y=alt.value(0),  # 0 pixels from top
            y2=alt.value(300),  # 300 pixels from top
        )
    )

    dry_df = pd.DataFrame({"x1": [0, 334], "x2": [152, 360]})

    dry_season1 = (
        alt.Chart(dry_df)
        .mark_rect(opacity=0.2, color="#b47e4f")
        .encode(
            x=alt.X("x1", title=""),
            x2="x2",
            y=alt.value(0),  # 0 pixels from top
            y2=alt.value(300),  # 300 pixels from top
        )
    )

    bandB = baseB.mark_errorband(extent="iqr", color="#3D3D45", opacity=0.3).encode(
        y="NDVI:Q"
    )

    nearest, selectors, rules, points, text = ruler(
        df, "DOY", "Q", lineB, "median(NDVI)"
    )

    fig3 = (
        lineB + bandB + rainy_season + dry_season1 + rules + selectors + points + text
    ).interactive()
    fig3.save('./assets/fig3.png')
    st.altair_chart(fig3, use_container_width=True)
    st.markdown(
        "<center>Figure 3. Variation in NDVI values per Day of Year (DOY)</center><br>",
        unsafe_allow_html=True,
    )

    def q75(x):
        return x.quantile(0.75)

    def q25(x):
        return x.quantile(0.25)

    doy_df = df.groupby("DOY")[["NDVI"]].agg({"NDVI": [q75, q25, "median"]})
    doy_df.columns = doy_df.columns.get_level_values(0)
    doy_df.reset_index(inplace=True)
    doy_df.columns = ["DOY", "Q75", "Q25", "Median"]
    doy_df["variation"] = doy_df.Q75 - doy_df.Q25

    max_day = doy_df.loc[doy_df.Median.argmax(), "DOY"]
    min_day = doy_df.loc[doy_df.Median.argmin(), "DOY"]
    var_max_day = doy_df.loc[doy_df.variation.argmax(), "DOY"]
    var_min_day = doy_df.loc[doy_df.variation.argmin(), "DOY"]

    if str(max_day)[-1] == "1":
        max_str = "st"
    elif str(max_day)[-1] == "2":
        max_str = "nd"
    elif str(max_day)[-1] == "3":
        max_str = "rd"
    else:
        max_str = "th"

    if str(min_day)[-1] == "1":
        min_str = "st"
    elif str(min_day)[-1] == "2":
        min_str = "nd"
    elif str(min_day)[-1] == "3":
        min_str = "rd"
    else:
        min_str = "th"

    if str(var_max_day)[-1] == "1":
        var_max_str = "st"
    elif str(var_max_day)[-1] == "2":
        var_max_str = "nd"
    elif str(var_max_day)[-1] == "3":
        var_max_str = "rd"
    else:
        var_max_str = "th"

    if str(var_min_day)[-1] == "1":
        var_min_str = "st"
    elif str(var_min_day)[-1] == "2":
        var_min_str = "nd"
    elif str(var_min_day)[-1] == "3":
        var_min_str = "rd"
    else:
        var_min_str = "th"

    st.markdown(
        f"""
        <p align="justify">Figure 3 shows the median of mean NDVI, represented by the <font color="#5378A9">blue line</font> 
        and the corresponding variation per day (i.e., Inter-quartile range)
        across a year, represented by the <font color="#BFC9D5">light-blue band</font>.</p>
        
        <p align="justify">The maximum NDVI (i.e., <strong>{doy_df.Median.max():.2f}</strong>) is measured on the <strong>{max_day}{max_str} day</strong>, while the minimum
        NDVI (i.e., <strong>{doy_df.Median.min():.2f}</strong>) is measured on the <strong>{min_day}{min_str} day</strong> of the year.</p>

        <p align="justify">The largest variation in NDVI values is observed on the <strong>{var_max_day}{var_max_str} day </strong>of the year, while the 
        smallest variation is observed on the <strong>{var_min_day}{var_min_str} day</strong>.</p>

        The two (2) colored regions correspond to the dry (orange) and wet (blue) season.

        <p align="justify"><em><font color="#85221A">Note: The information presented in this app 
        serves as a guide only. Ground validation should still be conducted to verify its accuracy.</font></em></p> 
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
