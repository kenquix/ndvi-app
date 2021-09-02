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
# import extra_streamlit_components as stx

from statsmodels.nonparametric.smoothers_lowess import lowess

import folium
from folium import plugins

from utils.utils import *

def app():
    Map = geemap.Map()
    st.image(r'./assets/header.jpg', use_column_width=True)
    st.title('Vegetation Assessment and Monitoring App')
    # tab_id = stx.tab_bar(data=[
    #     stx.TabBarItemData(id=1, title="Data Collection", description=""),
    #     stx.TabBarItemData(id=2, title="Dashboard", description=""),
    #     stx.TabBarItemData(id=3, title="Data Analytics", description=""),
    # ], default=2)
    
    st.subheader('A. Data Collection')
    with st.expander('', expanded=True):
        st.markdown(f"""
            <p align="justify">Select an Area of Interest (AOI) by either <strong>(a) uploading a local file</strong> (*.KML format) or <strong>(b) drawing a bounding box</strong>.</p> 
            """, unsafe_allow_html=True)	
        inputFile = st.file_uploader('a. Upload a file', type=['kml'],
                                help='Currently, only KML format AOI is accepted.')
        st.markdown('<p style="font-size:13px">b. Draw a bounding box</p>', unsafe_allow_html=True)
        st.markdown(f"""
            <ul>
            <li>Click OK to accept cookies.</li>
            <li>Select user-defined AOI by utilizing the toolbars located at
            the upper left corner of the map or using the search box.</li>
            <li>At the lower left corner, select the <strong>GEOJSON</strong> format and copy
            the coordinates.</li><br>
            """, unsafe_allow_html=True)
        components.iframe('https://boundingbox.klokantech.com/', height=500)

        # dc1, dc2 = st.columns((3.5,1.5))
        # with dc1:
        inputRegion = st.text_input(f'Paste AOI coordinates here and press Enter:', 
                            help='Currently, only GeoJSON formatted AOI is accepted')
        # with dc2:
        #     cloud_score = st.number_input('Select cloud mask value', 0, 100, 20, 1, 
        #     help='Computes a simple cloud-likelihood score in the range [0,100]. Default is 20')

        default_region = '[[[125.4727148947,8.9012996164],\
                            [125.5990576681,8.9012996164],\
                            [125.5990576681,8.9828386907],\
                            [125.4727148947,8.9828386907],\
                            [125.4727148947,8.9012996164]]]'
        zip_dir = os.path.join(os.path.expanduser("~"), 'assets')

        try:
            if inputFile is None and len(inputRegion)==0:
                data = json.loads(default_region)

            elif inputFile is not None and len(inputRegion) == 0:	
                gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'	
                input_df = gpd.read_file(inputFile, driver='KML')
                s = str(input_df.geometry.iloc[0])
                g1 = shapely.wkt.loads(s)
                g2 = shapely.geometry.mapping(g1)
                g3 = json.dumps(g2)
                data = ast.literal_eval(g3)['coordinates']
                data = [[i[:2] for i in data[0]]]

            else:
                if inputRegion[:3] == '[[[':
                    data = json.loads(inputRegion)
                else:
                    inputRegion = '[[' + inputRegion + ']]'
                    data = json.loads(inputRegion)
        except:
            st.error(f'Error: Expected a different input. Make sure you selected the GEOJSON format.')
            return

    aoi = ee.FeatureCollection(ee.Geometry.Polygon(data))
    datamask = ee.Image('UMD/hansen/global_forest_change_2015').select('datamask').eq(1)
    lon, lat = aoi.geometry().centroid().getInfo()['coordinates']
    l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterBounds(aoi)
    l8_ndvi_cloudless = l8.map(cloudlessNDVI)
    
    with st.container():		
        st.subheader('B. Dashboard')
        # with st.expander('', expanded=True):
        #     st.markdown(f"""
        #     1. Use the slider widget to select the DOI.<br>
        #     2. Wait for the app to load the NDVI composite images available for the AOI and DOI.
        #     3. Explore the layers.
        #     4. Generate timelapse of Annual Landsat composites (Convert to NDVI)
        #     """, unsafe_allow_html=True)

        with st.spinner(text="Fetching data from GEE server..."):
            date_list = date_range(l8_ndvi_cloudless, aoi)

        startdate, enddate = st.select_slider('Date Slider', 
            date_list.Timestamp.unique().tolist(), 
            value=[date_list.Timestamp.unique().tolist()[0], date_list.Timestamp.unique().tolist()[-1]],
            help="Use the slider to select the DOI's (start and end date)")

        startdate_format = startdate.strftime('%B %d, %Y')
        enddate_format = enddate.strftime('%B %d, %Y')

        with st.spinner(text="Fetching data from GEE server..."):
            df, report_df, start_img, end_img, diff_img, diff_bin_norm, hist_df = read_data(l8_ndvi_cloudless, startdate, enddate, aoi, datamask, 20)
            df['Timestamp'] = pd.to_datetime(df.Timestamp)
            df_annual = transform(df)

        visParams = {
            'min': 0,
            'max': 1,
                'palette': [
                'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
                '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
                '012E01', '011D01', '011301'
            ],
            'opacity':.8
            }

        visParams_diff = {
            'min': -1,
            'max': 1,
                'palette': [
                '#FF4C56', '#FFFFFF', '#73DA6E'
            ],
            'opacity':.7,
            }

        # Create a folium map object.
        my_map = folium.Map(location=[lat, lon], zoom_start=13)

        # Add custom basemaps
        basemaps['Google Satellite Hybrid'].add_to(my_map)

        my_map.add_ee_layer(diff_img.clip(aoi.geometry()), visParams_diff, 'Difference Image')
        my_map.add_ee_layer(start_img.clip(aoi.geometry()), visParams, f'{startdate.strftime("%d %B %Y")} Image')
        my_map.add_ee_layer(end_img.clip(aoi.geometry()), visParams, f'{enddate.strftime("%d %B %Y")} Image')
        
        # # Add a layer control panel to the map.
        my_map.add_child(folium.LayerControl())
        my_map.add_child(folium.LatLngPopup())

        # # Add fullscreen button
        plugins.Fullscreen().add_to(my_map)
        plugins.MiniMap().add_to(my_map)

    # st.markdown('---')
    # st.subheader('C. Data Analytics and Visualization')
    highlightA = alt.selection(type='single', on='mouseover', fields=['Year'], nearest=True)

    baseC = alt.Chart(df).encode(
        x=alt.X('Timestamp:T', title=None),
        y=alt.Y('NDVI_Lowess:Q', title='NDVI', 
                scale=alt.Scale(domain=[df.NDVI_Lowess.min(), 
                                df.NDVI_Lowess.max()]))
        )

    pointsC = baseC.mark_circle().encode(
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip('Timestamp:T', title='Date'),
            alt.Tooltip('NDVI_Lowess:Q', title='NDVI')
        ]).add_selection(highlightA)

    linesC = baseC.mark_line().encode(
        size=alt.condition(~highlightA, alt.value(1), alt.value(3))
        )

    regC = baseC.transform_regression('Timestamp', 'NDVI_Lowess').mark_line(color="#C32622").encode(
        size=alt.condition(~highlightA, alt.value(1), alt.value(3)))

    altAA = (pointsC + linesC + regC).interactive(bind_y=False)

    baseB = alt.Chart(df).encode(
        x=alt.X('DOY:Q', scale=alt.Scale(domain=(0, 340)), title='DOY'))

    lower = df.groupby('DOY')['NDVI'].quantile(.25).min()
    upper = df.groupby('DOY')['NDVI'].quantile(.75).max()

    lineB = baseB.mark_line().encode(
        y=alt.Y('median(NDVI):Q', 
            scale=alt.Scale(domain=[lower,upper])
            ),
        
        )

    rainy_df = pd.DataFrame({
        'x1': [152],
        'x2': [334]
    })

    rainy_season = alt.Chart(rainy_df).mark_rect(
        opacity=0.2, color='#A5DCFF'
            ).encode(
                x=alt.X('x1', title=''),
                x2='x2',
                y=alt.value(0),  # 0 pixels from top
                y2=alt.value(300)  # 300 pixels from top
                
            )

    dry_df = pd.DataFrame({
        'x1': [0, 334],
        'x2': [152, 360]
    })

    dry_season1 = alt.Chart(dry_df).mark_rect(
        opacity=0.2, color='#b47e4f'
            ).encode(
                x=alt.X('x1', title=''),
                x2='x2',
                y=alt.value(0),  # 0 pixels from top
                y2=alt.value(300),  # 300 pixels from top
            )

    bandB = baseB.mark_errorband(extent='iqr', color='#3D3D45', opacity=0.3).encode(
            y='NDVI:Q')

    pointsB = baseB.mark_circle().encode(
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip('DOY:Q', title='DOY'),
            alt.Tooltip('median(NDVI):Q', title='NDVI')
        ]).add_selection(highlightA)

    altB = (lineB + bandB + pointsB + rainy_season + dry_season1).interactive()

    start_lower = hist_df[f'{startdate}'].quantile(.25).min()
    start_upper = hist_df[f'{startdate}'].quantile(.75).max()
    end_lower = hist_df[f'{enddate}'].quantile(.25).min()
    end_upper = hist_df[f'{enddate}'].quantile(.75).max()
    
    altC = alt.Chart(hist_df[(hist_df.iloc[:,0] != 0) & (hist_df.iloc[:,1] != 0)]).transform_fold([f'{startdate}', f'{enddate}'], as_=['Dates', 'NDVI']
        ).mark_area( opacity=0.3, interpolate='step').encode(
            x=alt.X('NDVI:Q', bin=alt.Bin(maxbins=200)),
            y=alt.Y('count()', stack=None),
            color=alt.Color('Dates:N', legend=alt.Legend(orient='top-left')),
            tooltip=[alt.Tooltip('Dates:N', title='Date'),
                    alt.Tooltip('NDVI:Q', bin=alt.Bin(maxbins=100)),
                    alt.Tooltip('count()', title='Count')]
        ).interactive()
    
    x = np.array(pd.to_datetime(df.Timestamp), dtype=float)
    y = df.NDVI_Lowess
    results = np.polyfit(x,y, deg=1)
    slope, _ = results
    _, positive_change = diff_bin_norm

    mean_ndvi = df.loc[df.shape[0]-1,'NDVI']
    diff_ndvi = mean_ndvi - df.loc[0,'NDVI']

    if slope > 0:
        trending = 'Up'
    else:
        trending = 'Down'

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(f'Mean NDVI', f'{mean_ndvi:0.3f}', f'{diff_ndvi:0.3f}')
        st.metric(f'Pixel Difference', f'{positive_change:0.2%}', f'Positive Change', delta_color='off')
    
    with metric2:
        st.metric('Trend', f'{trending}', f'{slope:0.2e}')
        st.metric(f'Area (in has)', f'{millify(aoi.geometry().area().getInfo()/10000, precision=2)}', 
                f'{millify(aoi.geometry().area().getInfo()/1000000, precision=2)} KM2', delta_color='off')

    with metric3:
        st.metric(f'Cloud Cover', f'{(hist_df.iloc[:,0] == 0).mean():0.2%}', 
                f'{startdate_format}', delta_color='off')
        st.metric(f'Days between', f'{millify((enddate - startdate).days, precision=2)}', 
        f'{millify((enddate - startdate).days/365.25, precision=2)} years', delta_color='off')
    
    with metric4:
        st.metric(f'Cloud Cover', f'{(hist_df.iloc[:,1] == 0).mean():0.2%}', 
                f'{enddate_format}', delta_color='off')
        st.metric(f'Landsat 8 Images', f'{df.shape[0]}', f'~{0.919*df.shape[0]:0.2f} GB', delta_color='off')

    folium_static(my_map)

    st.image(r'./assets/scale.png', use_column_width=True)

    with st.spinner(text="Fetching data from GEE server... Generating timelapse animation..."):
        with st.expander('Timelapse of Annual NDVI Composite Images'):
            timelapse = st.checkbox('Check to generate the animation.')
            
            if timelapse:
                out_dir = os.path.join(os.path.expanduser('~'), 'assets')

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                # out_gif = os.path.join(out_dir, 'landsat_ndvi_ts.gif')

                # add bands for DOY and Year
                def add_doy(img):
                    doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year')
                    return img.set('doy', doy)
                
                def add_year(img):
                    year = ee.Date(img.get('system:time_start')).get('year')
                    return img.set('year', year)
                
                l8_ndvi_cloudless = l8_ndvi_cloudless.map(add_doy)
                l8_ndvi_cloudless = l8_ndvi_cloudless.map(add_year)

                filenames = []
                images = []
                region = aoi.geometry().bounds()
                l8_ndvi_cloudless = l8_ndvi_cloudless.select('NDVI')
                
                for i in range(startdate.year, enddate.year + 1):
                    timelapse_dir = os.path.join(zip_dir, f'landsat_{i}.png')
                    fcol = l8_ndvi_cloudless.filterMetadata('year', 'equals', i).reduce(ee.Reducer.median()).clip(aoi).updateMask(datamask)
                    geemap.get_image_thumbnail(fcol, timelapse_dir, visParams, region=region, dimensions=500, format='png')
                    img = Image.open(timelapse_dir)
                    draw = ImageDraw.Draw(img)
                    draw.text((0,0), f'Year {i}')
                    img.save(timelapse_dir)
                    filenames.append(timelapse_dir)

                for filename in filenames:
                    images.append(imageio.imread(filename))
                
                imageio.mimsave(os.path.join(zip_dir, 'landsat_ndvi_ts.gif'), images, fps=1)

                file_ = open(os.path.join(zip_dir, 'landsat_ndvi_ts.gif'), "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                
                st.markdown(f"""<center><img src="data:image/gif;base64,{data_url}" alt="timelapse gif"></center>""",
                    unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)			

    # st.write('')
    # st.info(f"""Overall, mean NDVI for the *selected AOI* and *dates* is **{mean_ndvi:0.3f}** and is **trending {trending}!** 📈 and \
    #             the areas where a **positive NDVI change** is observed is at **{positive_change:0.2%}**!
    #             \nSelected dates: **{startdate_format} - {enddate_format}**   
    #             Number of days between selected dates: **{(enddate - startdate).days:,} days**    
    #             Number of images available between the selected dates: **{df.shape[0]}**    
    #             Area of AOI is **{aoi.geometry().area().getInfo()/10000:,.02f} has**. Centroid is located at **({lat:.02f} N, {lon:.02f} E)**.    
    #             """)

    decrease_list = list(report_df[report_df.Interpretation.isin(['Decrease'])].index)
    increase_list = list(report_df[report_df.Interpretation.isin(['Increase'])].index)

    st.subheader('C. Data Analytics and Visualization')
    st.markdown('Table 1. Area (in hectares) and Percent Change between Selected Dates across NDVI classes')

    st.dataframe(report_df)
    st.markdown(f"""
        <p align="justify">A decrease in the first four (4) NDVI classes (i.e., -1 - 0.6) or an increase in the remaining NDVI classes
        (i.e., 0.6 above) is interpreted as positive and vice versa. </p>
        
        <p align="justify">Given this and inspecting the table above, we observed <font color="#2D8532"><strong>increase</strong></font> in the following NDVI landcover classes: 
        <font color="#2D8532"><strong>{', '.join(increase_list)}</strong></font> while we observed a <font color="#A42F25"><strong>decrease</strong></font> in the following classes: 
        <font color="#A42F25"><strong>{', '.join(decrease_list)}</strong></font>.</p>
        """, unsafe_allow_html=True)

    # st.markdown('---')
    
    st.altair_chart(altC, use_container_width=True)
    st.markdown(f'<center>Figure 1. Distribution of NDVI values for images of selected dates</center><br>', unsafe_allow_html=True)

    st.markdown(f"""
        <p align="justify">Figure 1 shows the distribution of NDVI values for each pixel (100-meter resolution). The
        <strong><font color="#9198A2">blue bars</font></strong> correspond to the earliest available image between the selected dates (i.e., <strong>{startdate}</strong>), 
        while the <strong><font color="#BAA083">orange bars</font></strong> correspond to the most recent available image (i.e., <strong>{enddate}</strong>).</p>
        <p align="justify">For the image on {startdate}, the figure shows that 50% of the data lies between <strong>{start_lower:.2f} - {start_upper:.2f}</strong>
        (inclusive), while for the image on {enddate}, 50% of the data lies between <strong>{end_lower:.2f} - {end_upper:.2f}</strong>
        (inclusive).</p>
        """, unsafe_allow_html=True)

    option1, option2 = st.columns((2))
    rule_option = option1.selectbox(label='Select Line Aggregation', 
                                    options=['Mean', 'Median', 'Maximum', 'Minimum'], 
                                    help='Defines the location of the horizontal red line along the y-axis')
    band_option = option2.selectbox(label='Select Error Bar Extent', 
                        options=['Inter-quartile Range', 'Confidence Interval', 
                        'Standard Error', 'Standard Deviation'],
                        help='Defines the extent of the colored region')

    standardized = st.checkbox('Click to standardize')
    rule_option_dict = {'Mean': 'mean', 'Median': 'median', 'Maximum': 'max', 'Minimum': 'min'}
    band_option_dict = {'Inter-quartile Range': 'iqr', 'Confidence Interval': 'ci', 
        'Standard Error': 'stderr', 'Standard Deviation': 'stdev'}

    annual = alt.Chart(df_annual).mark_circle(size=50).encode(
        x=alt.X('Timestamp:T'),
        y=alt.Y('NDVI:Q'),
        # opacity=opacity,
        color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(orient='bottom')),
        tooltip=[
            alt.Tooltip('NDVI:Q', title='Annual Mean', format=',.4f')
        ])

    if standardized:	
        baseA = alt.Chart(df).encode(
            x=alt.X('Timestamp:T', title=None),
            y=alt.Y('Standard:Q')
            )

        linesA = baseA.mark_bar().encode(
            color=alt.Color('Standard:Q', scale=alt.Scale(scheme='redblue'), legend=None)
            )
        
        rule = alt.Chart(df).mark_rule(color='red').encode(
            y=alt.Y('mean(Standard):Q'))

        altA = (linesA + rule).interactive(bind_y=False)

    else:
        baseA = alt.Chart(df).encode(
            x=alt.X('Timestamp:T', title=None),
            y=alt.Y('NDVI:Q'),
            )
        pointsA = baseA.mark_circle().encode(
            opacity=alt.value(0),
            tooltip=[
                alt.Tooltip('Timestamp:T', title='Date'),
                alt.Tooltip('NDVI:Q', title='NDVI')
            ]).add_selection(highlightA)

        linesA = baseA.mark_line().encode(
            size=alt.condition(~highlightA, alt.value(1), alt.value(3)),
            color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None)
            )
            
        rule = alt.Chart(df).mark_rule(color='red').encode(
            y=alt.Y(f'{rule_option_dict[rule_option]}(NDVI):Q'),
            tooltip=[alt.Tooltip(f'{rule_option_dict[rule_option]}(NDVI):Q', 
            title=f'{rule_option} NDVI Line', format=',.4f')])

        bandA = alt.Chart(df).mark_errorband(extent=f'{band_option_dict[band_option]}').encode(
            y=alt.Y('NDVI:Q'))

        altA = (annual + pointsA + linesA + rule + bandA).interactive(bind_y=False)

    st.altair_chart(altA, use_container_width=True)
    st.markdown('<center>Figure 2. Mean NDVI Time-series over the AOI for imagery at every available date</center><br>', unsafe_allow_html=True)
    st.markdown(f"""
        <p align="justify">Figure 2 shows a time-series that plots the average NDVI values over the AOI at each available image 
        between the selected dates. We can observe that the maximum mean NDVI (i.e., <strong>{df.NDVI.max():.2f})</strong> is observed on 
        <strong>{df.loc[df.NDVI.argmax(),'Timestamp'].strftime('%B %d, %Y')}</strong> while the minimum mean NDVI (i.e., 
        <strong>{df.NDVI.min():.2f}</strong>) is observed on <strong>{df.loc[df.NDVI.argmin(),'Timestamp'].strftime('%B %d, %Y')}</strong>. A difference of
        <strong>{df.NDVI.max()-df.NDVI.min():.2f}</strong>.</p>

        <p align="justify">The dots correspond to average NDVI values over the AOI aggregated per year. Maximum NDVI over a year span (i.e., <strong>{df_annual.NDVI.max():.2f})</strong>
        is observed in <strong>{df.loc[df_annual.NDVI.argmax(),'Timestamp'].strftime('%Y')}</strong>, while minimum NDVI (i.e., <strong>{df_annual.NDVI.min():.2f})</strong>
        is observed in <strong>{df.loc[df_annual.NDVI.argmin(),'Timestamp'].strftime('%Y')}</strong>.</p>

        <p align="justify">The red line corresponds to the average NDVI over the AOI and DOI.</p>		
        <p align="justify">Upon ticking the checkbox, the plot now shows the standardized values of the NDVI time-series, setting the mean of the series to zero (0) with a
        standard deviation equal to one (1).</p>
        """, unsafe_allow_html=True)

    st.altair_chart(altAA, use_container_width=True)
    st.markdown('<center>Figure 3. Smoothed Trend of Mean NDVI Time-series</center><br>', unsafe_allow_html=True)
    st.markdown(f"""
        <p align="justify">Figure 3 shows a smoothed version of the time-series plot that lessens the variations between time steps, 
        removes noise and easily visualizes the underlying trend. Given this, we can observe that the red line which corresponds to
        the best-fit line of the series is <strong>trending {trending}</strong>.</p>
        """, unsafe_allow_html=True)
    
    st.altair_chart(altB, use_container_width=True)
    st.markdown('<center>Figure 4. Variation in NDVI values per Day of Year (DOY)</center><br>', unsafe_allow_html=True)
    
    def q75(x):
        return x.quantile(0.75)

    def q25(x):
        return x.quantile(0.25)

    doy_df = df.groupby('DOY')[['NDVI']].agg({'NDVI': [q75, q25, 'median']})
    doy_df.columns = doy_df.columns.get_level_values(0)
    doy_df.reset_index(inplace=True)
    doy_df.columns = ['DOY', 'Q75', 'Q25', 'Median']
    doy_df['variation'] = doy_df.Q75 - doy_df.Q25
    
    max_day = doy_df.loc[doy_df.Median.argmax(),'DOY']
    min_day = doy_df.loc[doy_df.Median.argmin(),'DOY']
    var_max_day = doy_df.loc[doy_df.variation.argmax(),'DOY']
    var_min_day = doy_df.loc[doy_df.variation.argmin(),'DOY']

    if str(max_day)[-1] == '1':
        max_str = 'st'
    elif str(max_day)[-1] == '2':
        max_str = 'nd'
    elif str(max_day)[-1] == '3':
        max_str = 'rd'
    else:
        max_str = 'th'

    if str(min_day)[-1] == '1':
        min_str = 'st'
    elif str(min_day)[-1] == '2':
        min_str = 'nd'
    elif str(min_day)[-1] == '3':
        min_str = 'rd'
    else:
        min_str = 'th'

    if str(var_max_day)[-1] == '1':
        var_max_str = 'st'
    elif str(var_max_day)[-1] == '2':
        var_max_str = 'nd'
    elif str(var_max_day)[-1] == '3':
        var_max_str = 'rd'
    else:
        var_max_str = 'th'

    if str(var_min_day)[-1] == '1':
        var_min_str = 'st'
    elif str(var_min_day)[-1] == '2':
        var_min_str = 'nd'
    elif str(var_min_day)[-1] == '3':
        var_min_str = 'rd'
    else:
        var_min_str = 'th'

    st.markdown(f"""
        <p align="justify">Figure 4 shows the median of mean NDVI, represented by the <font color="#5378A9">blue line</font> 
        and the corresponding variation per day (i.e., Inter-quartile range)
        across a year, represented by the <font color="#BFC9D5">light-blue band</font>.</p>
        
        <p align="justify">The maximum NDVI (i.e., <strong>{doy_df.Median.max():.2f}</strong>) is measured on the <strong>{max_day}{max_str} day</strong>, while the minimum
        NDVI (i.e., <strong>{doy_df.Median.min():.2f}</strong>) is measured on the <strong>{min_day}{min_str} day</strong> of the year.</p>

        <p align="justify">The largest variation in NDVI values is observed on the <strong>{var_max_day}{var_max_str} day </strong>of the year, while the 
        smallest variation is observed on the <strong>{var_min_day}{var_min_str} day</strong>.</p>

        The two (2) colored regions correspond to the dry (orange) and wet (blue) season.
        """, unsafe_allow_html=True)

    st.markdown('---')