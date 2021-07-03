import os
import ee
import json
import base64
import geemap
from numpy.lib.function_base import select
import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
from fpdf import FPDF
from toolz.functoolz import do
from html2image import Html2Image
from datetime import timedelta, datetime

import streamlit.components.v1 as components
from streamlit_folium import folium_static

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import folium
from folium import plugins
  
st.set_page_config(page_title='Green App', initial_sidebar_state='expanded', page_icon='🌳')

# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    )}

# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    
    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):    
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
    
    except:
        print("Could not display {}".format(name))
    
# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

@st.cache(ttl=60*60*1, allow_output_mutation=True, persist=True)
def read_data(aoi):
	l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_8DAY_NDVI').select('NDVI')
	scale = 100

	out_dir = os.path.join(os.getcwd(), 'assets')

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	out_stats = os.path.join(out_dir, 'ndvi_stats.csv')

	geemap.zonal_statistics(l8, aoi, out_stats, statistics_type='MEAN', scale=scale)

	df = pd.read_csv(out_stats)
	df = pd.melt(df, id_vars=['system:index'])
	df.variable = df.variable.apply(lambda x: x[:8])
	df = df.rename(columns={'value':'NDVI', 'variable':'Timestamp', 'system:index':'Default'})

	df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
	df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
	df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
	df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
	df['Timestamp'] = pd.DatetimeIndex(df['Timestamp']).date

	df.dropna(axis=0, inplace=True)
	df.drop(columns=['Default'], inplace=True)

	df['NDVI_Lowess'] = lowess(df.NDVI.values, np.arange(len(df.NDVI.values)), frac=0.05)[:,1]

	end = (datetime(2000,1,1) + timedelta(df.shape[0]-1)).strftime('%Y-%m-%d')
	temp_index = pd.date_range(start='2000-01-01', end=end, freq='D')
	df = df.set_index(temp_index)
	decomposition = sm.tsa.seasonal_decompose(df.NDVI)
	df['Trend'] = decomposition.trend
	df['Seasonal'] = decomposition.seasonal
	df.reset_index(drop=True, inplace=True)

	return df

@st.cache(ttl=60*60*1, allow_output_mutation=True, suppress_st_warning=True)
def read_img(startdate, enddate, aoi):
	start1 = startdate.strftime('%Y-%m-%d')
	start2 = (startdate+timedelta(1)).strftime('%Y-%m-%d')
	end1 = enddate.strftime('%Y-%m-%d')
	end2 = (enddate+timedelta(1)).strftime('%Y-%m-%d')

	area = aoi.geometry().area().getInfo()/10000

	l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_8DAY_NDVI')
	scale=100

	start_img = l8.select('NDVI').filterDate(start1, start2).first().unmask()
	end_img = l8.select('NDVI').filterDate(end1, end2).first().unmask()
	
	start_img = start_img.reproject(crs=ee.Projection('EPSG:3395'), scale=scale)
	end_img = end_img.reproject(crs=ee.Projection('EPSG:3395'), scale=scale)

	diff_img = end_img.subtract(start_img)

	classes = ['Soil/Water [-1-0)', 'Very Low [0-0.2)', 'Low [0.2-0.4)', 
			'Mod. Low [0.4-0.6)', 'Mod. High [0.6-0.8)', 'High [0.8-1.0]']

	bins = [-1, 0 ,0.2 ,0.4 ,0.6 ,0.8 ,1.0]
	
	region = aoi.geometry()

	start_arr = geemap.ee_to_numpy(start_img, region=region)
	end_arr = geemap.ee_to_numpy(end_img, region=region)
	
	try:	
		start_bin = np.histogram(start_arr, bins=bins)[0]
	except:
		st.error('Too many pixels in sample; must be <= 262144. Please select a smaller region.')
		return 

	start_bin_norm = start_bin/sum(start_bin)

	end_bin = np.histogram(end_arr, bins=bins)[0]
	end_bin_norm = end_bin/sum(end_bin)

	diff_arr = geemap.ee_to_numpy(diff_img, region=aoi.geometry())
	diff_arr = diff_arr/10_000

	bin_list = [-1, 0, 1]
	diff_bin = np.histogram(diff_arr, bins=bin_list)[0]
	diff_bin_norm = diff_bin/sum(diff_bin)

	hist_df = pd.DataFrame({
		f'{start1}':start_arr.flatten(),
		f'{end1}':end_arr.flatten(),
		})

	with np.errstate(divide='ignore', invalid='ignore'):
		perc_change = np.divide((end_bin - start_bin), start_bin)
		perc_change[np.isnan(perc_change)] = 0
		perc_change[np.isinf(perc_change)] = 0

	interpretation = ['Increase' if i > 0 else 'No change' if  i == 0 else 'Decrease' for i in list(perc_change)]

	df = pd.DataFrame({f'{startdate}': start_bin_norm*area, 
			f'{enddate}': end_bin_norm*area,
			'%Change': perc_change*100, 'Interpretation': interpretation}, index=classes)

	report_df = df.copy()
	
	df.reset_index(inplace=True)
	df = df.rename(columns={'index':'NDVI Class'})

	return df, report_df, start_img, end_img, diff_img, diff_bin_norm, hist_df

def transform(df):
	dfm = df.copy()
	dfm.index = pd.DatetimeIndex(dfm.Timestamp)
	dfm = dfm.resample('A', label='left').mean()
	dfm.reset_index(inplace=True)
	return dfm

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

class PDF(FPDF):
    def header(self):
        # Logo
        self.image(r'./assets/header.jpg', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Vegetation Health Assessment Report', 0, 0, 'C')
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

def main():
	# st.write('Export maps and viz into reports, measure time for page to load, forecast, team members and references tab')
	st.image(r'./assets/header.jpg')
	st.sidebar.subheader('Customization Panel')
	navigation = st.sidebar.selectbox('Navigation', ['Assessment', 'Manual', 'Download file', 'Team members', 'References'])
	if navigation == 'Assessment':
		st.title('Vegetation Health Assessment App')
		st.markdown(f"""
			<p align="justify">The web app aims to provide helpful information that can aid in efforts to protect our forest
			ecosystem. It uses the <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_TOA">Landsat 8 Collection 1 Tier 1</a> 
			8-Day Normalized Difference Vegetation Index (NDVI) composite at 30-meter resolution to provide 
			an overview of vegetation health of an area from 2013 - present.</p>

			<p align="justify"><a href="https://gisgeography.com/ndvi-normalized-difference-vegetation-index/">NDVI</a> is an indicator used to quantify vegetation health based on how the vegetation respond to 
			light at the Near Infrared (NIR) and Red bands. NDVI ranges from -1.0 to +1.0.</p> 

			<p align="justify">NDVI <em>less than 0.2 are classified as Non-vegetation</em> (barren lands, build-up area, road network), <em>between 0.2 to 0.5 (inclusive) as Low Vegetation</em> 
			(shrubs and grassland ) and <em>greater than 0.5 to 1.0 as Dense vegetation</em> (temperate and tropical urban forest).

			<p align="justify"><em>Note: Reliable ground truth activities are still needed to verify the accuracy of the information generated by this web app.</em></p> 
			
			<p align="justify">Instructions: You can either <strong>upload a file</strong> (*.shp format) or <strong>draw a bounding box</strong> of the region of interest
			and input the coordinates (i.e, GeoJSON format) to obtain the historical NDVI values for that region.</p> 
			""", unsafe_allow_html=True)

		with st.beta_expander('Click to draw Region of Interest (ROI)'):
			st.markdown(f"""
				You can select the region of interest (ROI) here. \n
				Click OK to accept cookies. Select ROI by utilizing the toolbars located at
				the upper left of the map or using the search box.
				
				**Important!** At the lower left corner, select the **GEOJSON** format and copy
				the coordinates.
				""")
			inputFile = st.file_uploader('Upload a file', type=['shp'])
			components.iframe('https://boundingbox.klokantech.com/', height=500)

		inputRegion = st.text_input(f'Paste ROI coordinates here and press Enter:')

		Map = geemap.Map()

		default_region = '[[[120.9845116619,14.5558642572],\
							[121.1177208904,14.5558642572],\
							[121.1177208904,14.6030462531],\
							[120.9845116619,14.6030462531],\
							[120.9845116619,14.5558642572]]]'
		
		try:
			if inputFile is None and len(inputRegion)==0:
				# region = geemap.shp_to_geojson(r'./assets/butuan_city_gcs.shp')
				# data = region['features'][0]['geometry']['coordinates']
				data = json.loads(default_region)

			elif inputFile and len(inputRegion) == 0:
				region = geemap.shp_to_ee(os.path.join(os.getcwd(),'assets',inputFile.name))
				data = region.geometry().getInfo()['coordinates']

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
		
		df = read_data(aoi)
		df_annual = transform(df)

		lon, lat = aoi.geometry().centroid().getInfo()['coordinates']
		
		st.session_state.lon = lon
		st.session_state.lat = lat

		st.markdown(f"""
			Area of ROI is *{aoi.geometry().area().getInfo()/10000:,.02f}* has. The centroid is 
			located at *({lat:.02f} N, {lon:.02f} E)*.
			""")

		startdate, enddate = st.select_slider('Use the slider to select the start and end date', 
			df.Timestamp.unique().tolist(), 
			value=[df.Timestamp.unique().tolist()[0], df.Timestamp.unique().tolist()[-1]])

		startdate_format = startdate.strftime('%B %d, %Y')
		enddate_format = enddate.strftime('%B %d, %Y')

		df = df[(df.Timestamp >= startdate) & (df.Timestamp <= enddate)]
		
		export_df, report_df, start_img, end_img, diff_img, diff_bin_norm, hist_df = read_img(startdate, enddate, aoi)
			
		visParams = {
			'bands': ['NDVI'],
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
			'bands': ['NDVI'],
			'min': -1,
			'max': 1,
				'palette': [
				'#FF4C56', '#FFFFFF', '#73DA6E'
			],
			'opacity':.7,
			}

		# Create a folium map object.
		my_map = folium.Map(location=[lat, lon], zoom_start=12)

		# Add custom basemaps
		basemaps['Google Maps'].add_to(my_map)
		basemaps['Google Satellite Hybrid'].add_to(my_map)

		my_map.add_ee_layer(start_img.clip(aoi.geometry()), visParams, f'{startdate}_IMG')
		my_map.add_ee_layer(end_img.clip(aoi.geometry()), visParams, f'{enddate}_IMG')
		my_map.add_ee_layer(diff_img.clip(aoi.geometry()), visParams_diff, 'NDVI Diff_IMG')

		# # Add a layer control panel to the map.
		my_map.add_child(folium.LayerControl())
		my_map.add_child(folium.LatLngPopup())

		# # Add fullscreen button
		plugins.Fullscreen().add_to(my_map)
		plugins.MiniMap().add_to(my_map)

		folium_static(my_map)
		
		timelapse = st.checkbox('Check to show a timelapse of annual Landsat composite of the selected region')
		
		if timelapse:
			out_dir = os.path.join(os.getcwd(), 'assets')

			if not os.path.exists(out_dir):
				os.makedirs(out_dir)

			out_gif = os.path.join(out_dir, 'landsat_ts.gif')

			s1 = startdate.strftime('%Y-%m-%d')
			d1 = enddate.strftime('%Y-%m-%d')

			band_options = {'Color Infrared (Vegetation)': ['NIR', 'Red', 'Green'],
						'False Color': ['SWIR2', 'SWIR1', 'Red'],
						'Natural Color': ["Red", "Green", 'Blue'],
						'Agriculture': ['SWIR1', 'NIR', 'Blue'],
						'Healthy Vegetation': ['NIR', 'SWIR1', 'Blue'],
						'Land/Water': ['NIR', 'SWIR1', 'Red'],
						'Natural with Atmospheric Removal': ['SWIR2', 'NIR', 'Green'],
						'Shortwave Infrared': ['SWIR2', 'NIR', 'Green'],
						'Vegetation Analysis': ['SWIR1', 'NIR', 'Red']}

			ts_bands = st.selectbox('Select bands to visualize', list(band_options.keys()))

			Map.add_landsat_ts_gif(
				layer_name="Timelapse",
				roi=aoi.geometry(),
				# label='Timelapse',
				start_year=startdate.year,
				end_year=enddate.year,
				start_date=s1[5:],
				end_date=d1[5:],
				bands=band_options[ts_bands],
				vis_params=None,
				dimensions=700,
				frames_per_second=.5,
				font_size=30,
				font_color="white",
				add_progress_bar=True,
				progress_bar_color="cyan",
				progress_bar_height=5,
				out_gif=out_gif,
				download=True)

			file_ = open(out_gif, "rb")
			contents = file_.read()
			data_url = base64.b64encode(contents).decode("utf-8")
			file_.close()
			
			st.markdown(f"""<center><img src="data:image/gif;base64,{data_url}" alt="timelapse gif"></center>""",
				unsafe_allow_html=True)
					
		# create an interval selection over an x-axis encoding
		brush = alt.selection_interval(encodings=['x'])

		# determine opacity based on brush
		opacity = alt.condition(brush, alt.value(0.9), alt.value(0.1))

		highlightA = alt.selection(
		type='single', on='mouseover', fields=['Year'], nearest=True)

		annual = alt.Chart(df_annual).mark_circle(size=50).encode(
			x=alt.X('Timestamp:T'),
			y=alt.Y('NDVI:Q'),
			# opacity=opacity,
			color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None),
			tooltip=[
				alt.Tooltip('NDVI:Q', title='Annual Mean', format=',.4f')
			])

		df['Timestamp'] = pd.to_datetime(df.Timestamp)
		baseA = alt.Chart(df).encode(
			x=alt.X('Timestamp:T', title=None),
			y=alt.Y('NDVI:Q'),
			# opacity=opacity
			)

		pointsA = baseA.mark_circle().encode(
			opacity=alt.value(0),
			tooltip=[
				alt.Tooltip('Timestamp:T', title='Date'),
				alt.Tooltip('NDVI:Q', title='NDVI')
			]).add_selection(highlightA)

		linesA = baseA.mark_line().encode(
			size=alt.condition(~highlightA, alt.value(1), alt.value(3)),
			color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None))

		rule = alt.Chart(df).mark_rule(color='red').encode(
			y=alt.Y('mean(NDVI):Q'),
			tooltip=[alt.Tooltip('mean(NDVI):Q', title='Mean NDVI Line', format=',.4f')])

		baseC = alt.Chart(df).encode(
			x=alt.X('Timestamp:T', title=None),
			y=alt.Y('NDVI_Lowess:Q', title='NDVI', scale=alt.Scale(domain=[df.NDVI_Lowess.min(), df.NDVI_Lowess.max()]))
			)

		pointsC = baseC.mark_circle().encode(
			opacity=alt.value(0),
			tooltip=[
				alt.Tooltip('Timestamp:T', title='Date'),
				alt.Tooltip('NDVI_Lowess:Q', title='NDVI')
			]).add_selection(highlightA)

		linesC = baseC.mark_line().encode(
			size=alt.condition(~highlightA, alt.value(1), alt.value(3)),
			color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None))

		regC = baseC.transform_regression('Timestamp', 'NDVI_Lowess').mark_line(color="#C32622").encode(
			size=alt.condition(~highlightA, alt.value(1), alt.value(3)))

		altA = (annual + pointsA + linesA + rule).interactive(bind_y = False)
		altAA = (pointsC + linesC + regC).interactive(bind_y = False)

		baseB = alt.Chart(df).encode(
			x=alt.X('DOY:Q', scale=alt.Scale(domain=(0, 340)))) # )

		lower = df.groupby('DOY')['NDVI'].quantile(.25).min()
		upper = df.groupby('DOY')['NDVI'].quantile(.75).max()

		lineB = baseB.mark_line().encode(
			y=alt.Y('median(NDVI):Q', 
				scale=alt.Scale(domain=[lower,upper])
				)
			)

		bandB = baseB.mark_errorband(extent='iqr').encode(
			y='NDVI:Q')

		altB = (lineB + bandB).interactive()

		start_lower = hist_df[f'{startdate}'].quantile(.25).min()
		start_upper = hist_df[f'{startdate}'].quantile(.75).max()
		end_lower = hist_df[f'{enddate}'].quantile(.25).min()
		end_upper = hist_df[f'{enddate}'].quantile(.75).max()
		
		altC = alt.Chart(hist_df).transform_fold([f'{startdate}', f'{enddate}'], as_=['Dates', 'NDVI']
			).mark_area( opacity=0.3, interpolate='step').encode(
				x=alt.X('NDVI:Q', bin=alt.Bin(maxbins=200)),
				y=alt.Y('count()', stack=None),
				color=alt.Color('Dates:N', legend=None),
				tooltip=[alt.Tooltip('Dates:N', title='Date'),
						alt.Tooltip('NDVI:Q', bin=alt.Bin(maxbins=100)),
						alt.Tooltip('count()', title='Count')]
			).interactive()
		
		#save visualizations as png
		# altA.save('chart1.png')
		# altAA.save('chart1a.png')
		# (lineB + bandB).save('chart2.png')
		# altC.save('chart3.png')

		x = np.array(pd.to_datetime(df.Timestamp), dtype=float)
		y = df.NDVI_Lowess
		results = np.polyfit(x,y, deg=1)
		slope, _ = results
		_, positive_change = diff_bin_norm
		mean_ndvi = df.NDVI.mean()

		if slope > 0:
			trending = 'UP'
		else:
			trending =' DOWN'

		st.write(' ')
		st.info(f"""Overall, mean NDVI for the *selected region* and *dates* is **{mean_ndvi:0.3f}** and is **trending {trending}!** 📈 and \
					the area where a **positive NDVI change** is observed is at **{positive_change:0.2%}**!
					\nSelected dates: **{startdate_format} - {enddate_format}**   
					Number of days betweet selected dates: **{(enddate - startdate).days:,} days**    
					Number of datapoints between the selected dates: **{df.shape[0]}**   
					""")

		decrease_list = list(report_df[report_df.Interpretation.isin(['Decrease'])].index)
		increase_list = list(report_df[report_df.Interpretation.isin(['Increase'])].index)

		st.markdown('Table 1. Area (in hectares) and Percent Change between Selected Dates across NDVI classes')

		st.dataframe(report_df)
		st.markdown(f"""
			<p align="justify">The NDVI values were classified into six (6) classes: <em>{', '.join(list(report_df.index))}</em>.</p>

			<p align="justify">A decrease in the first four (4) NDVI classes (i.e., -1 - 0.6) or an increase in the remaining NDVI classes
			(i.e., 0.6 above) is interpreted as positive and vice versa. </p>
			
			<p align="justify">Given this and inspecting the table above, we observed increase in the following NDVI landcover classes: 
			<strong>{', '.join(increase_list)}</strong> while we observed a decrease in the following: <strong>{', '.join(decrease_list)}</strong>.</p>
			""", unsafe_allow_html=True)
		st.markdown('---')
		st.subheader('Exploratory Visualization')
		st.markdown(f'Figure 1. Distribution of NDVI values for images of selected dates', unsafe_allow_html=True)
		st.altair_chart(altC, use_container_width=True)
		st.markdown(f"""
			<p align="justify">Figure 1 shows the distribution of NDVI values for each pixel (100-meter resolution). The
			<strong><font color="#9198A2">blue bars</font></strong> correspond to the earliest available image between the selected dates (i.e., <strong>{startdate}</strong>), 
			while the <strong><font color="#BAA083">orange bars</font></strong> corresponds to the most recent available image (i.e., <strong>{enddate}</strong>).</p>

			<p align="justify">For the image on {startdate}, the figure shows that 50% of the data lies between <strong>{start_lower:.2f} - {start_upper:.2f}</strong>
			(inclusive), while for the image on {enddate}, 50% of the data lies between <strong>{end_lower:.2f} - {end_upper:.2f}</strong>
			(inclusive).</p>
			""", unsafe_allow_html=True)

		st.markdown('Figure 2. Mean NDVI Time-series over the region for imagery at every available date')
		st.altair_chart(altA, use_container_width=True)
		st.markdown(f"""
			<p align="justify">Figure 2 shows a time-series that plots the average NDVI values over the ROI at each available image 
			between the selected dates. We can observe that the maximum mean NDVI (i.e., <strong>{df.NDVI.max():.2f})</strong> is observed on 
			<strong>{df.loc[df.NDVI.argmax(),'Timestamp'].strftime('%B %d, %Y')}</strong> while the minimum mean NDVI (i.e., 
			<strong>{df.NDVI.min():.2f}</strong>) is observed on <strong>{df.loc[df.NDVI.argmin(),'Timestamp'].strftime('%B %d, %Y')}</strong>. A difference of
			<strong>{df.NDVI.max()-df.NDVI.min():.2f}</strong>.</p>

			<p align="justify">The dots correspond to average NDVI values over the ROI aggregated per year. Maximum NDVI over a year span (i.e., <strong>{df_annual.NDVI.max():.2f})</strong>
			is observed on <strong>{df.loc[df_annual.NDVI.argmax(),'Timestamp'].strftime('%Y')}</strong>, while mimimum NDVI (i.e., <strong>{df_annual.NDVI.min():.2f})</strong>
			is observed on <strong>{df.loc[df_annual.NDVI.argmin(),'Timestamp'].strftime('%Y')}</strong>.</p>

			<p align="justify">The red line corresponds to the average NDVI over the ROI across the selected dates.</p>
			""", unsafe_allow_html=True)

		st.markdown('Figure 3. Smoothed Trend of Mean NDVI Time-series')
		st.altair_chart(altAA, use_container_width=True)
		st.markdown(f"""
			<p align="justify">Figure 3 shows a smoothed version of the time-series plot that lessen the variations between time steps, 
			remove noise and easily visualize the underlying trend. Given this, we can observe that the red line which corresponds to
			the best-fit line of the series is <strong>trending {trending}</strong>.</p>
			""", unsafe_allow_html=True)
		st.markdown('Figure 4. Variation in NDVI values per Day of Year (DOY)')
		st.altair_chart(altB, use_container_width=True)

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
			<p align="justify">Figure 4 shows the median of mean NDVI and corresponding variation per day (i.e., Inter-quartile range) across a year. 
			The maximum NDVI (i.e., <strong>{doy_df.Median.max():.2f}</strong>) is measured on the <strong>{max_day}{max_str} day</strong>, while the minimum
			NDVI (i.e., <strong>{doy_df.Median.min():.2f}</strong>) is measured on the <strong>{min_day}{min_str} day</strong> of the year.</p>

			<p align="justify">The largest variation in NDVI values is observed on the <strong>{var_max_day}{var_max_str} day </strong>of the year, while the 
			smallest variation is observed on <strong>{var_min_day}{var_min_str} day</strong>.</p>
			""", unsafe_allow_html=True)

		st.markdown('---')
		with st.beta_expander('Discussion board.', expanded=True):
			components.iframe('https://padlet.com/kaquisado/v9y0rhx2lrf0tilk', height=500)

		export_as_pdf = st.sidebar.button("Generate Summary Report")

		if export_as_pdf:
			pdf = PDF()
			hti = Html2Image()
			pdf.alias_nb_pages()
			pdf.add_page()
			pdf.set_font('Times', '', 12)
			pdf.cell(0, 10, f'Report Generated on {datetime.now().date()}', 0, 1)
			
			export_map = folium.Map(location=[lat, lon], zoom_start=14)
			basemaps['Google Satellite Hybrid'].add_to(export_map)
			# export_map.add_ee_layer(start_img.clip(aoi.geometry()), visParams, f'{startdate}_IMG')
			# export_map.add_ee_layer(end_img.clip(aoi.geometry()), visParams, f'{enddate}_IMG')
			export_map.add_ee_layer(diff_img.clip(aoi.geometry()), visParams_diff, '')
			plugins.MiniMap().add_to(export_map)
			export_map.save('map.html')

			hti.screenshot(html_file='map.html', save_as='map.png')
			pdf.image('map.png', w=190, h=100)
			pdf.ln(5)
			headers = list(export_df.columns)
			w, h = 37, 10
			for header in headers:
				if header is not headers[-1]:
					pdf.cell(w=w, h=h, txt=header, border=1, ln=0, align='C')
				else:
					pdf.cell(w=w, h=h, txt=header, border=1, ln=1, align='C')
				
			for row in range(export_df.shape[0]):
				for col in headers:
					if col is not headers[-1] and col is not headers[-2]:
						try:
							text = f'{export_df.loc[row, col]:,.2f}'
						except:
							text = str(export_df.loc[row, col])
						pdf.cell(w=w, h=h, txt=text, border=1, ln=0, align='C')
					elif col is headers[-2]:
						text = f'{export_df.loc[row, col]:.2f}'
						pdf.cell(w=w, h=h, txt=text, border=1, ln=0, align='C')
					else:
						pdf.cell(w=w, h=h, txt=export_df.loc[row, col], border=1, ln=1, align='C')

			# pdf.ln(5)
			# pdf.image('chart1.png', w=190, h=100)
			# pdf.ln(5)
			# pdf.image('chart1a.png', w=190, h=100)
			# pdf.ln(5)
			# pdf.image('chart2.png', w=190, h=80)
			# pdf.ln(5)
			# pdf.image('chart3.png', w=190, h=80)
			# pdf.ln(5)
			
			html = create_download_link(pdf.output(dest="S").encode("latin-1"), f"Green Report")

			st.sidebar.markdown(html, unsafe_allow_html=True)

	elif navigation == 'Manual':
		st.title('Manual')

	else:
		st.title('Generate Report')
		
	# remove 'Made with Streamlit' footer
	hide_streamlit_style = """
				<style>
				#MainMenu {visibility: hidden;}
				footer {visibility: hidden;}
				</style>
				"""
	st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
	
if __name__ == '__main__':
	main()