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

from datetime import timedelta, datetime

import streamlit.components.v1 as components
from streamlit_folium import folium_static

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import folium
from folium import plugins
  
st.set_page_config(page_title='NDVI App', initial_sidebar_state='collapsed', page_icon='🌳')

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

	interpretation = ['Increase ↑' if i > 0 else 'No change ⟷' if  i == 0 else 'Decrease ↓' for i in list(perc_change)]

	df = pd.DataFrame({f'{startdate}': start_bin_norm*area, 
			f'{enddate}': end_bin_norm*area,
			'%Change': perc_change, 'Interpretation': interpretation}, index=classes)

	df = df.style.format({f'{startdate}': "{:,.2f}", f'{enddate}': '{:,.2f}', 
		'%Change' : '{:.2%}'})

	return df, start_img, end_img, diff_img, diff_bin_norm, hist_df

def transform(df):
	dfm = df.copy()
	dfm.index = pd.DatetimeIndex(dfm.Timestamp)
	dfm = dfm.resample('A', label='left').mean()
	dfm.reset_index(inplace=True)
	return dfm

def main():
	st.text('Add option to download timelapse, publish maps')
	st.image(r'./assets/header.jpg')
	navigation = st.sidebar.selectbox('Navigation', ['Assessment', 'Manual', 'Generate Report'])
	if navigation == 'Assessment':
		st.title('Vegetation Health Assessment App')
		st.markdown(f"""
			The web app aims to provide helpful information that can aid in efforts to protect our forest
			ecosystem. It uses the [Landsat 8 Collection 1 Tier 1](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_TOA) 
			8-Day Normalized Difference Vegetation Index (NDVI) composite at 30-meter resolution to provide 
			an overview of vegetation health of an area from 2013 - present.

			[NDVI](https://gisgeography.com/ndvi-normalized-difference-vegetation-index/) is an indicator used to quantify vegetation health based on how the vegetation respond to 
			light at the Near Infrared (NIR) and Red bands. NDVI ranges from -1.0 to +1.0. 

			NDVI *less than 0.2 are classified as Non-vegetation* (barren lands, build-up area, road network), *between 0.2 to 0.5 (inclusive) as Low Vegetation* 
			(shrubs and grassland ) and *greater than 0.5 to 1.0 as Dense vegetation* (temperate and tropical urban forest).

			You can either **upload a file** (*.shp format) or **draw a bounding box** of the region of interest
			and input the coordinates (i.e, GeoJSON format) to obtain the historical NDVI values for that region.
			""")

		with st.beta_expander('Click to draw Region of Interest (ROI)'):
			st.markdown(f"""
				You can select the region of interest (ROI) here. \n
				Click OK to accept cookies. Select ROI by utilizing the toolbars located at
				the upper left of the map or using the search box.
				
				**Important!** At the lower left corner, select the **GEOJSON** format and copy
				the coordinates.
				""")
			components.iframe('https://boundingbox.klokantech.com/', height=500)

		inputRegion = st.text_input(f'Paste ROI coordinates here and press Enter:')

		Map = geemap.Map()

		st.sidebar.subheader('Customization Panel')
		inputFile = st.sidebar.file_uploader('Upload a file', type=['shp'])
		# image_collection = st.sidebar.selectbox('Select Image Collection', ['MODIS', 'Landsat', 'Sentinel'])
		# band = st.sidebar.selectbox('Select band', ['NDVI', 'EVI'])
		# export_csv = st.sidebar.button('Export report (not yet working)')
		# https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/gallery/file_download/file_download.py
		
		try:
			if inputFile is None and len(inputRegion)==0:
				region = geemap.shp_to_geojson(r'./assets/butuan_city_gcs.shp')
				data = region['features'][0]['geometry']['coordinates']

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

		st.markdown(f"""
			Area of ROI is *{aoi.geometry().area().getInfo()/10000:,.02f}* has. The centroid is 
			located at *({lat:.02f} N, {lon:.02f} E)*.
			""")

		startdate, enddate = st.select_slider('Use the slider to select the start and end date', 
			df.Timestamp.unique().tolist(), 
			value=[df.Timestamp.unique().tolist()[0], df.Timestamp.unique().tolist()[-1]])

		df = df[(df.Timestamp >= startdate) & (df.Timestamp <= enddate)]
		
		report_df, start_img, end_img, diff_img, diff_bin_norm, hist_df = read_img(startdate, enddate, aoi)
		
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
		'opacity':.7
		}

		# Create a folium map object.
		my_map = folium.Map(location=[lat, lon], zoom_start=11)

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

			Map.add_landsat_ts_gif(
				layer_name="Timelapse",
				roi=aoi.geometry(),
				# label='Timelapse',
				start_year=startdate.year,
				end_year=enddate.year,
				start_date=s1[5:],
				end_date=d1[5:],
				bands=["Red", "Green", 'Blue'],
				vis_params=None,
				dimensions=650,
				frames_per_second=1,
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

			st.markdown(f"""<img src="data:image/gif;base64,{data_url}" alt="timelapse gif">""",
				unsafe_allow_html=True,
)
					
		# create an interval selection over an x-axis encoding
		brush = alt.selection_interval(encodings=['x'])

		# determine opacity based on brush
		opacity = alt.condition(brush, alt.value(0.9), alt.value(0.1))

		highlightA = alt.selection(
		type='single', on='mouseover', fields=['Year'], nearest=True)

		annual = alt.Chart(df_annual, title='Mean NDVI (Annual)').properties(height=50, width=620).mark_circle(size=50).encode(
			x=alt.X('Timestamp:T', axis=alt.Axis(labels=False, tickOpacity=0), title=None),
			y=alt.Y('NDVI:Q', title=None, scale=alt.Scale(domain=[df_annual.NDVI.min(), df_annual.NDVI.max()])),
			opacity=opacity,
			color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None),
			tooltip=[
				alt.Tooltip('NDVI:Q', title='Annual Mean', format=',.4f')
			]).add_selection(brush)

		baseA = alt.Chart(df, title=f'Mean NDVI over the region for imagery at every available date').properties(height=200, width=620).encode(
			x=alt.X('Timestamp:T', title=None, scale=alt.Scale(domain=brush)),
			y=alt.Y('NDVI:Q', scale=alt.Scale(domain=[df.NDVI.min(), df.NDVI.max()])),
			opacity=opacity)

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

		baseC = alt.Chart(df, title='Trend').properties(height=200, width=620).encode(
			x=alt.X('Timestamp:T', title='Date', scale=alt.Scale(domain=brush)),
			y=alt.Y('NDVI_Lowess:Q', title='Smoothed NDVI', scale=alt.Scale(domain=[df.NDVI_Lowess.min(), df.NDVI_Lowess.max()]))
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


		altA = (annual & (pointsA + linesA + rule) & (pointsC + linesC + regC))

		baseB = alt.Chart(df, title='NDVI values per Day of Year (DOY) with IQR band').encode(
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

		altC = alt.Chart(hist_df, title='Histogram of NDVI values for images of selected dates').transform_fold([f'{startdate}', f'{enddate}'], as_=['Dates', 'NDVI']
			).mark_area( opacity=0.3, interpolate='step').encode(
				x=alt.X('NDVI:Q', bin=alt.Bin(maxbins=100)),
				y=alt.Y('count()', stack=None),
				color=alt.Color('Dates:N', legend=None),
				tooltip=[alt.Tooltip('Dates:N', title='Date'),
						alt.Tooltip('NDVI:Q', bin=alt.Bin(maxbins=100)),
						alt.Tooltip('count()', title='Count')]
			)

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
					\nSelected dates: **{startdate} to {enddate}**   
					Number of days betweet selected dates: **{(enddate - startdate).days:,} days**    
					Number of datapoints between the selected dates: **{df.shape[0]}**   
					""")

		st.write('Area (in has) Percent Change between Selected Dates across NDVI classes')

		st.dataframe(report_df)

		st.subheader('Exploratory Visualization')
		st.altair_chart(altC, use_container_width=True)

		st.write('Use the Annual Mean NDVI chart to zoom on the next two (2) charts.')
		
		st.altair_chart(altA, use_container_width=True)
		st.altair_chart(altB, use_container_width=True)

		with st.sidebar.beta_expander('Discussion board. Click to expand.'):
			components.iframe('https://padlet.com/kaquisado/v9y0rhx2lrf0tilk', height=500)

	elif navigation == 'Manual':
		st.title('Manual')

	else:
		st.title('Generate Report')
		
if __name__ == '__main__':
	main()