import os
import ee
import json
import base64
import geemap
import imageio
import zipfile
from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw 
import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
from fpdf import FPDF
# from html2image import Html2Image
from datetime import timedelta, datetime

import streamlit.components.v1 as components
# from streamlit.uploaded_file_manager import UploadedFile
from streamlit_folium import folium_static

import statsmodels.api as sm

from statsmodels.nonparametric.smoothers_lowess import lowess
# from statsmodels.tsa.stattools import adfuller

import folium
from folium import plugins

st.set_page_config(page_title='Vega M', page_icon='🌳')

# remove 'Made with Streamlit' footer MainMenu {visibility: hidden;}
hide_streamlit_style = """
			<style>
			footer {visibility: hidden;}
			</style>
			"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

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
        # elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
        #     ee_object_new = ee_object.mosaic()
        #     map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
        #     folium.raster_layers.TileLayer(
        #     tiles = map_id_dict['tile_fetcher'].url_format,
        #     attr = 'Google Earth Engine',
        #     name = name,
        #     overlay = True,
        #     control = True
        #     ).add_to(self)
        # # display ee.Geometry()
        # elif isinstance(ee_object, ee.geometry.Geometry):    
        #     folium.GeoJson(
        #     data = ee_object.getInfo(),
        #     name = name,
        #     overlay = True,
        #     control = True
        # ).add_to(self)
        # # display ee.FeatureCollection()
        # elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
        #     ee_object_new = ee.Image().paint(ee_object, 0, 2)
        #     map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
        #     folium.raster_layers.TileLayer(
        #     tiles = map_id_dict['tile_fetcher'].url_format,
        #     attr = 'Google Earth Engine',
        #     name = name,
        #     overlay = True,
        #     control = True
        # ).add_to(self)
    
    except:
        print("Could not display {}".format(name))
    
# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_data(l8, startdate, enddate, aoi, datamask):
	start1 = startdate.strftime('%Y-%m-%d')
	start2 = (startdate+timedelta(1)).strftime('%Y-%m-%d')
	end1 = enddate.strftime('%Y-%m-%d')
	end2 = (enddate+timedelta(1)).strftime('%Y-%m-%d')

	area = aoi.geometry().area().getInfo()/10000
	scale = 100
	start_img = l8.select('NDVI').filterDate(start1, start2).first().unmask()
	end_img = l8.select('NDVI').filterDate(end1, end2).first().unmask()
	
	start_img = start_img.reproject(crs=ee.Projection('EPSG:3395'), scale=scale)
	end_img = end_img.reproject(crs=ee.Projection('EPSG:3395'), scale=scale)

	diff_img = end_img.subtract(start_img)

	classes = ['Soil/Water', 'Very Low', 'Low', 
			'Mod. Low', 'Mod. High', 'High']

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
		# perc_change[np.isnan(perc_change)] = 0
		perc_change[np.isinf(perc_change)] = np.nan

	interpretation = ['Increase' if i > 0 else 'Decrease' if  i < 0 else 'No change' if  i == 0 else 'Increase' for i in list(perc_change)]

	tabular_df = pd.DataFrame({f'Area ({startdate})': start_bin_norm*area, 
			f'Area ({enddate})': end_bin_norm*area,
			'%Change': perc_change*100, 'Interpretation': interpretation}, index=classes)

	out_dir = os.path.join(os.getcwd(), 'assets')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	out_stats = os.path.join(out_dir, 'ndvi_stats.csv')

	col = l8.select('NDVI').filterDate(start1, end1)
	geemap.zonal_statistics(col, aoi, out_stats, statistics_type='MEAN', scale=scale)

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
	df['Standard'] = (df.NDVI - df.NDVI.mean())/df.NDVI.std()
	df.reset_index(drop=True, inplace=True)

	return df, tabular_df, start_img.updateMask(datamask), end_img.updateMask(datamask), diff_img.updateMask(datamask), diff_bin_norm, hist_df

@st.cache()
def date_range(l8, aoi):
	l8 = l8.select('NDVI')
	scale=10000

	out_dir = os.path.join(os.getcwd(), 'assets')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	out_stats = os.path.join(out_dir, 'date_range.csv')

	geemap.zonal_statistics(l8, aoi, out_stats, statistics_type='MEAN', scale=scale)

	df = pd.read_csv(out_stats)
	df = pd.melt(df, id_vars=['system:index'])
	df.variable = df.variable.apply(lambda x: x[:8])
	df = df.rename(columns={'value':'NDVI', 'variable':'Timestamp', 'system:index':'Default'})

	df['Timestamp'] = pd.DatetimeIndex(df['Timestamp']).date

	df.dropna(axis=0, inplace=True)
	df.drop(columns=['Default'], inplace=True)

	return df

def transform(df):
	dfm = df.copy()
	dfm.index = pd.DatetimeIndex(dfm.Timestamp)
	dfm = dfm.resample('A', label='left').mean()
	dfm.reset_index(inplace=True)
	return dfm

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

# class PDF(FPDF):
#     def header(self):
#         # Logo
#         self.image(r'./assets/header.jpg', 10, 8, 33)
#         # Arial bold 15
#         self.set_font('Arial', 'B', 15)
#         # Move to the right
#         self.cell(80)
#         # Title
#         self.cell(30, 10, 'Vegetation Assessment and Monitoring Report', 0, 0, 'C')
#         # Line break
#         self.ln(20)

#     # Page footer
#     def footer(self):
#         # Position at 1.5 cm from bottom
#         self.set_y(-15)
#         # Arial italic 8
#         self.set_font('Arial', 'I', 8)
#         # Page number
#         self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def main():
	st.image(r'./assets/header.jpg')
	# st.sidebar.subheader('Navigation Panel')
	navigation = st.sidebar.selectbox('Navigation', ['Home', 'Manual'])
	if navigation == 'Home':
		st.title('Vegetation Assessment and Monitoring App')
		with st.beta_expander('About the app', expanded=True):
			st.markdown(f"""
				<p align="justify">Vegetation Assessment and Monitoring App (Vega Map) aims to provide helpful information that can aid in efforts to protect our forest
				ecosystem. It utilizes the <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_8DAY_NDVI">Landsat 8 Collection 1 Tier 1</a> 
				8-Day Normalized Difference Vegetation Index (NDVI) composite at 30-meter resolution to provide 
				an overview of vegetation health of an area from 2013 - present.</p>

				<p align="justify"><a href="https://gisgeography.com/ndvi-normalized-difference-vegetation-index/">NDVI</a> is an indicator used to quantify vegetation health based on how the vegetation responds
				light at the Near Infrared (NIR) and Red bands. NDVI ranges from -1.0 to +1.0.</p> 

				<p align="justify">The NDVI values are classified into six (6) distinct classes:
				<ul>
				<li>Bare Soil and/or Water: -1 to 0 (exclusive)</li>
				<li>Very Low Vegetation: 0 to 0.2 (exclusive)</li>
				<li>Low Vegetation: 0.2 to 0.4 (exclusive)</li>
				<li>Moderately Low Vegetation: 0.4 to 0.6 (exclusive)</li>
				<li>Moderately High Vegetation: 0.6 to 0.8 (exclusive)</li>
				<li>High Vegetation: 0.8 to 1.0 (inclusive)</li></ul></p>

				<p align="justify"><em><font color="#85221A">Note: The information presented in this app serves as a guide only. Ground validation should still be conducted to verify its accuracy.</font></em></p> 
				""", unsafe_allow_html=True)

		st.subheader('A. Data Collection')
		with st.beta_expander('', expanded=True):
			st.markdown(f"""
				<p align="justify">Select an Area of Interest (AOI) by either <strong>(a) uploading a local file</strong> (*.shp format) or <strong>(b) drawing a bounding box</strong>.</p> 
				""", unsafe_allow_html=True)	
			inputFile = st.file_uploader('a. Upload a file', type=['zip'],
									help='Currently, only compressed file (*.zip) of the SHP AOI is accepted. ')
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
		
			inputRegion = st.text_input(f'Paste AOI coordinates here and press Enter:', 
									help='Currently, only GeoJSON formatted AOI is accepted')

			Map = geemap.Map()

			default_region = '[[[125.4727148947,8.9012996164],\
								[125.5990576681,8.9012996164],\
								[125.5990576681,8.9828386907],\
								[125.4727148947,8.9828386907],\
								[125.4727148947,8.9012996164]]]'
				
			try:
				if inputFile is None and len(inputRegion)==0:
					data = json.loads(default_region)

				elif inputFile is not None and len(inputRegion) == 0:
					with zipfile.ZipFile(inputFile, 'r') as zip_ref:
						zip_ref.extractall(r'./assets/')
					# files = [file for file in os.listdir(r'./assets') if file.endswith('.shp')]
					region = geemap.shp_to_ee(os.path.join(os.path.expanduser("~"), 'assets', inputFile.name.split('.')[0]+'.shp'))
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
			l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_8DAY_NDVI')
			aoi = ee.FeatureCollection(ee.Geometry.Polygon(data))
			datamask = ee.Image('UMD/hansen/global_forest_change_2015').select('datamask').eq(1)
			lon, lat = aoi.geometry().centroid().getInfo()['coordinates']
			
			st.markdown(f"""
				Area of AOI is **{aoi.geometry().area().getInfo()/10000:,.02f} has**. The centroid is 
				located at **({lat:.02f} N, {lon:.02f} E)**.
				""")
		
		with st.beta_container():		
			# st.markdown('---')
			st.subheader('B. Map Visualization')
			st.markdown(f"""
			1. Use the slider widget to select the DOI.<br>
			2. Wait for the app to load the NDVI composite images available for the AOI and DOI.
			3. Explore the layers.
			4. Generate timelapse of Annual Landsat composites (Convert to NDVI)
			""", unsafe_allow_html=True)

			date_list = date_range(l8, aoi)

			startdate, enddate = st.select_slider('DOI Slider', 
				date_list.Timestamp.unique().tolist(), 
				value=[date_list.Timestamp.unique().tolist()[0], date_list.Timestamp.unique().tolist()[-1]],
				help="Use the slider to select the DOI's (start and end date)")

			startdate_format = startdate.strftime('%B %d, %Y')
			enddate_format = enddate.strftime('%B %d, %Y')

			# df = df[(df.Timestamp >= startdate) & (df.Timestamp <= enddate)]

			df, report_df, start_img, end_img, diff_img, diff_bin_norm, hist_df = read_data(l8, startdate, enddate, aoi, datamask)
			df['Timestamp'] = pd.to_datetime(df.Timestamp)
			df_annual = transform(df)
							
			visParams = {
				# 'bands': ['NDVI'],
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
				# 'bands': ['NDVI'],
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
			# basemaps['Google Maps'].add_to(my_map)
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
			
			# st.markdown('<br>', unsafe_allow_html=True)	
			with st.beta_expander('Timelapse of Annual NDVI Composite Images'):
				timelapse = st.checkbox('Check to generate the animation.')
				
				if timelapse:
					out_dir = os.path.join(os.getcwd(), 'assets')

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
					
					l8 = l8.map(add_doy)
					l8 = l8.map(add_year)

					filenames = []
					images = []
					region = aoi.geometry().bounds()

					for i in range(startdate.year, enddate.year + 1):
						
						fcol = l8.filterMetadata('year', 'equals', i).reduce(ee.Reducer.median()).clip(aoi).updateMask(datamask)
						geemap.get_image_thumbnail(fcol, f'./assets/landsat_{i}.png', visParams, region=region, dimensions=500, format='png')
						img = Image.open(f'./assets/landsat_{i}.png')
						draw = ImageDraw.Draw(img)
						draw.text((0,0), f'Year {i}')
						img.save(f'./assets/landsat_{i}.png')
						filenames.append(f'./assets/landsat_{i}.png')

					for filename in filenames:
						images.append(imageio.imread(filename))
					
					imageio.mimsave('./assets/landsat_ndvi_ts.gif', images, fps=1)
					# s1 = startdate.strftime('%Y-%m-%d')
					# d1 = enddate.strftime('%Y-%m-%d')

					# band_options = {'Color Infrared (Vegetation)': ['NIR', 'Red', 'Green'],
					# 			'False Color': ['SWIR2', 'SWIR1', 'Red'],
					# 			'Natural Color': ["Red", "Green", 'Blue'],
					# 			'Agriculture': ['SWIR1', 'NIR', 'Blue'],
					# 			'Healthy Vegetation': ['NIR', 'SWIR1', 'Blue'],
					# 			'Land/Water': ['NIR', 'SWIR1', 'Red'],
					# 			'Natural with Atmospheric Removal': ['SWIR2', 'NIR', 'Green'],
					# 			'Shortwave Infrared': ['SWIR2', 'NIR', 'Green'],
					# 			'Vegetation Analysis': ['SWIR1', 'NIR', 'Red']}

					# ts_bands = st.selectbox('Select bands to visualize', list(band_options.keys()),
					# 						help="""Color Infrared (Vegetation), ['NIR', 'Red', 'Green']\
					# 								False Color, ['SWIR2', 'SWIR1', 'Red']\
					# 								Natural Color, ['Red', 'Green', 'Blue']\
					# 								Agriculture, ['SWIR1', 'NIR', 'Blue']\
					# 								Healthy Vegetation, ['NIR', 'SWIR1', 'Blue']\
					# 								Land/Water, ['NIR', 'SWIR1', 'Red']\
					# 								Natural with Atmospheric Removal, ['SWIR2', 'NIR', 'Green']\
					# 								Shortwave Infrared, ['SWIR2', 'NIR', 'Green']\
					# 								Vegetation Analysis, ['SWIR1', 'NIR', 'Red']
					# 						""")

					# Map.add_landsat_ts_gif(
					# 	layer_name="Timelapse",
					# 	roi=aoi.geometry(),
					# 	# label='Timelapse',
					# 	start_year=startdate.year,
					# 	end_year=enddate.year,
					# 	start_date=s1[5:],
					# 	end_date=d1[5:],
					# 	bands=band_options[ts_bands],
					# 	vis_params=None,
					# 	dimensions=650,
					# 	frames_per_second=.5,
					# 	font_size=30,
					# 	font_color="white",
					# 	add_progress_bar=True,
					# 	progress_bar_color="cyan",
					# 	progress_bar_height=5,
					# 	out_gif=out_gif,
					# 	download=True)

					file_ = open('./assets/landsat_ndvi_ts.gif', "rb")
					contents = file_.read()
					data_url = base64.b64encode(contents).decode("utf-8")
					file_.close()
					
					st.markdown(f"""<center><img src="data:image/gif;base64,{data_url}" alt="timelapse gif"></center>""",
						unsafe_allow_html=True)

				st.markdown('<br>', unsafe_allow_html=True)			

		st.markdown('---')
		st.subheader('C. Data Analytics')
		highlightA = alt.selection(
		type='single', on='mouseover', fields=['Year'], nearest=True)

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
			size=alt.condition(~highlightA, alt.value(1), alt.value(3)),
			color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None)
			)

		regC = baseC.transform_regression('Timestamp', 'NDVI_Lowess').mark_line(color="#C32622").encode(
			size=alt.condition(~highlightA, alt.value(1), alt.value(3)))

		altAA = (pointsC + linesC + regC).interactive(bind_y=False)

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
		st.info(f"""Overall, mean NDVI for the *selected AOI* and *dates* is **{mean_ndvi:0.3f}** and is **trending {trending}!** 📈 and \
					the areas where a **positive NDVI change** is observed is at **{positive_change:0.2%}**!
					\nSelected dates: **{startdate_format} - {enddate_format}**   
					Number of days between selected dates: **{(enddate - startdate).days:,} days**    
					Number of images available between the selected dates: **{df.shape[0]}**   
					""")

		decrease_list = list(report_df[report_df.Interpretation.isin(['Decrease'])].index)
		increase_list = list(report_df[report_df.Interpretation.isin(['Increase'])].index)

		st.markdown('Table 1. Area (in hectares) and Percent Change between Selected Dates across NDVI classes')

		st.dataframe(report_df)
		st.markdown(f"""
			<p align="justify">A decrease in the first four (4) NDVI classes (i.e., -1 - 0.6) or an increase in the remaining NDVI classes
			(i.e., 0.6 above) is interpreted as positive and vice versa. </p>
			
			<p align="justify">Given this and inspecting the table above, we observed <font color="#2D8532"><strong>increase</strong></font> in the following NDVI landcover classes: 
			<font color="#2D8532"><strong>{', '.join(increase_list)}</strong></font> while we observed a <font color="#A42F25"><strong>decrease</strong></font> in the following classes: 
			<font color="#A42F25"><strong>{', '.join(decrease_list)}</strong></font>.</p>
			""", unsafe_allow_html=True)
		st.markdown('---')
		st.subheader('Exploratory Analysis')
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

		option1, option2 = st.beta_columns((2))
		rule_option = option1.selectbox(label='Select Line Aggregation', 
										options=['Mean', 'Median', 'Maximum', 'Minimum'], 
										help='Defines the location of the horizontal red line along the y-axis')
		band_option = option2.selectbox(label='Select Error Bar Extent', 
							options=['Inter-quartile Range', 'Confidence Interval', 
							'Standard Error', 'Standard Deviation'],
							help='Defines the extent of the colored region')
		standardized = st.checkbox('Click to standardized')
		rule_option_dict = {'Mean': 'mean', 'Median': 'median', 'Maximum': 'max', 'Minimum': 'min'}
		band_option_dict = {'Inter-quartile Range': 'iqr', 'Confidence Interval': 'ci', 
			'Standard Error': 'stderr', 'Standard Deviation': 'stdev'}

		annual = alt.Chart(df_annual).mark_circle(size=50).encode(
			x=alt.X('Timestamp:T'),
			y=alt.Y('NDVI:Q'),
			# opacity=opacity,
			color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None),
			tooltip=[
				alt.Tooltip('NDVI:Q', title='Annual Mean', format=',.4f')
			])

		if standardized:	
			baseA = alt.Chart(df).encode(
				x=alt.X('Timestamp:T', title=None),
				y=alt.Y('Standard:Q')
				)

			linesA = baseA.mark_bar().encode(
				color=alt.Color('Standard:Q', scale=alt.Scale(scheme='redblue'), legend=None))
			
			rule = alt.Chart(df).mark_rule(color='red').encode(
				y=alt.Y('mean(Standard):Q'),
				tooltip=[alt.Tooltip(f'mean(Standard):Q', 
				title=f'{rule_option} NDVI Line', format=',.4f')])

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
				color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis'), legend=None))

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
			<p align="justify">Upon ticking the checkbox, the plot now shows the standardized values of tne NDVI time series setting the mean of the series to zero with a
			standard deviation equal to 1.</p>
			""", unsafe_allow_html=True)
		
		# adf = adfuller(df['NDVI'])
		# 'Since the critical value is less than all of the t-statistics, we can reject the null hypothesis. The series is stationary. We can perform forecasting'
		# if adf[0] <= adf[4]['1%'] and adf[0] <= adf[4]['5%'] and adf[0] <= adf[4]['10%']:
		# 	hypothesis = ['', '']
		# else:
		# 	hypothesis = ['not', 'non-']
			
		# hypothesis = f'Since the critical value is {hypothesis[0]} less than all of the t-statistics, we can{hypothesis[0]} reject the null hypothesis.\
		# 	The series is {hypothesis[1]}stationary. We can{hypothesis[0]} perform forecasting.'
		
		# with st.beta_expander('Check if we can forecast', expanded=True):
		# 	st.markdown(f"""<p align="justify"><strong>Test for stationarity using the Augmented Dickey Fuller Test</strong>
		# 	<ul>
		# 	<li>Critical value: {adf[0]:.2e}</li>
		# 	<li>p-value: {adf[1]:.2e}</li>
		# 	<li>t-statistics: {adf[4]['1%']:.2e} (1%), {adf[4]['5%']:.2e} (5%), {adf[4]['10%']:.2e} (10%)</li>
		# 	</ul>
		# 	{hypothesis}</p>
		# 		""", unsafe_allow_html=True)

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
			<p align="justify">Figure 4 shows the median of mean NDVI, represented by the <font color="#5378A9">blue line</font> and the corresponding variation per day (i.e., Inter-quartile range)
			across a year, represented by the <font color="#BFC9D5">light-blue band</font>.</p>
			
			<p align="justify">The maximum NDVI (i.e., <strong>{doy_df.Median.max():.2f}</strong>) is measured on the <strong>{max_day}{max_str} day</strong>, while the minimum
			NDVI (i.e., <strong>{doy_df.Median.min():.2f}</strong>) is measured on the <strong>{min_day}{min_str} day</strong> of the year.</p>

			<p align="justify">The largest variation in NDVI values is observed on the <strong>{var_max_day}{var_max_str} day </strong>of the year, while the 
			smallest variation is observed on the <strong>{var_min_day}{var_min_str} day</strong>.</p>
			""", unsafe_allow_html=True)

		st.markdown('---')
		with st.beta_expander('Discussion board.', expanded=True):
			components.iframe('https://padlet.com/kaquisado/v9y0rhx2lrf0tilk', height=500)

		# export_as_pdf = st.sidebar.button("Generate Summary Report")

		# if export_as_pdf:
			# pdf = PDF()
			# hti = Html2Image()
			# pdf.alias_nb_pages()
			# pdf.add_page()
			# pdf.set_font('Times', '', 12)
			# pdf.cell(0, 10, f'Report Generated on {datetime.now().date()}', 0, 1)
			
			# # save visualizations as png
			# altA.save('chart1.png')
			# altAA.save('chart1a.png')
			# (lineB + bandB).save('chart2.png')
			# altC.save('chart3.png')
			
			# export_map = folium.Map(location=[lat, lon], zoom_start=14)
			# basemaps['Google Satellite Hybrid'].add_to(export_map)
			# # export_map.add_ee_layer(start_img.clip(aoi.geometry()), visParams, f'{startdate}_IMG')
			# # export_map.add_ee_layer(end_img.clip(aoi.geometry()), visParams, f'{enddate}_IMG')
			# export_map.add_ee_layer(diff_img.clip(aoi.geometry()), visParams_diff, '')
			# plugins.MiniMap().add_to(export_map)
			# export_map.save('map.html')

			# hti.screenshot(html_file='map.html', save_as='map.png')
			# pdf.image('map.png', w=190, h=100)
			# pdf.ln(5)
			# headers = list(export_df.columns)
			# w, h = 37, 10
			# for header in headers:
			# 	if header is not headers[-1]:
			# 		pdf.cell(w=w, h=h, txt=header, border=1, ln=0, align='C')
			# 	else:
			# 		pdf.cell(w=w, h=h, txt=header, border=1, ln=1, align='C')
				
			# for row in range(export_df.shape[0]):
			# 	for col in headers:
			# 		if col is not headers[-1] and col is not headers[-2]:
			# 			try:
			# 				text = f'{export_df.loc[row, col]:,.2f}'
			# 			except:
			# 				text = str(export_df.loc[row, col])
			# 			pdf.cell(w=w, h=h, txt=text, border=1, ln=0, align='C')
			# 		elif col is headers[-2]:
			# 			text = f'{export_df.loc[row, col]:.2f}'
			# 			pdf.cell(w=w, h=h, txt=text, border=1, ln=0, align='C')
			# 		else:
			# 			pdf.cell(w=w, h=h, txt=export_df.loc[row, col], border=1, ln=1, align='C')

			# pdf.ln(5)
			# pdf.image('chart1.png', w=190, h=100)
			# pdf.ln(5)
			# pdf.image('chart1a.png', w=190, h=100)
			# pdf.ln(5)
			# pdf.image('chart2.png', w=190, h=80)
			# pdf.ln(5)
			# pdf.image('chart3.png', w=190, h=80)
			# pdf.ln(5)
			
			# html = create_download_link(pdf.output(dest="S").encode("latin-1"), f"Green Report")

			# st.sidebar.markdown(html, unsafe_allow_html=True)

	elif navigation == 'Forest Cover':
		st.title('Forest Cover Assessment App')

	else:
		st.title('Generate Report')
	
if __name__ == '__main__':
	main()