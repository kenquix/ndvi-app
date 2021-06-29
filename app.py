import os, ee, json, geemap
import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

from datetime import date, timedelta, datetime

import streamlit.components.v1 as components
from streamlit_folium import folium_static

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import folium
from folium import plugins
  
st.set_page_config(page_title='NDVI App', initial_sidebar_state='collapsed', page_icon='🌳') #, layout='wide')

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
    )
}

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

@st.cache(ttl=60*60*1, allow_output_mutation=True)
def read_data(aoi):
	modis = ee.ImageCollection('MODIS/006/MOD13Q1').select('NDVI')

	out_dir = os.path.join(os.getcwd(), 'assets')
	out_stats = os.path.join(out_dir, 'ndvi_stats.csv')

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	geemap.zonal_statistics(modis, aoi, out_stats, statistics_type='MEAN', scale=1000)

	df = pd.read_csv(out_stats)
	df = pd.melt(df, id_vars=['system:index'])
	df.variable = df.variable.apply(lambda x: x[:10]).str.replace('_', '-')
	df.value = df.value/10000

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

@st.cache(ttl=60*60*1, allow_output_mutation=True)
def read_img(startdate, enddate, aoi):
	start1 = startdate.strftime('%Y-%m-%d')
	start2 = (startdate+timedelta(1)).strftime('%Y-%m-%d')

	end1 = enddate.strftime('%Y-%m-%d')
	end2 = (enddate+timedelta(1)).strftime('%Y-%m-%d')

	area = aoi.geometry().area().getInfo()/10000
	
	start_img = ee.ImageCollection('MODIS/006/MOD13A2').select(
		'NDVI').filterDate(start1, start2).first().unmask()

	end_img = ee.ImageCollection('MODIS/006/MOD13A2').select(
		'NDVI').filterDate(end1, end2).first().unmask()

	diff_img = end_img.subtract(start_img)

	classes = ['Soil/Water [-1-0)', 'Very Low [0-0.2)', 'Low [0.2-0.4)', 
			'Mod. Low [0.4-0.6)', 'Mod. High [0.6-0.8)', 'High [0.8-1.0]']
	class_range = ['[-1 - 0)', '[0 - 0.2)', '[0.2 - 0.4)', '[0.4 - 0.6)', '[0.6 - 0.8)', '[0.8 - 1.0]']
	bins = [-1, 0 ,0.2 ,0.4 ,0.6 ,0.8 ,1.0]
	
	region = aoi.geometry()

	start_arr = geemap.ee_to_numpy(start_img, region=region)
	start_arr = start_arr/10000
	start_bin = np.histogram(start_arr, bins=bins)[0]
	start_bin_norm = start_bin/sum(start_bin)

	end_arr = geemap.ee_to_numpy(end_img, region=region)
	end_arr = end_arr/10000
	end_bin = np.histogram(end_arr, bins=bins)[0]
	end_bin_norm = end_bin/sum(end_bin)

	diff_arr = geemap.ee_to_numpy(diff_img, region=aoi.geometry())
	diff_bin = np.histogram(diff_arr, bins=[-9000, 0, 9000])[0]
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

	def color_cell(val):
	    color = ''
	    if val == 'No change ⟷':
	        color += '#FBDD4E'
	    elif val == 'Increase ↑':
	        color +='#389935'
	    elif val == 'Decrease ↓':
	        color += '#FF4106'
	    else:
	        color += ''
	    return 'background-color: %s' % color

	df = pd.DataFrame({f'Area ({startdate})': start_bin_norm*area, 
			f'Area ({enddate})': end_bin_norm*area,
			'%Change': perc_change, 'Interpretation': interpretation}, index=classes)

	df = df.style.format({f'Area ({startdate})': "{:,.2f}", f'Area ({enddate})': '{:,.2f}', 
		'%Change' : '{:.2%}'}).applymap(color_cell)

	return df, start_img, end_img, diff_img, diff_bin_norm, hist_df

def transform(df):
	dfm = df.copy()
	dfm.index = pd.DatetimeIndex(dfm.Timestamp)
	dfm = dfm.resample('A', label='left').mean()
	dfm.reset_index(inplace=True)
	return dfm

def main():
	st.image(r'./assets/header.jpg')
	st.sidebar.selectbox('Navigation', ['Assessment', 'Manual'])
	st.header('Vegetation Health Assessment App')
	st.markdown(f"""
		The web app aims to provide helpful information that can aid in efforts to protect our forest
		ecosystem. It uses the [Terra MODIS](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD13Q1) 
		16-Day Normalized Difference Vegetation Index (NDVI) composite at 250 meter resolution to provide 
		an overview of vegetation health of an area from 2000 - present.

		[NDVI](https://gisgeography.com/ndvi-normalized-difference-vegetation-index/) is an indicator used to quantify vegetation health based on how the vegetation respond to 
		light at the Near Infrared (NIR) and Red bands. NDVI values range from -1.0 to +1.0. 

		NDVI values *less than 0.2 are classified as Non-vegetation* (barren lands, build-up area, road network), *between 0.2 to 0.5 (inclusive) as Low Vegetation* 
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
	st.sidebar.selectbox('Select Image Collection', ['MODIS', 'Landsat', 'Sentinel'])
	band = st.sidebar.selectbox('Select band', ['NDVI', 'EVI'])
	export_csv = st.sidebar.button('Export report')
	
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
	  'max': 9000,
	    'palette': [
	    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
	    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
	    '012E01', '011D01', '011301'
	  ],
	  'opacity':.8
	}

	visParams_diff = {
	  'bands': ['NDVI'],
	  'min': -9000,
	  'max': 9000,
	    'palette': [
	    '#FF4C56', '#FFFFFF', '#73DA6E'
	  ],
	  'opacity':.8
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

	baseA = alt.Chart(df, title=f'Mean NDVI (16-Day)').properties(height=200, width=620).encode(
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
		# x=alt.X(),
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

	lower = df.groupby('DOY')['NDVI'].quantile(.25).min()
	upper = df.groupby('DOY')['NDVI'].quantile(.75).max()

	baseB = alt.Chart(df, title='NDVI values per Day of Year (DOY) with IQR band').encode(
	    x=alt.X('DOY:Q', scale=alt.Scale(domain=(0, 340)))) # )
	
	lineB = baseB.mark_line().encode(
	    y=alt.Y('median(NDVI):Q', 
	    	scale=alt.Scale(domain=[lower-0.1*lower,upper+0.1*lower])
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
	slope, intercept = results
	negative_change, positive_change = diff_bin_norm
	mean_ndvi = df.NDVI.mean()

	if slope > 0:
		trending = 'UP'
	else:
		trending =' DOWN'

	narrative = f"""Overall, mean NDVI for the *selected region* and *date* is **{mean_ndvi:0.3f}** and is **trending {trending}!** 📈 and \
				the area where a **positive NDVI change** is observed is at **{positive_change:0.2%}**!
				\nSelected dates: **{startdate} to {enddate}**   
				Number of days betweet selected dates: **{(enddate - startdate).days:,} days**    
				Number of datapoints between the selected dates: **{df.shape[0]}**   
				"""
				# Minimum NDVI: **{df.NDVI.min():.03f}** observed at {df.Timestamp.iloc[df.NDVI.argmin()]}   
				# Maximum NDVI: **{df.NDVI.max():.03f}** observed at {df.Timestamp.iloc[df.NDVI.argmax()]}
	
	if mean_ndvi > 0.6:
		if slope > 0 and positive_change > 0.8:
			st.info(f"""Green Rating: ★★★★★ out of 5
				\n**Excellent news!** {narrative}""")
		elif slope > 0 and positive_change > 0.6:
			st.info(f"""Green Rating: ★★★★✰ out of 5
				\n**Great news!** {narrative}""")
		else:
			st.info(f"""Green Rating: ★★★✰✰ out of 5
				\n**Good news.** {narrative}""")
	
	elif mean_ndvi > 0.4:
		if slope > 0 and positive_change > 0.7:
			st.info(f"""Green Rating: ★★★✰✰ out of 5
				\n**Good news.** {narrative}""")

		elif slope > 0 and positive_change >= 0.4:
			st.info(f"""Green Rating: ★★✰✰✰ out of 5
				\n**Not so good news.** {narrative}""")
		else:
			st.info(f"""Green Rating: ★✰✰✰✰ out of 5
				\n**NOT a good news!** {narrative}""")

	else:
		if slope > 0 and positive_change >= 0.6:
			st.info(f"""Green Rating: ★★✰✰✰ out of 5
				\n**Not so good news.** {narrative}""")
		else:
			st.info(f"""Green Rating: ★✰✰✰✰ out of 5
				\n**NOT a good news!** {narrative}""")

	st.subheader('Area (in has) Percent Change between Selected Dates across NDVI classes')

	st.dataframe(report_df)

	st.subheader('Exploratory Visualization')
	st.altair_chart(altC, use_container_width=True)

	st.write('Use the Annual Mean NDVI chart to zoom on the next two (2) charts.')
	
	st.altair_chart(altA, use_container_width=True)
	st.altair_chart(altB, use_container_width=True)

if __name__ == '__main__':
	main()