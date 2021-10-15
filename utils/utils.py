import os
import ee
import base64
import geemap

import streamlit as st
import pandas as pd
import numpy as np

from datetime import timedelta, datetime
from streamlit_folium import folium_static

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import folium

import altair as alt

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

@st.cache()
def cloudlessNDVI(image):
    cloud = ee.Algorithms.Landsat.simpleCloudScore(image).select('cloud')
    mask = cloud.lte(20)
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    return image.addBands(ndvi).updateMask(mask)

def cloud_mask(image):
    cloud = ee.Algorithms.Landsat.simpleCloudScore(image).select('cloud')
    mask = cloud.lte(20)
    return mask
			
@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def read_data(l8, startdate, enddate, aoi, datamask, scale):
	start1 = startdate.strftime('%Y-%m-%d')
	start2 = (startdate+timedelta(1)).strftime('%Y-%m-%d')
	end1 = enddate.strftime('%Y-%m-%d')
	end2 = (enddate+timedelta(1)).strftime('%Y-%m-%d')

	area = aoi.geometry().area().getInfo()/10000
	scale = scale
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
		st.error('Too many pixels in sample; must be <= 262144. Please select a smaller region or decrease resolution (i.e. Increase the value of scale.).')
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
	df.variable = df.variable.apply(lambda x: x[12:20])
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

	start_img = cloudlessNDVI(l8.filterDate(start1, start2).first()).select('NDVI')
	end_img = cloudlessNDVI(l8.filterDate(end1, end2).first()).select('NDVI')

	start_mask = cloud_mask(l8.filterDate(start1, start2).first())
	end_mask = cloud_mask(l8.filterDate(end1, end2).first())

	diff_img = diff_img.updateMask(start_mask)
	diff_img = diff_img.updateMask(end_mask)
	
	return df, tabular_df, start_img.updateMask(datamask), end_img.updateMask(datamask), diff_img.updateMask(datamask), diff_bin_norm, hist_df

@st.cache(show_spinner=False)
def date_range(l8, aoi):
	l8 = l8.select('NDVI')

	out_dir = os.path.join(os.getcwd(), 'assets')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	out_stats = os.path.join(out_dir, 'date_range.csv')

	geemap.zonal_statistics(l8, aoi, out_stats, statistics_type='MEAN', scale=10000)

	df = pd.read_csv(out_stats)
	df = pd.melt(df, id_vars=['system:index'])
	# df.variable = df.variable.apply(lambda x: x[:8]) # for 8-day L8 NDVI composite
	df.variable = df.variable.apply(lambda x: x[12:20])
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

# def create_download_link(val, filename):
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def ruler(data, field, field_type, base_chart, text_value):
	nearest = alt.selection(
		type="single",
		nearest=True,
		on="mouseover",
		fields=[f"{field}"],
		empty="none",
	)

	selectors = (
		alt.Chart(data)
		.mark_point()
		.encode(x=f"{field}:{field_type}", opacity=alt.value(0))
		.add_selection(nearest)
	)

	rules = (
		alt.Chart(data)
		.mark_rule(color="#1E1E1E", opacity=0.2, strokeWidth=0.5)
		.encode(x=f"{field}:{field_type}")
		.transform_filter(nearest)
	)

	# Draw points on the line, and highlight based on selection
	points = base_chart.mark_point(color="#A3AF80").encode(
		opacity=alt.condition(nearest, alt.value(1), alt.value(0))
	)

	# Draw text labels near the points, and highlight based on selection
	text = base_chart.mark_text(align="left", dx=5, dy=-5).encode(
		text=alt.condition(nearest, f"{text_value}", alt.value(" "))
	)

	return nearest, selectors, rules, points, text