<a href="https://vegamap.herokuapp.com/">ndvi-app</a>

<p align="justify">A web app that aims to provide information that can 
complement the efforts of our decision-makers in monitoring the overall vegetation health of an area, especially our 
forest ecosystems. The app uses freely available remotely sensed derived information from satellite images from 2013 - present.</p>

<ul>
<li><strong>Indicator: <a href="https://gisgeography.com/ndvi-normalized-difference-vegetation-index/">Normalized Difference Vegetation Index (NDVI)</a></strong></li>
<p style="margin-left: 30px" align="justify">NDVI is a widely used indicator to quantify vegetation health based on how the vegetation responds to light at the Near 
Infrared (NIR) and Red bands.</p>
</ul>

<p style="margin-left: 30px" align="justify">NDVI, which ranges in value from -1.0 to 1.0, is computed using the equation below, where NDVI
is equal to the ratio of the difference and sum of the Near-Infrared (NIR) (reflectance at 0.86 <em>u</em>m) and Red bands (reflectance at 0.65 <em>u</em>m).</p>

<p style="margin-left: 30px" align="justify">The figure below provides a visual interpretation of NDVI values for healthy and unhealthy vegetations.</p>

<p style="margin-left: 30px" align="justify">The NDVI values were classified into six (6) distinct classes:
<ul style="margin-left: 30px">
<li style="list-style-type:square">Bare Soil and/or Water: -1 to 0 (exclusive)</li>
<li style="list-style-type:square">Very Low Vegetation: 0 to 0.2 (exclusive)</li>
<li style="list-style-type:square">Low Vegetation: 0.2 to 0.4 (exclusive)</li>
<li style="list-style-type:square">Moderately Low Vegetation: 0.4 to 0.6 (exclusive)</li>
<li style="list-style-type:square">Moderately High Vegetation: 0.6 to 0.8 (exclusive)</li>
<li style="list-style-type:square">High Vegetation: 0.8 to 1.0 (inclusive)</li></ul></p>

<ul>
<li><strong>Data: <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_TOA">USGS Landsat 8 Collection 1 Tier 1 TOA Reflectance (30-meter resolution)</a></strong></li>
<p style="margin-left: 30px" align="justify">The primary data used to get NDVI values are derived from the Landsat 8 calibrated Top of Atmosphere reflectance.</p>
</ul>

<ul >
<li><strong>Methodology</strong></li>
<p style="margin-left: 30px" align="justify">The team utilized the capability of <a href="https://earthengine.google.com/">Google Earth Engine (GEE)</a> 
to provide access to volumes of satellite images without actually downloading data. In essence, the backend of this web app is GEE. The team implemented
a five (5) step process in creating the web app, listed below.</p>

<li style="list-style-type:square" align="justify"><em>Data acquistion</em> - the Landsat 8 NDVI Composites are accessed from the Google Earth Engine repository.</li>
<li style="list-style-type:square" align="justify"><em>Data filtering</em> - the satellite images are filtered spatially, and temporally based on user input. If no input is provided, 
the web app will show default area of interest (AOI), set in Butuan City from 2013 - present.</li>
<li style="list-style-type:square" align="justify"><em>Data Wrangling</em> - the raw images are aggregated, spatially and temporally, into mean NDVI, per available date.</li>
<li style="list-style-type:square" align="justify"><em>Data Analysis</em> - the difference between NDVI values between the earliest and latest available images are obtained. 
Then, the percent change across the NDVI classes is computed.</li>
<li style="list-style-type:square" align="justify"><em>Data Visualization</em> - visual expolatory data analysis is performed, providing various time-series visualization of mean NDVI for all available images.</li>
<li style="list-style-type:square" align="justify"><em>Interactive Web app</em> - a web app is developed to serve as platform for visualization and analysis.</li><br>

<br>
<h3>Resources:</h3>
<ul>
<li>Perez, G.J., and Comiso, J.C. (2014). Seasonal and Interannual Variability of Philippine Vegetation as Seen from Space, Philippine Journal of Science Vol. 143 No. 2 pp. 147-155 (ISI). Accessed <a = href="https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol143no2/pdf/Seasonal_and_interannual_variability_of_philippine_vegetation.pdf">here</a>.</li>
<li>Aquino, D., Rocha Neto, O., Moreira, M., Teixeira, A., & Andrade, E. (2018). Use of remote sensing to identify areas at risk of degradation in the semi-arid region. 49(3). 
doi: 10.5935/1806-6690.20180047. Accessed <a href="https://www.scielo.br/j/rca/a/JByZddTmJGRh67Fj8xQWtZL/?lang=en#">here</a>.</li>
<li>Barka, I., Bucha, T., Moln??r, T., M??ricz, N., Somogyi, Z., & Kore??, M. (2019). Suitability of MODIS-based NDVI index for forest monitoring and its seasonal applications in Central 
Europe. Central European Forestry Journal, 65(3-4), 206-217. doi: 10.2478/forj-2019-0020. Accessed <a href="https://sciendo.com/article/10.2478/forj-2019-0020">here</a>.</li>
<li>USGS Landsat 8 Collection 1 Tier 1 TOA Reflectance. (2021). Retrieved {now}. Accessed <a href=https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_TOA>here.</a></li>
<li>Time Series Visualization with Altair | Google Earth Engine. (2021). Retrieved {now}. Accessed <a href="https://developers.google.com/earth-engine/tutorials/community/time-series-visualization-with-altair">here</a>.</li>
<li>Bounding Box Tool. (2021). Retrieved {now}. Accesed <a href="https://boundingbox.klokantech.com/">here</a>.</li>
<li>Sparta Portal. (2021). Retrieved 13 October 2021. Accessed <a href="https://sparta.dap.edu.ph/opendata/lgu/butuancity/datasets">here</a>.</li>
<li>GADM. (2021). Retrieved 13 October 2021, Accessed <a href="https://gadm.org/download_country_v3.html">here</a>.</li>
</ul>
