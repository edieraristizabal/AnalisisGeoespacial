
import ee
from ee_plugin import Map

l8_2019=ee.Image('LANDSAT/LC08/C01/T1_RT/LC08_009056_20190903');
Map.addLayer(l8_2019,{},'l8_2019');

ndvi2019=l8_2019.normalizedDifference(['B5','B4']);
Map.addLayer(ndvi2019,{},'NDVI_funcion');

Map.centerObject(ndvi2019,8)