import ee
from ee_plugin import Map

collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')\
    .filter(ee.Filter.eq('WRS_PATH', 9))\
    .filter(ee.Filter.eq('WRS_ROW', 56))\
    .filterDate('2020-01-01', '2020-04-01')

median = collection.median()

Map.setCenter(-75.4, 6.5, 12)
Map.addLayer(median, {"bands": ['B4', 'B3', 'B2'], "max": 0.3}, 'median')