import ee
from ee_plugin import Map

# get a single feature
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
country = countries.filter(ee.Filter.eq('country_na', 'Colombia'))

# TEST: add feature to the Map
Map.addLayer(country, { 'color': 'orange' }, 'feature')

# set Map center using coordinates and zoom
Map.setCenter(-73.926, 4.657, 5)

