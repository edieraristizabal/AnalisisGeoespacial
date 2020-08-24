import ee
from ee_plugin import Map

AMVA = ee.FeatureCollection("G:\My Drive\INVESTIGACION\Cartografia/AMVA");
SRTM = ee.Image("USGS/SRTMGL1_003");

AMVA = ee.FeatureCollection("users/evaristizabalg/AMVA");
SRTM_AMVA=SRTM.clip(AMVA);
hillshade=ee.Terrain.hillshade(SRTM_AMVA)
Map.addLayer(hillshade,{},'AMVA');
Map.setCenter(-75.5627, 6.2288,10);
