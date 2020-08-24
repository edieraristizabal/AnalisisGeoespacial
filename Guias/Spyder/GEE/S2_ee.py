# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:06:26 2020

@author: Edier Aristizabal
"""

import ee
import ee.mapclient

ee.Initialize()

image = ee.ImageCollection('COPERNICUS/S2') \
  .filterDate('2017-01-01', '2017-01-02').median() \
  .divide(10000) \
  .select(
  ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B12'],
  ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B12']
  )

vis = {'bands': ['B12', 'B08', 'B04'], 'min': 0.05, 'max': 0.5}

# TEST: add feature to the Map
ee.mapclient.addToMap(image, vis, 'S2')
