# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:40:16 2019

@author: Edier Aristizabal
"""

import ee
import numpy as np
import gdal
from osgeo import osr
import time

# init the ee object
ee.Initialize()

# Define the area
area = ee.Geometry.Polygon([[[-75.69327597674794,6.097637610738173],[-75.41861777362294,6.097637610738173],[-75.41861777362294,6.318123436693424],[-75.69327597674794,6.318123436693424],[-75.69327597674794,6.097637610738173]]])
 
# define the image
img = ee.Image("COPERNICUS/S2/20190827T152649_20190827T152643_T18NVM")
 
# do any ee operation here
ndvi = ee.Image(img.normalizedDifference(['B8', 'B4']))
timedate = img.get('GENERATION_TIME').getInfo()

# get the lat lon and add the ndvi
latlon = ee.Image.pixelLonLat().addBands(ndvi)
 
# apply reducer to list
latlon = latlon.reduceRegion(
  reducer=ee.Reducer.toList(),
  geometry=area,
  maxPixels=1e8,
  scale=20);
  
# get data into three different arrays
data = np.array((ee.Array(latlon.get("nd")).getInfo()))
lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
 
# get the unique coordinates
uniqueLats = np.unique(lats)
uniqueLons = np.unique(lons)
 
# get number of columns and rows from coordinates
ncols = len(uniqueLons)    
nrows = len(uniqueLats)
 
# determine pixelsizes
ys = uniqueLats[1] - uniqueLats[0] 
xs = uniqueLons[1] - uniqueLons[0]
 
# create an array with dimensions of image
arr = np.zeros([nrows, ncols], np.float32) #-9999
 
# fill the array with values
counter =0
for y in range(0,len(arr),1):
    for x in range(0,len(arr[0]),1):
        if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
            counter+=1
            arr[len(uniqueLats)-1-y,x] = data[counter] # we start from lower left corner
 
# in case you want to plot the image
import matplotlib.pyplot as plt        
plt.imshow(arr)
plt.show()
 
# set the 
#SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
transform = (np.min(uniqueLons),xs,0,np.max(uniqueLats),0,-ys)
 
# set the coordinate system
target = osr.SpatialReference()
target.ImportFromEPSG(4326)
 
# set driver
driver = gdal.GetDriverByName('GTiff')
 
timestring = time.strftime("%Y%m%d_%H%M%S")
outputDataset = driver.Create("G:\My Drive\ANALISIS GEOESPACIAL\Talleres en Python/output.tif", ncols,nrows, 1,gdal.GDT_Float32)
 
# add some metadata
outputDataset.SetMetadata( {'time': str(timedate), 'someotherInfo': 'lala'} )
 
outputDataset.SetGeoTransform(transform)
outputDataset.SetProjection(target.ExportToWkt())
outputDataset.GetRasterBand(1).WriteArray(arr)
outputDataset.GetRasterBand(1).SetNoDataValue(-9999)
outputDataset = None
