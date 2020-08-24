# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:14:18 2019

@author: Edier Aristizabal
"""

import ee
import numpy as np
 
# Define the roi
area = ee.Geometry.Polygon([[-75.69327597674794,6.197637610738173],\
                            [-75.51861777362294,6.197637610738173],\
                            [-75.51861777362294,6.318123436693424],\
                            [-75.69327597674794,6.318123436693424],\
                            [-75.69327597674794,6.197637610738173]])
 
# define the image
collection = ee.ImageCollection("COPERNICUS/S2").filterBounds(area)\
                                      .filterDate("2018-01-01","2019-01-10")\
                                      .filterMetadata("CLOUDY_PIXEL_PERCENTAGE","less_than",10)\
                                      .select(['B8', 'B4'])
 
print(" number of images: ",collection.size().getInfo())
 
# perform any calculation on the image collection here
def anyFunction(img):
    ndvi = ee.Image(img.normalizedDifference(['B8', 'B4'])).rename(["ndvi"])
    return ndvi
 
# export the latitude, longitude and array
def LatLonImg(img):
    img = img.addBands(ee.Image.pixelLonLat())
 
    img = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                        geometry=area,\
                                        maxPixels=1e13,\
                                        scale=10);
 
    data = np.array((ee.Array(img.get("result")).getInfo()))
    lats = np.array((ee.Array(img.get("latitude")).getInfo()))
    lons = np.array((ee.Array(img.get("longitude")).getInfo()))
    return lats, lons, data
 
# covert the lat, lon and array into an image
def toImage(lats,lons,data):
 
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
    return arr
 
# map over the image collection
myCollection  = collection.map(anyFunction)
 
# get the median
result = ee.Image(myCollection.median()).rename(['result'])
 
# get the lon, lat and result as 1d array
lat, lon, data = LatLonImg(result)
 
# 1d to 2d array
image  = toImage(lat,lon,data)
 
# in case you want to plot the image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()