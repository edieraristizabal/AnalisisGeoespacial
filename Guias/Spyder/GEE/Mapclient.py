# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:54:47 2019

@author: Edier Aristizabal
"""

# Import the Earth Engine Python Packages
import ee
import ee.mapclient
 
# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()
 
# Print the information for an image asset.
image = ee.Image('srtm90_v4') 

# create the vizualization parameters
viz = {'min':0.0, 'max':4000, 'palette':"000000,0000FF,FDFF92,FF2700,FF00E7"};

# display the map
ee.mapclient.addToMap(image,viz, "mymap")