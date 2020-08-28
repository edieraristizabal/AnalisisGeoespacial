# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:41:30 2020

@author: Edier Aristizabal
"""

import rasterio as rio

!curl -O -J https://hello.planet.com/data/s/UG2TX98suVmmi9q/download

# This notebook explores a single 4 band (blue, green, red, NIR) PlanetScope scene in a UTM projection.
image_file = "20190321_174348_0f1a_3B_AnalyticMS.tif"

satdat = rio.open(image_file)

# satdat is our open dataset object
print(satdat)

# let's look at some basic information about this geoTIFF:

# dataset name
print(satdat.name)

# number of bands in this dataset
print(satdat.count)

# The dataset reports a band count.
print(satdat.count)

# And provides a sequence of band indexes.  These are one indexing, not zero indexing like Numpy arrays.
print(satdat.indexes)

# PlanetScope 4-band band order: BGRN

blue, green, red, nir = satdat.read()

# Or the slightly less efficient:
#    blue = satdat.read(1)
#    green = satdat.read(2)
#    red = satdat.read(3)
#    nir = satdat.read(4)

# Or read the entire dataset into a single 3D array:
#    data = satdat.read()


# using the blue band as an example, examine the width & height of the image (in pixels)

w = blue.shape[0]
h = blue.shape[1]

print("width: {w}, height: {h}".format(w=w, h=h))

# Minimum bounding box in projected units
print(satdat.bounds)

# Get dimensions, in map units (using the example GeoTIFF, that's meters)

width_in_projected_units = satdat.bounds.right - satdat.bounds.left
height_in_projected_units = satdat.bounds.top - satdat.bounds.bottom

print("Width: {}, Height: {}".format(width_in_projected_units, height_in_projected_units))

# Number of rows and columns.

print("Rows: {}, Columns: {}".format(satdat.height, satdat.width))

# This dataset's projection uses meters as distance units.  What are the dimensions of a single pixel in meters?

xres = (satdat.bounds.right - satdat.bounds.left) / satdat.width
yres = (satdat.bounds.top - satdat.bounds.bottom) / satdat.height

print(xres, yres)
print("Are the pixels square: {}".format(xres == yres))

# Get coordinate reference system
satdat.crs

# Convert pixel coordinates to world coordinates.

# Upper left pixel
row_min = 0
col_min = 0

# Lower right pixel.  Rows and columns are zero indexing.
row_max = satdat.height - 1
col_max = satdat.width - 1

# Transform coordinates with the dataset's affine transformation.
topleft = satdat.transform * (row_min, col_min)
botright = satdat.transform * (row_max, col_max)

print("Top left corner coordinates: {}".format(topleft))
print("Bottom right corner coordinates: {}".format(botright))

# All of the metadata required to create an image of the same dimensions, datatype, format, etc. is stored in
# the dataset's profile:

satdat.profile