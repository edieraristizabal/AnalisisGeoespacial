# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:30:28 2020

@author: Edier Aristizabal
"""

# Import the Python 3 print function
from __future__ import print_function

#Import the "gdal" submodule from within the "osgeo" module
from osgeo import gdal

print("GDAL's version is: " + gdal.__version__)
print(gdal)

print(gdal.GDT_Byte)


# Open a GDAL dataset
ruta='G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/Composite_LE70090532003066EDC00.tif'

dataset = gdal.Open(ruta, gdal.GA_ReadOnly)

print(dataset)

# How many bands does this image have?
num_bands = dataset.RasterCount
print('Number of bands in image: {n}\n'.format(n=num_bands))

# How many rows and columns?
rows = dataset.RasterYSize
cols = dataset.RasterXSize
print('Image size is: {r} rows x {c} columns\n'.format(r=rows, c=cols))

# Does the raster have a description or metadata?
desc = dataset.GetDescription()
metadata = dataset.GetMetadata()

print('Raster description: {desc}'.format(desc=desc))
print('Raster metadata:')
print(metadata)
print('\n')

# What driver was used to open the raster?
driver = dataset.GetDriver()
print('Raster driver: {d}\n'.format(d=driver.ShortName))

# What is the raster's projection?
proj = dataset.GetProjection()
print('Image projection:')
print(proj + '\n')

# What is the raster's "geo-transform"
gt = dataset.GetGeoTransform()
print('Image geo-transform: {gt}\n'.format(gt=gt))


# Open the blue band in our image
blue = dataset.GetRasterBand(1)

print(blue)

# What is the band's datatype?
datatype = blue.DataType
print('Band datatype: {dt}'.format(dt=blue.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(blue.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(blue.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = blue.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))


# No alias
import numpy
print(numpy.__version__)

# Alias or rename to "np" -- a very common practice
import numpy as np
print(np.__version__)


help(blue.ReadAsArray)


blue_data = blue.ReadAsArray()

print(blue_data)
print()
print('Blue band mean is: {m}'.format(m=blue_data.mean()))
print('Size is: {sz}'.format(sz=blue_data.shape))

# Initialize a 3d array -- use the size properties of our image for portability!
image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))

# Loop over all bands in dataset
for b in range(dataset.RasterCount):
    # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
    band = dataset.GetRasterBand(b + 1)
    
    # Read in the band's data into the third dimension of our array
    image[:, :, b] = band.ReadAsArray()

print(image)
print(image.dtype)

dataset.GetRasterBand(1).DataType


from osgeo import gdal_array

# DataType is a property of the individual raster bands
image_datatype = dataset.GetRasterBand(1).DataType

# Allocate our array, but in a more efficient way
image_correct = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))

# Loop over all bands in dataset
for b in range(dataset.RasterCount):
    # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
    band = dataset.GetRasterBand(b + 1)
    
    # Read in the band's data into the third dimension of our array
    image_correct[:, :, b] = band.ReadAsArray()

print("Compare datatypes: ")
print("    when unspecified: {dt}".format(dt=image.dtype))
print("    when specified: {dt}".format(dt=image_correct.dtype))