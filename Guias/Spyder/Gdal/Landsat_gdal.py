# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:30:28 2020

@author: Edier Aristizabal
"""
#importar la librería de gdal
import gdal

#importar una imagen (banda)
data = gdal.Open(r'G:\My Drive\CATEDRA\SENSORES REMOTOS\Imagen\LE70090532003066EDC00_B1.tif')

#para saber el número de bandas
print(data.RasterCount)

#Para saber número de filas y columnas
print(data.RasterYSize)
print(data.RasterXSize)
print('Image size is: {r} rows x {c} columns\n'.format(r=data.RasterYSize, c=data.RasterXSize))

#para saber sistema de proyección
print(data.GetProjection())

#para obtener el gdal.band
banda = data.GetRasterBand(1)
# How about some band statistics?
band_max, band_min, band_mean, band_stddev = banda.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))

#para convertirlo en array
b1 = banda.ReadAsArray()

#para saber las dimensiones de un array
print(b1.shape)

#para graficar uan matriz se utiliza la funcion imshow
import matplotlib.pyplot as plt
plt.imshow(b1)
plt.colorbar()


#importar uan imagen compuesta
composite = gdal.Open(r'G:\My Drive\CATEDRA\SENSORES REMOTOS\Imagen\barranquilla\Composite_LE70090532003066EDC00.tif')

#para saber el número de bandas
print(composite.RasterCount)

from osgeo import gdal_array
import numpy as np

# Allocate our array using the first band's datatype
image_datatype = composite.GetRasterBand(1).DataType

# Allocate our array, but in a more efficient way
image = np.zeros((composite.RasterYSize, composite.RasterXSize, composite.RasterCount),
                 dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
         

# Ajustar el indice y pasar a capa image
for b in range(composite.RasterCount):
    # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
    bandas = composite.GetRasterBand(b + 1)
    
    # Read in the band's data into the third dimension of our array
    image[:, :, b] = bandas.ReadAsArray()

print(image)
print(image.shape)

#calcular el NDVI utilizando bandas
b2 = np.ndarray.flatten(image[:, :, 2])
b3 = np.ndarray.flatten(image[:, :, 3])

#plotear scatter
plt.scatter(b2, b3)

#con formato
plt.scatter(b2, b3, color='r', marker='o')
plt.xlabel('Red Reflectance')
plt.ylabel('NIR label')
plt.title('Red vs NIR')

# use "imshow" for an image -- nir in first subplot, red in second
plt.subplot(121)
plt.imshow(image[:, :, 3], cmap=plt.cm.Greys)
plt.colorbar()


######
# Now red band in the second subplot (indicated by last of the 3 numbers)
plt.subplot(122)
plt.imshow(image[:, :, 2], cmap=plt.cm.Greys)
plt.colorbar()


#calcular el NDVI utilizando la imagen compuesta
ndvi1 = ((image[:, :, 3] - image[:, :, 2]) / (image[:, :, 3] + image[:, :, 2])).astype(np.float64)
plt.imshow(ndvi1)
plt.colorbar()

#calcular el NDVI utilizando la imagen compuesta
b3= image[:, :, 3]
b2= image[:, :, 2]
suma= b3+b2
resta=b3-b2
ndvi2= (resta/suma).astype(np.float64)
plt.imshow(ndvi2)
plt.colorbar()

ndvi3= np.where(ndvi2>5,np.nan,ndvi2)
plt.imshow(ndvi3)
plt.colorbar()

# create new file
driver = gdal.GetDriverByName('GTiff')
file = driver.Create( 'G:/My Drive/ndvi.tif', composite.RasterXSize , composite.RasterYSize , 1)
file.GetRasterBand(1).WriteArray(ndvi3)

# spatial ref system
proj = composite.GetProjection()
georef = composite.GetGeoTransform()
file.SetProjection(proj)
file.SetGeoTransform(georef)
file.GetRasterBand(1).SetNoDataValue(-9999)
file.FlushCache()