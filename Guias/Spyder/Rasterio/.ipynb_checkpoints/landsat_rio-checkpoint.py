# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:37:30 2019

@author: Edier
"""
import numpy as np
import rasterio as rio
from rasterio.plot import show_hist
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.merge import merge

##############################################################################
#Para importar una banda
ruta='G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/LE70090532003066EDC00_B1.TIF'
raster = rio.open(ruta)
show(raster);
type(raster)
raster.dtypes
print(raster.width, raster.height)    #para saber numero de filas y columnas
print(raster.shape)            #para saber numero de filas y columnas
print(raster.crs)              #para saber el sistema de coordenadas
print(raster.transform)
print(raster.count)            #para saber cuantas bandas hay
print(raster.indexes)
print(raster.bounds) 

B1=raster.read(1)
type(B1)
plt.imshow(B1)
plt.colorbar();

show_hist(B1, bins=100, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram")

plt.boxplot(B1) #se demora mucho

vector=B1.ravel()
vector_rec=vector[vector!=0]
vector_rec.mean()

B1_nan=np.where(B1==0,np.nan,B1)
show(B1_nan)

check = np.logical_and ( B1_nan > 100, B1_nan < 120 )
B1_rango=np.where(check,1,np.nan)
show(B1_rango)

###############################################################################
#Para importar un mosaico
composite = rio.open('G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/Composite_LE70090532003066EDC00.tif')

print(composite.count)            #para saber cuantas bandas hay

show((composite, 4), cmap='Reds')
show((composite, 3), cmap='Greens')
show((composite, 2), cmap='Blues')

show_hist((composite,1), bins=100, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram Banda 1")
show_hist((composite,2), bins=100, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram Banda 2")
show_hist((composite,3), bins=100, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram banda 3")

##############################################################################
#Para hacer un mosaico
mosaic = []
B2 = rio.open('G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/LE70090532003066EDC00_B2.TIF')
B2.dtypes
mosaic.append(B2)
B3 = rio.open('G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/LE70090532003066EDC00_B3.TIF')
B3.dtypes
mosaic.append(B3)
B4 = rio.open('G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/LE70090532003066EDC00_B4.TIF')
B4.dtypes
mosaic.append(B4)

mosaic_vf, output_transform=merge(mosaic)
show(mosaic_vf)

###############################################################################
#Para calcular el NDVI
B4 = rio.open('G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/LE70090532003066EDC00_B4.TIF')
matriz_B4=B4.read(1)
B5 = rio.open('G:\My Drive\CATEDRA\SENSORES REMOTOS\TALLERES\Taller 2_LANDSAT\Imagen/barranquilla/LE70090532003066EDC00_B5.TIF')
matriz_B5=B5.read(1)
#Next we need to tweak the behaviour of numpy a little bit. By default numpy will #complain about dividing with zero values. 
#We need to change that behaviour because we have a lot of 0 values in our data.
np.seterr(divide='ignore', invalid='ignore')
#Now we need to initialize with zeros before we do the calculations 
NDVI = np.empty(B4.shape, dtype=rio.float32)
#First, we can create a filter where we calculate the values on such pixels that have a value larger than 0. siempre para dos condiciones
check = np.logical_or ( matriz_B4 > 0, matriz_B5 > 0 )
#aplicr formula
NDVI = np.where ( check,  (matriz_B5 - matriz_B4 ) / ( matriz_B5 + matriz_B4 ), -999 )
NDVI=np.where(NDVI==-999,np.nan,NDVI)
plt.imshow(NDVI)
plt.colorbar();

ndvi = np.zeros(B5.shape, dtype=rio.float32)
ndvi = np.divide(matriz_B5 - matriz_B4, matriz_B5 + matriz_B4, where=(matriz_B5 - matriz_B4)!=0)
ndvi[ndvi == 0] = np.nan
plt.imshow(ndvi)
plt.colorbar();

np.nanmean(NDVI)
show(NDVI, cmap='summer');

#Para exportar
meta=B4.profile
B4_transform = meta['transform']
B4_crs = meta['crs']

with rio.open('G:/My Drive/SENSORES REMOTOS/TALLERES/Taller 3_NDVI/Imagen/barranquilla/NDVI.TIF', 'w', 
              driver='Gtiff',height=matriz_B4.shape[0],width=matriz_B4.shape[1],count=1,
              dtype='float64',nodata=-999,crs=B4_crs,transform=B4_transform) as dst:
    dst.write(NDVI,1)