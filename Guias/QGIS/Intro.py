layer=iface.activeLayer() # cuadno esta cargada
layer = QgsVectorLayer(r"G:\My Drive\INVESTIGACION\Cartografia/Antioquia","Antioquia") # cuadno no está cargada
rlayer = QgsRasterLayer(r"G:\My Drive\INVESTIGACION\Cartografia/Antioquia/Colombia_hillshade.tif", "SRTM layer name")
iface.zoomToActiveLayer()
iface.showAttributeTable(layer)
QgsProject.instance().addMapLayer(layer) # para visualizarla
layer.featureCount()
features=list(layer.getFeatures())
print('Número de barrios:',len(features))
feature=features[5]
print(feature['ID_BARRIO'])

# para saber mapas cargados
layers = QgsProject.instance().mapLayers()
print(layers)