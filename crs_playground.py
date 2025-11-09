import rioxarray

raster = rioxarray.open_rasterio("data/satellite.tif")
print("Original CRS:", raster.rio.crs)

reprojected = raster.rio.reproject("EPSG:32633")
print("Reprojected CRS:", reprojected.rio.crs)