import rioxarray

raster = rioxarray.open_rasterio("data/crs_tif/sample.tif")
print("Original CRS:", raster.rio.crs)

reprojected = raster.rio.reproject("EPSG:32633")
print("Reprojected CRS:", reprojected.rio.crs)