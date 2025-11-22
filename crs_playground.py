import rioxarray
import matplotlib.pyplot as plt

# new_crs = "EPSG:32633" # full black image after reprojection
new_crs = "EPSG:4326" # geographic projection, useful for global datasets, GPS and web maps
new_crs = "EPSG:32631" # original


raster = rioxarray.open_rasterio("data/crs_tif/sample.tif") # Sample file downloaded from here: https://github.com/mommermi/geotiff_sample/blob/master/sample.tif
print("Original CRS:", raster.rio.crs)

reprojected = raster.rio.reproject(new_crs)
print("Reprojected CRS:", reprojected.rio.crs)

# visualize the original and the reprojected image

original_band = raster.isel(band=0)
reprojected_band = reprojected.isel(band=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(original_band, cmap="gray")
ax1.set_title("Original")
ax1.axis("off")

ax2.imshow(reprojected_band, cmap="gray")
ax2.set_title("Reprojected")
ax2.axis("off")

plt.tight_layout()
plt.show()