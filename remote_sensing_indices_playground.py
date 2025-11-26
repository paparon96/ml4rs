import rasterio
import numpy as np
import matplotlib.pyplot as plt

def load_bands(paths):
    return [rasterio.open(p).read(1) for p in paths]

data_folder_name = "Sopronbanfalva"

paths = [f'./data/{data_folder_name}/B04.jpg',  # Red band
         f'./data/{data_folder_name}/B03.jpg',  # Green band
         f'./data/{data_folder_name}/B02.jpg',  # Blue band
         # f'./data/{data_folder_name}/B08.jpg',  # Near-infrared band
         # f'./data/{data_folder_name}/B11.jpg'  # Short-wave infrared band 
         ]

red, green, blue = load_bands(paths)
rgb = np.dstack([red, green, blue])

plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("RGB Image (Sentinel-2)")
plt.axis('off')
plt.show()