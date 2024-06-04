# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:39:55 2024

@author: ghazaryan
"""

import glob
import os
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def extract_doy(filename):
    match = re.search(r'_doy_(\d+)', filename)
    return int(match.group(1)) if match else None

def load_images_from_directory(directory, pattern):
    image_files = glob.glob(os.path.join(directory, pattern))
    image_data = {}
    for file in image_files:
        doy = extract_doy(file)
        with rasterio.open(file) as src:
            image_data[doy] = src.read(1), src.transform, src.crs
    return image_data

def resample_to_match(sentinel_data, sentinel_transform, sentinel_crs, planet_transform, planet_crs, planet_shape):
    resampled_sentinel = np.empty(planet_shape, dtype=np.float32)
    reproject(
        source=sentinel_data,
        destination=resampled_sentinel,
        src_transform=sentinel_transform,
        dst_transform=planet_transform,
        src_crs=sentinel_crs,
        dst_crs=planet_crs,
        resampling=Resampling.bilinear
    )
    return resampled_sentinel

# Load NDVI images from Sentinel-2 and Planet directories
sentinel_directory = 'S2_2022'
planet_directory = 'PS_2022'
sentinel_images = load_images_from_directory(sentinel_directory, '*_doy_*.tif')
planet_images = load_images_from_directory(planet_directory, '*_doy_*.tif')

# Pair images by closest DOY and resample
resampled_sentinel_images = {}
for p_doy, (planet_data, planet_transform, planet_crs) in planet_images.items():
    closest_doy = min(sentinel_images.keys(), key=lambda d: abs(d - p_doy))
    sentinel_data, sentinel_transform, sentinel_crs = sentinel_images[closest_doy]
    resampled_sentinel = resample_to_match(sentinel_data, sentinel_transform, sentinel_crs, planet_transform, planet_crs, planet_data.shape)
    resampled_sentinel_images[closest_doy] = resampled_sentinel

# Prepare data for linear regression
ndvi_sentinel = [img for img in resampled_sentinel_images.values()]
ndvi_planet = [data for data, _, _ in planet_images.values()]

#this should be later editied 
def preprocess_ndvi(ndvi_data):
    # Replace NaN and infinity values with 0
    ndvi_data = np.nan_to_num(ndvi_data, nan=0.0, posinf=0.0, neginf=0.0)
    return ndvi_data

# Flatten and preprocess the NDVI arrays for regression
ndvi_sentinel_flat = [preprocess_ndvi(img.flatten()) for img in ndvi_sentinel]
ndvi_planet_flat = [preprocess_ndvi(img.flatten()) for img in ndvi_planet]

# Combine all data points
X = np.concatenate(ndvi_sentinel_flat)
y = np.concatenate(ndvi_planet_flat)


# Linear regression
model = LinearRegression().fit(X.reshape(-1, 1), y)
X_flat = X.flatten()
y_flat = y.flatten()

# Scatter plot with values
# To do: deal with 0s
plt.figure(figsize=(8, 6))
plt.scatter(X_flat, y_flat, alpha=0.5)
plt.plot(X_flat, model.predict(X.reshape(-1, 1)), color='red')  # Regression line
plt.xlabel('Sentinel-2 NDVI Values')
plt.ylabel('Planet NDVI Values')
plt.show()


# Apply the model to resampled Sentinel NDVI images

synthetic_planet_ndvi = {}
for doy, img in resampled_sentinel_images.items():
    img_preprocessed = preprocess_ndvi(img)
    img_flat = img_preprocessed.flatten()
    img_flat = img_flat.reshape(-1, 1)  # Reshape for prediction
    predicted = model.predict(img_flat)
    synthetic_planet_ndvi[doy] = predicted.reshape(img_preprocessed.shape)


doy_to_plot = next(iter(synthetic_planet_ndvi))
# Real Planet NDVI image
real_planet_ndvi = planet_images[doy_to_plot][0]

# Synthetic  image
synthetic_ndvi = synthetic_planet_ndvi[doy_to_plot]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot Real Planet NDVI
im0 = axs[0].imshow(real_planet_ndvi, cmap='viridis')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title(f"Real Planet NDVI - DOY {doy_to_plot}")

# Plot Synthetic NDVI (based on S2 but with PS properites)
im1 = axs[1].imshow(synthetic_ndvi, cmap='viridis')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title(f"Synthetic NDVI - DOY {doy_to_plot}")

plt.show()

