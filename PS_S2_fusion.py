import glob
import os
import pickle
import re
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.crs
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
import random
import fiona
import rasterio
from rasterio import features

# Global Configuration (Consider using a configuration file)
CONFIG = {"SCALE_DATA": False, "FILL_WITH_ZEROS": False, "MASK_MAIZE": True}


class NDVIImage:
    def __init__(self, ndvi: np.ndarray, transform: rasterio.Affine, crs: rasterio.CRS):
        self.ndvi: np.ndarray = ndvi
        self.transform: rasterio.Affine = transform
        self.crs: rasterio.CRS = crs


class NDVIDataLoader:
    """Handles loading and preprocessing NDVI image data."""

    @staticmethod
    def extract_day_of_year(filename: str) -> int:
        """Extracts the day of year (DOY) from a filename."""
        match = re.search(r"_doy_(\d+)", filename)
        return int(match.group(1)) if match else None

    def load_ndvi_images(self, directory: str, pattern: str) -> Dict[int, NDVIImage]:
        """Loads NDVI images and their metadata from a directory."""
        image_files = glob.glob(os.path.join(directory, pattern))
        images = {}
        for file in image_files:
            day_of_year = self.extract_day_of_year(file)
            if day_of_year is not None:
                with rasterio.open(file) as src:
                    images[day_of_year] = NDVIImage(src.read(1), src.transform, src.crs)
        return images


class NDVIProcessor:
    """Preprocesses and prepares NDVI data for regression."""

    def mask_array_with_shapefile(self, ndiv_data, transform, shapefile_path):
        """
        Masks a NumPy array representing raster data using a shapefile.

        Args:
            ndiv_data: The 2D or 3D NumPy array containing the raster data.
            crs: The coordinate reference system (CRS) of the raster data.
            transform: The affine transformation of the raster data.
            shapefile_path: Path to the shapefile.

        Returns:
            The masked NumPy array (same dimensions as input) with a nodata
            value applied outside the shapefile's geometry.
        """
        with fiona.open(shapefile_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        # Create a mask array (initially all False)
        mask = features.rasterize(
            ((geom, 0) for geom in shapes),
            out_shape=ndiv_data.shape,  # Assuming ndiv_data is 2D or 3D
            fill=1,  # Fill with 1 (True) for areas outside the shapes
            transform=transform,
            all_touched=True,  # Or False, depending on your needs
            dtype=rasterio.uint8,
        ).astype(
            bool
        )  # Convert to boolean

        # Apply the mask to the raster array
        masked_array = ndiv_data.copy()  # Work on a copy
        if masked_array.ndim == 2:
            masked_array[mask] = np.nan  # Mask 2D array
        elif masked_array.ndim == 3:
            for band in range(masked_array.shape[0]):
                masked_array[band][mask] = np.nan  # Mask each band in 3D array
        else:
            raise ValueError("Input raster array should be 2D or 3D.")

        return masked_array

    @staticmethod
    def resample_sentinel_to_planet(
        sentinel_image: NDVIImage,
        planet_image: NDVIImage,
    ) -> NDVIImage:
        """Resamples Sentinel-2 data to Planet data resolution and projection."""
        resampled_sentinel_data = np.empty(planet_image.ndvi.shape, dtype=np.float32)
        reproject(
            source=sentinel_image.ndvi,
            destination=resampled_sentinel_data,
            src_transform=sentinel_image.transform,
            dst_transform=planet_image.transform,
            src_crs=sentinel_image.crs,
            dst_crs=planet_image.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        return NDVIImage(
            resampled_sentinel_data, sentinel_image.transform, sentinel_image.crs
        )

    @staticmethod
    def resample_planet_to_sentinel(
        sentinel_image: NDVIImage,
        planet_image: NDVIImage,
    ) -> NDVIImage:
        """Resamples Sentinel-2 data to Planet data resolution and projection."""
        resampled_planet_data = np.empty(sentinel_image.ndvi.shape, dtype=np.float32)
        reproject(
            source=planet_image.ndvi,
            destination=resampled_planet_data,
            src_transform=planet_image.transform,
            dst_transform=sentinel_image.transform,
            src_crs=planet_image.crs,
            dst_crs=sentinel_image.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        return NDVIImage(
            resampled_planet_data, sentinel_image.transform, sentinel_image.crs
        )

    @staticmethod
    def get_invalid_mask(
        sentinel_ndvi_data: np.ndarray, planet_ndvi_data: np.ndarray
    ) -> np.ndarray:
        """Handles invalid NDVI values (0 is treated as invalid)."""
        sentinel_invalid_mask = np.logical_or(
            np.isnan(sentinel_ndvi_data), np.isinf(sentinel_ndvi_data)
        )
        planet_invalid_mask = np.logical_or(
            np.isnan(planet_ndvi_data), np.isinf(planet_ndvi_data)
        )
        return np.logical_or(sentinel_invalid_mask, planet_invalid_mask)

    def extract_data_for_regression(
        self,
        sentinel_images: Dict[int, NDVIImage],
        planet_images: Dict[int, NDVIImage],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Extracts and combines NDVI data from Sentinel-2 and Planet images for regression.

        Args:
            sentinel_images: Sentinel-2 NDVI images by day of year.
            planet_images: Planet NDVI images by day of year.

        Returns:
            Tuple containing the preprocessed Sentinel-2 NDVI data and Planet NDVI data.
        """
        sentinel_combined_data = []
        planet_combined_data = []

        for planet_doy, planet_image in planet_images.items():
            closest_sentinel_doy = min(
                sentinel_images.keys(), key=lambda d: abs(d - planet_doy)
            )
            invalid_mask = self.get_invalid_mask(
                sentinel_images[closest_sentinel_doy].ndvi, planet_image.ndvi
            )

            preprocessed_sentinel_data = self.get_preprocessed_ndiv_data(
                sentinel_images[closest_sentinel_doy], invalid_mask
            )

            preprocessed_planet_data = self.get_preprocessed_ndiv_data(
                planet_image, invalid_mask
            )

            # Combine data from all images into a single array
            sentinel_combined_data.append(preprocessed_sentinel_data)
            planet_combined_data.append(preprocessed_planet_data)

        # Convert to numpy arrays outside the loop
        sentinel_combined_data = np.concatenate(sentinel_combined_data)
        planet_combined_data = np.concatenate(planet_combined_data)

        return sentinel_combined_data, planet_combined_data

    def resample_sentinel_images(
        self,
        sentinel_images: Dict[int, NDVIImage],
        planet_images: Dict[int, NDVIImage],
    ) -> Dict[int, NDVIImage]:
        """
        Resamples Sentinel-2 images to match Planet image resolution.

        Args:
            sentinel_images: Sentinel-2 NDVI images by day of year.
            planet_images: Planet NDVI images by day of year.

        Returns:
            A dictionary of resampled Sentinel-2 images.
        """
        resampled_sentinel_images = {}

        for planet_doy, planet_image in planet_images.items():
            closest_sentinel_doy = min(
                sentinel_images.keys(), key=lambda d: abs(d - planet_doy)
            )
            sentinel_image = sentinel_images[closest_sentinel_doy]

            resampled_sentinel_image = self.resample_sentinel_to_planet(
                sentinel_image, planet_image
            )
            resampled_sentinel_images[closest_sentinel_doy] = resampled_sentinel_image

        return resampled_sentinel_images

    def get_preprocessed_ndiv_data(
        self,
        image: NDVIImage,
        invalid_mask: np.ndarray,
    ) -> np.ndarray:
        ndiv_data = image.ndvi

        if CONFIG["FILL_WITH_ZEROS"]:
            ndiv_data[invalid_mask] = 0
        else:
            # Remove invalid values
            ndiv_data = ndiv_data[~invalid_mask]

        ndiv_data = ndiv_data.reshape(-1, 1)

        if CONFIG["SCALE_DATA"]:
            scaler = MaxAbsScaler()
            ndiv_data = scaler.fit_transform(ndiv_data)

        return ndiv_data


class NDVIVisualizer:
    """
    Provides visualization tools for NDVI data and predictions.
    """

    def __init__(self):
        pass

    @staticmethod
    def visualize_scatter_plot(
        X: np.ndarray, y: np.ndarray, model: LinearRegression | RandomForestRegressor
    ):
        """Plots a scatter plot of Sentinel-2 NDVI vs. Planet NDVI,
        along with the regression line."""
        for feat in X.T:
            feat_flat = feat.flatten()
            y_flat = y.flatten()

            # Downsample for scatter plot
            sample_indices_scatter = random.sample(range(len(feat_flat)), 10000)
            sample_indices_line = random.sample(range(len(feat_flat)), 10000)
            plt.rcParams["agg.path.chunksize"] = 10000

            plt.figure(figsize=(8, 8))
            plt.scatter(
                feat_flat[sample_indices_scatter],
                y_flat[sample_indices_scatter],
                alpha=0.5,
            )
            prediction = model.predict(X)[sample_indices_line]
            plt.plot(
                feat_flat[sample_indices_line],
                prediction,
                color="red",
            )
            plt.xlabel("Sentinel-2 NDVI Values")
            plt.ylabel("Planet NDVI Values")
            plt.title("Sentinel-2 NDVI vs. Planet NDVI")

            # Set axis limits to 0 to 1
            # plt.xlim(-0.2, 1)
            # plt.ylim(-0.2, 1)

            plt.show()

    @staticmethod
    def visualize_ndvi_images(
        real_sentinel_ndvi: np.ndarray,
        synthetic_sentinel_ndvi: np.ndarray,
        resampled_planet_ndvi: np.ndarray,
        day_of_year: int,
        year: int,
    ) -> None:
        """
        Visualizes real and synthetic NDVI images side by side.

        Args:
            real_sentinel_ndvi: The real Planet NDVI image.
            synthetic_sentinel_ndvi: The synthetically generated NDVI image.
            resampled_planet_ndvi: The resampled Sentinel NDVI image.
            day_of_year: The day of year for the images.
        """

        fig, axs = plt.subplots(3, 2, figsize=(12, 12))

        axs[0, 0].imshow(real_sentinel_ndvi, cmap="viridis", vmin=-0.2, vmax=1)
        fig.colorbar(
            axs[0, 0].imshow(real_sentinel_ndvi, cmap="viridis", vmin=-0.2, vmax=1),
            ax=axs[0, 0],
        )
        axs[0, 0].set_title(f"Real Sentinel 2 NDVI - DOY {day_of_year} - Year {year}")

        axs[0, 1].imshow(real_sentinel_ndvi, cmap="viridis", vmin=-0.2, vmax=1)
        fig.colorbar(
            axs[0, 1].imshow(real_sentinel_ndvi, cmap="viridis", vmin=-0.2, vmax=1),
            ax=axs[0, 1],
        )
        axs[0, 1].set_title(f"Real Sentinel 2 NDVI - DOY {day_of_year} - Year {year}")

        axs[1, 0].imshow(synthetic_sentinel_ndvi, cmap="viridis", vmin=-0.2, vmax=1)
        fig.colorbar(
            axs[1, 0].imshow(
                synthetic_sentinel_ndvi, cmap="viridis", vmin=-0.2, vmax=1
            ),
            ax=axs[1, 0],
        )
        axs[1, 0].set_title(
            f"Synthetic Sentinel 2 NDVI - DOY {day_of_year} - Year {year}"
        )

        axs[1, 1].imshow(resampled_planet_ndvi, cmap="viridis", vmin=-0.2, vmax=1)
        fig.colorbar(
            axs[1, 1].imshow(resampled_planet_ndvi, cmap="viridis", vmin=-0.2, vmax=1),
            ax=axs[1, 1],
        )
        axs[1, 1].set_title(
            f"Resampled Planet Scope NDVI - DOY {day_of_year} - Year {year}"
        )

        # Display Errors
        abs_error_synthetic = np.abs(synthetic_sentinel_ndvi - real_sentinel_ndvi)
        # print(f"Average absolute error synthetic:{np.nanmean(abs_error_synthetic)}")
        # print(f"Average error variance synthetic:{np.nanvar(abs_error_synthetic)}")

        axs[2, 0].imshow(abs_error_synthetic, cmap="Reds", vmin=0, vmax=1)
        fig.colorbar(
            axs[2, 0].imshow(abs_error_synthetic, cmap="Reds", vmin=0, vmax=1),
            ax=axs[2, 0],
        )
        axs[2, 0].set_title(
            f"Absolute error synthetic vs real NDVI - DOY {day_of_year} - Year {year}"
        )

        error_resampled = np.abs(resampled_planet_ndvi - real_sentinel_ndvi)
        # print(f"Average absolute error resampled:{np.nanmean(error_resampled)}")
        # print(f"Average error variance resampled:{np.nanvar(error_resampled)}")

        axs[2, 1].imshow(error_resampled, cmap="Reds", vmin=0, vmax=1)
        fig.colorbar(
            axs[2, 1].imshow(error_resampled, cmap="Reds", vmin=0, vmax=1),
            ax=axs[2, 1],
        )
        axs[2, 1].set_title(
            f"Absolute error resampled vs real NDVI - DOY {day_of_year} - Year {year}"
        )

        plt.tight_layout()
        plt.show()

    def plot_time_series(
        self,
        mse_dict_synth,
        mse_dict_resampled,
        mean_ndiv_dict_synth,
        mean_ndiv_dict_resampled,
        mean_ndiv_dict_real,
    ):
        # --- Data Preparation ---
        doys = sorted(mse_dict_synth.keys())

        # Create lists of data for easier plotting with Plotly
        mse_synth = [mse_dict_synth[doy] for doy in doys]
        mse_resampled = [mse_dict_resampled[doy] for doy in doys]

        mean_synth = [mean_ndiv_dict_synth[doy] for doy in doys]
        mean_resampled = [mean_ndiv_dict_resampled[doy] for doy in doys]
        mean_real = [mean_ndiv_dict_real[doy] for doy in doys]

        # --- Plotly Figure ---
        fig = go.Figure()

        # 1. Mean NDVI Lines
        fig.add_trace(
            go.Scatter(
                x=doys,
                y=mean_real,
                mode="lines",
                name="Real Sentinel-2",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=doys,
                y=mean_synth,
                mode="lines",
                name="Synthetic",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=doys,
                y=mean_resampled,
                mode="lines",
                name="Resampled PlanetScope",
                line=dict(color="orange"),
            )
        )

        # 2. Error Bars
        fig.add_trace(
            go.Scatter(
                x=doys,
                y=mean_synth,
                error_y=dict(type="data", array=mse_synth, visible=True),
                mode="markers",
                marker=dict(color="blue"),
                name="Synthetic Error",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=doys,
                y=mean_resampled,
                error_y=dict(type="data", array=mse_resampled, visible=True),
                mode="markers",
                marker=dict(color="orange"),
                name="Resampled Error",
            )
        )

        # --- Layout and Styling ---
        fig.update_layout(
            title="NDVI Comparison: Real, Synthetic, and Resampled",
            xaxis_title="Day of Year (DOY)",
            yaxis_title="Mean NDVI",
            width=1200,  # Width in pixels
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        fig.show()
