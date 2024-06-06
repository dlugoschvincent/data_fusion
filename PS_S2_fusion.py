import glob
import os
import pickle
import re
from typing import Tuple, Dict

from joblib import parallel_backend
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.crs
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import random

# Global Configuration (Consider using a configuration file)
CONFIG = {
    "SCALE_DATA": True,
    "FILL_WITH_ZEROS": True,
}


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
            resampling=Resampling.bilinear,
        )
        return NDVIImage(
            resampled_sentinel_data, sentinel_image.transform, sentinel_image.crs
        )

    @staticmethod
    def get_invalid_mask(ndvi_data: np.ndarray) -> np.ndarray:
        """Handles invalid NDVI values (0 is treated as invalid)."""
        ndvi_data[ndvi_data == 0] = np.nan
        return np.logical_or(np.isnan(ndvi_data), np.isinf(ndvi_data))

    def prepare_data_for_regression(
        self,
        sentinel_images: Dict[int, NDVIImage],
        planet_images: Dict[int, NDVIImage],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Dict[int, NDVIImage],
    ]:
        """
        Resamples, preprocesses, and prepares NDVI data for regression.

        Args:
            sentinel_images: Sentinel-2 NDVI images by day of year.
            planet_images: Planet NDVI images by day of year.

        Returns:
            Tuple containing the preprocessed Sentinel-2 NDVI data,
            Planet NDVI data, and a dictionary of resampled Sentinel-2 images.
        """
        resampled_sentinel_images = {}
        sentinel_combined_data = []
        planet_combined_data = []

        for planet_doy, planet_image in planet_images.items():
            closest_sentinel_doy = min(
                sentinel_images.keys(), key=lambda d: abs(d - planet_doy)
            )
            sentinel_image = sentinel_images[closest_sentinel_doy]

            resampled_sentinel_image = self.resample_sentinel_to_planet(
                sentinel_image, planet_image
            )

            sentinel_mask = self.get_invalid_mask(resampled_sentinel_image.ndvi)
            planet_mask = self.get_invalid_mask(planet_image.ndvi)
            combined_mask = np.logical_or(sentinel_mask, planet_mask)

            # Combine data from all images into a single array

            # Set invalid values to 0
            if CONFIG["FILL_WITH_ZEROS"]:
                resampled_sentinel_image.ndvi[combined_mask] = 0
                planet_image.ndvi[combined_mask] = 0
            else:
                # Remove invalid values
                resampled_sentinel_image.ndvi = resampled_sentinel_image.ndvi[
                    ~combined_mask
                ]
                planet_image.ndvi = planet_image.ndvi[~combined_mask]

            planet_image_data = planet_image.ndvi
            resampled_sentinel_image_data = resampled_sentinel_image.ndvi
            if CONFIG["SCALE_DATA"]:
                scaler = MaxAbsScaler()

                resampled_sentinel_image_data = scaler.fit_transform(
                    resampled_sentinel_image_data.reshape(-1, 1)
                )
                planet_image_data = scaler.fit_transform(
                    planet_image_data.reshape(-1, 1)
                )

            sentinel_combined_data.append(resampled_sentinel_image_data)
            planet_combined_data.append(planet_image_data)
            resampled_sentinel_image.ndvi = resampled_sentinel_image_data.reshape(
                resampled_sentinel_image.ndvi.shape
            )
            resampled_sentinel_images[closest_sentinel_doy] = resampled_sentinel_image

        # Convert to numpy arrays outside the loop
        sentinel_combined_data = np.concatenate(sentinel_combined_data)
        planet_combined_data = np.concatenate(planet_combined_data)

        return sentinel_combined_data, planet_combined_data, resampled_sentinel_images


class NDVIRegressor:
    """Trains and evaluates regression models for NDVI prediction."""

    def __init__(self, model: LinearRegression | RandomForestRegressor):
        """
        Initializes the NDVIRegressor.

        Args:
            model: The regression model (LinearRegression or RandomForestRegressor).
        """
        if not isinstance(model, (LinearRegression, RandomForestRegressor)):
            raise TypeError("Model should be LinearRegression or RandomForestRegressor")
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the model, loading from file if it exists."""
        model_filename = self._get_model_filename()
        if os.path.exists(model_filename):
            print(f"Loading model from: {model_filename}")
            with open(model_filename, "rb") as file:
                self.model = pickle.load(file)
        else:
            print("Training model from scratch...")
            # with parallel_backend("threading", n_jobs=10):
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluates the model using MSE."""

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts NDVI values."""
        return self.model.predict(X)

    def save_model(self) -> None:
        """Saves the trained model to a file."""
        filename = self._get_model_filename()
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved to: {filename}")

    def _get_model_filename(self) -> str:
        """Generates the model filename based on configuration and model parameters."""
        model_params_str = ""
        if isinstance(self.model, RandomForestRegressor):
            model_params_str = "rf_model_" + "_".join(
                f"{k}={v}"
                for k, v in self.model.get_params().items()
                if k in ["max_depth", "max_features", "n_estimators"]
            )
        elif isinstance(self.model, LinearRegression):
            model_params_str = "linear_regression"

        filename = f"{model_params_str}_scaled_{CONFIG['SCALE_DATA']}.pkl"
        return filename


class NDVISynthesizer:
    """
    Generates synthetic NDVI images using a trained regression model.
    """

    def __init__(self, model: NDVIRegressor):
        """
        Initializes the NDVISynthesizer.

        Args:
            model: The trained NDVI regression model.
        """
        self.model = model

    def generate_synthetic_ndvi(self, resampled_sentinel_image: NDVIImage) -> NDVIImage:
        """
        Generates a synthetic Planet NDVI image using the trained model.

        Args:
            resampled_sentinel_image: The resampled Sentinel-2 NDVI image and metadata.

        Returns:
            The synthetic Planet NDVI image.
        """
        resampled_sentinel_array = resampled_sentinel_image.ndvi
        invalid_mask = NDVIProcessor.get_invalid_mask(resampled_sentinel_array)
        resampled_sentinel_array[invalid_mask] = 0
        # Scale the valid Sentinel values before prediction if scale_data is True
        if CONFIG["SCALE_DATA"]:
            scaler = MaxAbsScaler()
            resampled_sentinel_array_scaled = scaler.fit_transform(
                resampled_sentinel_array.reshape(-1, 1)
            )
            predicted_values = self.model.predict(
                resampled_sentinel_array_scaled.reshape(-1, 1)
            )
        else:
            predicted_values = self.model.predict(
                resampled_sentinel_array.reshape(-1, 1)
            )

        synthetic_ndvi_image = NDVIImage(
            predicted_values.reshape(resampled_sentinel_array.shape), None, None
        )

        return synthetic_ndvi_image


class NDVIVisualizer:
    """
    Provides visualization tools for NDVI data and predictions.
    """

    def __init__(self):
        pass

    @staticmethod
    def visualize_scatter_plot(X: np.ndarray, y: np.ndarray, model: NDVIRegressor):
        """Plots a scatter plot of Sentinel-2 NDVI vs. Planet NDVI,
        along with the regression line."""
        X_flat = X.flatten()
        y_flat = y.flatten()

        # Downsample for scatter plot
        sample_indices_scatter = random.sample(range(len(X_flat)), 10000)
        sample_indices_line = random.sample(range(len(X_flat)), 10000)
        plt.rcParams["agg.path.chunksize"] = 10000

        plt.figure(figsize=(8, 8))
        plt.scatter(X_flat[sample_indices_scatter], y_flat[sample_indices_scatter], alpha=0.5)
        plt.plot(X_flat[sample_indices_line], model.predict(X)[sample_indices_line], color="red")
        plt.xlabel("Sentinel-2 NDVI Values")
        plt.ylabel("Planet NDVI Values")
        plt.title("Sentinel-2 NDVI vs. Planet NDVI")

        # Set axis limits to 0 to 1
        plt.xlim(-0.2, 1)
        plt.ylim(-0.2, 1)

        plt.show()

    @staticmethod
    def visualize_ndvi_images(
        real_ndvi_image: NDVIImage,
        synthetic_ndvi_image: NDVIImage,
        day_of_year: int,
    ):
        """
        Visualizes real and synthetic NDVI images side by side.

        Args:
            real_ndvi_image: The real Planet NDVI image.
            synthetic_ndvi_image: The synthetically generated NDVI image.
            day_of_year: The day of year for the images.
        """
        if CONFIG["SCALE_DATA"]:
            scaler = MaxAbsScaler()
            real_ndvi_flattened = real_ndvi_image.ndvi.flatten()
            real_ndvi_image.ndvi = scaler.fit_transform(
                real_ndvi_flattened.reshape(-1, 1)
            ).reshape(real_ndvi_image.ndvi.shape)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        im0 = axs[0].imshow(real_ndvi_image.ndvi, cmap="viridis", vmin=-0.2, vmax=1)
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_title(f"Real Planet NDVI - DOY {day_of_year}")

        im1 = axs[1].imshow(
            synthetic_ndvi_image.ndvi, cmap="viridis", vmin=-0.2, vmax=1
        )
        fig.colorbar(im1, ax=axs[1])
        axs[1].set_title(f"Synthetic NDVI - DOY {day_of_year}")

        plt.tight_layout()
        plt.show()
