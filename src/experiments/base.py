from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import wasserstein_distance
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.algorithms.base import InpaintingAlgorithm
from src.experiments.utils.visualization import plot_inpainting_result
from src.settings import DATA_PATH, DATASETS_PATH


@dataclass
class InpaintingMetrics:
    """Dataclass to store inpainting quality metrics."""

    psnr: float  # Peak signal-to-noise ratio
    ssim: float  # Structural similarity index
    emd: float  # Earth mover's distance (Wasserstein)
    mae: float  # Mean absolute error
    execution_time: float  # Time taken for inpainting in seconds

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "PSNR": self.psnr,
            "SSIM": self.ssim,
            "EMD": self.emd,
            "MAE": self.mae,
            "Time (s)": self.execution_time,
        }

    @staticmethod
    def get_header() -> str:
        """Get CSV header string."""
        return "image_name,algorithm,psnr,ssim,emd,mae,execution_time\n"

    def to_csv_line(self, image_name: str, algorithm: str) -> str:
        """Convert metrics to CSV line."""
        return (
            f"{image_name},{algorithm},{self.psnr:.4f},{self.ssim:.4f},"
            f"{self.emd:.4f},{self.mae:.4f},{self.execution_time:.4f}\n"
        )


class InpaintingExperiment(ABC):
    """Base class for all inpainting experiments."""

    def __init__(
        self,
        name: str,
        algorithm: InpaintingAlgorithm,
        dataset_dir: str | Path | None = None,
    ):
        self.name = name
        self.algorithm = algorithm
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATASETS_PATH

        # Create directory structure
        self.output_dir = DATA_PATH / "results" / self.name
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.results_dir = self.output_dir / "inpainted"

        for directory in [
            self.output_dir,
            self.metrics_dir,
            self.visualizations_dir,
            self.results_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize metrics file
        self.metrics_file = self.metrics_dir / "metrics.csv"
        if not self.metrics_file.exists():
            with open(self.metrics_file, "w") as f:
                f.write(InpaintingMetrics.get_header())

        logger.info(f"Initialized experiment '{name}' using {algorithm.name} algorithm")

    def compute_metrics(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        execution_time: float,
        mask: np.ndarray | None = None,
    ) -> InpaintingMetrics:
        """Compute comprehensive quality metrics for inpainting result.

        Args:
            original: Original image
            inpainted: Inpainted result
            execution_time: Time taken for inpainting
            mask: Optional mask to focus evaluation on inpainted region

        Returns:
            InpaintingMetrics object containing all computed metrics
        """
        # Ensure images are in the correct format
        if original.dtype != np.uint8:
            original = (original * 255).astype(np.uint8)
        if inpainted.dtype != np.uint8:
            inpainted = (inpainted * 255).astype(np.uint8)

        # Compute metrics
        try:
            if mask is not None:
                # Focus on inpainted region
                mask_bool = mask.astype(bool)
                psnr = peak_signal_noise_ratio(
                    original[mask_bool], inpainted[mask_bool], data_range=255
                )
                ssim = structural_similarity(
                    original[mask_bool].reshape(-1, 1),
                    inpainted[mask_bool].reshape(-1, 1),
                    data_range=255,
                )
                emd = wasserstein_distance(
                    original[mask_bool].flatten(), inpainted[mask_bool].flatten()
                )
                mae = np.mean(
                    np.abs(original[mask_bool].astype(float) - inpainted[mask_bool].astype(float))
                )
            else:
                # Compute on whole image
                psnr = peak_signal_noise_ratio(original, inpainted, data_range=255)
                ssim = structural_similarity(original, inpainted, data_range=255)
                emd = wasserstein_distance(original.flatten(), inpainted.flatten())
                mae = np.mean(np.abs(original.astype(float) - inpainted.astype(float)))

            return InpaintingMetrics(psnr, ssim, emd, mae, execution_time)

        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            # Return default metrics in case of error
            return InpaintingMetrics(0.0, 0.0, float("inf"), float("inf"), execution_time)

    def save_metrics(
        self,
        metrics: InpaintingMetrics,
        image_name: str,
    ) -> None:
        """Save metrics to CSV file.

        Args:
            metrics: Computed metrics
            image_name: Name of the processed image
        """
        try:
            with open(self.metrics_file, "a") as f:
                f.write(metrics.to_csv_line(image_name, self.algorithm.name))
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def load_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Load the dataset for the experiment with improved error handling.

        Returns:
            Dictionary of image data in the format:
            {
                'image_name': {
                    'image': image_data,
                    'mask': mask_data
                }
            }

        Raises:
            FileNotFoundError: If dataset directory doesn't exist
            ValueError: If no valid image-mask pairs found
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        dataset = self._load_dataset()

        if not dataset:
            raise ValueError(f"No valid image-mask pairs found in {self.dataset_dir}")

        return dataset

    @abstractmethod
    def _load_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Load the dataset for the experiment.

        To be implemented by subclasses based on specific dataset structure.
        """
        pass

    def validate_image_mask_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_name: str,
    ) -> tuple[bool, str]:
        """Validate image and mask compatibility.

        Args:
            image: Input image
            mask: Inpainting mask
            image_name: Name of the image for logging

        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None or mask is None:
            return False, f"Invalid image or mask for {image_name}"

        if image.shape[:2] != mask.shape[:2]:
            return False, f"Image and mask shapes don't match for {image_name}"

        if mask.dtype != np.uint8 and mask.dtype != bool:
            return False, f"Invalid mask dtype for {image_name}"

        if np.all(mask == 0):
            return False, f"Empty mask for {image_name}"

        return True, ""

    def run(self, save_visualizations: bool = True) -> None:
        """Run the experiment with comprehensive logging and error handling."""
        logger.info(f"Running experiment: {self.name}")

        try:
            # Load dataset
            dataset = self.load_dataset()
            logger.info(f"Loaded dataset with {len(dataset)} images")

            # Process each image
            for image_name, data in dataset.items():
                logger.info(f"Processing image: {image_name}")

                # Validate input
                is_valid, error_msg = self.validate_image_mask_pair(
                    data["image"], data["mask"], image_name
                )
                if not is_valid:
                    logger.error(error_msg)
                    continue

                try:
                    # Time the inpainting process
                    import time

                    start_time = time.time()
                    result = self.algorithm.inpaint(data["image"], data["mask"])
                    execution_time = time.time() - start_time

                    # Compute and save metrics
                    metrics = self.compute_metrics(
                        data["image"], result, execution_time, data["mask"]
                    )
                    self.save_metrics(metrics, image_name)

                    # Save results
                    output_path = self.results_dir / f"{image_name}_result.png"
                    self.algorithm.save_result(result, output_path)

                    # Save visualization
                    if save_visualizations:
                        vis_path = self.visualizations_dir / f"{image_name}_visualization.png"
                        plot_inpainting_result(
                            data["image"],
                            data["mask"],
                            result,
                            save_path=vis_path,
                            title=f"{self.algorithm.name} on {image_name}",
                            metrics=metrics,
                        )

                    logger.info(f"Completed processing {image_name}")
                    logger.info(f"Metrics: {metrics.to_dict()}")

                except Exception as e:
                    logger.error(f"Error processing {image_name}: {str(e)}")
                    continue

            logger.info("Experiment completed")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
