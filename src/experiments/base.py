from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger

from src.algorithms.base import InpaintingAlgorithm
from src.settings import DATA_PATH, DATASETS_PATH
from src.utils.visualization import plot_inpainting_result


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

        # Create output directory
        self.output_dir = DATA_PATH / "results" / self.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized experiment '{name}' using {algorithm.name} algorithm")

    def load_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Load the dataset for the experiment.

        Returns:
            Dictionary of image data in the format:
            {
                'image_name': {
                    'image': image_data,
                    'mask': mask_data
                }
            }
        """
        return self._load_dataset()

    @abstractmethod
    def _load_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Load the dataset for the experiment.

        To be implemented by subclasses based on specific dataset structure.
        """
        pass

    def run(self, save_visualizations: bool = True) -> None:
        logger.info(f"Running experiment: {self.name}")

        # Load dataset
        dataset = self.load_dataset()
        logger.info(f"Loaded dataset with {len(dataset)} images")

        # Process each image
        for image_name, data in dataset.items():
            logger.info(f"Processing image: {image_name}")

            # Run inpainting
            result = self.algorithm.inpaint(data["image"], data["mask"])

            # Save result
            output_path = self.output_dir / f"{image_name}_result.png"
            self.algorithm.save_result(result, output_path)

            # Save visualization
            if save_visualizations:
                vis_path = self.output_dir / f"{image_name}_visualization.png"
                plot_inpainting_result(
                    data["image"],
                    data["mask"],
                    result,
                    save_path=vis_path,
                    title=f"{self.algorithm.name} on {image_name}",
                )

        logger.info("Experiment completed")
