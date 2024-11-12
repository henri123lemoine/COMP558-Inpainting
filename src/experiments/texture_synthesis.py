import cv2
import numpy as np
from loguru import logger

from src.experiments.base import InpaintingExperiment


class TextureSynthesisExperiment(InpaintingExperiment):
    """Experiment for testing texture synthesis-based inpainting."""

    def _create_synthetic_dataset(self) -> None:
        """Create synthetic texture dataset if it doesn't exist."""
        dataset_dir = self.dataset_dir / "textures"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # List of patterns to create
        patterns = {
            "checker": self._create_checker_pattern,
            "diagonal": self._create_diagonal_pattern,
            "dots": self._create_dots_pattern,
            "random": self._create_random_pattern,
        }

        for name, pattern_func in patterns.items():
            image_path = dataset_dir / f"{name}.png"
            mask_path = dataset_dir / f"{name}_mask.png"

            if not image_path.exists() or not mask_path.exists():
                logger.info(f"Creating synthetic texture: {name}")
                image = pattern_func(64)
                mask = self._create_center_mask(image)

                cv2.imwrite(str(image_path), image)
                cv2.imwrite(str(mask_path), mask)

    def _create_checker_pattern(self, size: int) -> np.ndarray:
        """Create a checker pattern."""
        pattern = np.zeros((size, size), dtype=np.uint8)
        pattern[::4, ::4] = 255
        pattern[1::4, 1::4] = 255
        return pattern

    def _create_diagonal_pattern(self, size: int) -> np.ndarray:
        """Create a diagonal stripes pattern."""
        pattern = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            pattern[i, (i // 2) % (size // 4) :: size // 4] = 255
        return pattern

    def _create_dots_pattern(self, size: int) -> np.ndarray:
        """Create a pattern of dots."""
        pattern = np.zeros((size, size), dtype=np.uint8)
        for i in range(size // 8, size, size // 8):
            for j in range(size // 8, size, size // 8):
                cv2.circle(pattern, (i, j), 2, 255, -1)
        return pattern

    def _create_random_pattern(self, size: int) -> np.ndarray:
        """Create a random but structured pattern."""
        pattern = np.random.randint(0, 2, (size // 4, size // 4), dtype=np.uint8) * 255
        return cv2.resize(pattern, (size, size), interpolation=cv2.INTER_NEAREST)

    def _create_center_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mask in the center of the image."""
        h, w = image.shape[:2]
        mask = np.zeros_like(image)
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        mask[y1:y2, x1:x2] = 255
        return mask

    def _load_dataset(self) -> dict:
        """Load or create synthetic texture dataset."""
        # Create synthetic dataset if needed
        self._create_synthetic_dataset()

        dataset_dir = self.dataset_dir / "textures"
        dataset = {}

        # Load all image-mask pairs
        for image_path in dataset_dir.glob("*.png"):
            if "_mask" not in image_path.stem:
                mask_path = dataset_dir / f"{image_path.stem}_mask.png"
                if mask_path.exists():
                    image, mask = self.algorithm.load_image(image_path, mask_path, grayscale=True)
                    dataset[image_path.stem] = {"image": image, "mask": mask}

        return dataset
