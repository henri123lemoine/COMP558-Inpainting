from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class InpaintingAlgorithm(ABC):
    """Base class for all inpainting algorithms."""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {self.name} algorithm")

    def load_image(
        self,
        image_path: str | Path,
        mask_path: str | Path | None = None,
        grayscale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Load image and optional mask."""
        # Load image
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), flag)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB if color image
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Load mask if provided
        mask = None
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask from {mask_path}")
            # Normalize to [0, 1]
            mask = mask.astype(np.float32) / 255.0

        logger.debug(
            f"Loaded image {image_path} with shape {image.shape}"
            + (f" and mask {mask_path}" if mask is not None else "")
        )
        return image, mask

    def save_result(
        self,
        result: np.ndarray,
        output_path: str | Path,
        original: np.ndarray | None = None,
    ) -> None:
        """Save the inpainting result."""
        # Convert to uint8
        result = (result * 255).astype(np.uint8)

        # If original provided, create side-by-side comparison
        if original is not None:
            original = (original * 255).astype(np.uint8)
            result = np.hstack([original, result])

        # Convert RGB to BGR if color image
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Save
        cv2.imwrite(str(output_path), result)
        logger.info(f"Saved result to {output_path}")

    @abstractmethod
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Inpaint the masked region."""
        pass
