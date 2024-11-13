from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypeAlias

import cv2
import numpy as np
from loguru import logger

Image: TypeAlias = np.ndarray  # Shape: (H, W) or (H, W, C)
Mask: TypeAlias = np.ndarray  # Shape: (H, W), dtype: bool or uint8
PathLike: TypeAlias = str | Path


class InpaintingAlgorithm(ABC):
    """Base class for all inpainting algorithms."""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {self.name} algorithm")

    def load_image(
        self,
        image_path: PathLike,
        mask_path: Optional[PathLike] = None,
        grayscale: bool = False,
    ) -> tuple[Image, Optional[Mask]]:
        """Load and preprocess image and optional mask.

        Returns:
            Tuple of (image, mask) normalized to [0, 1] range
        """
        # Validate paths
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if mask_path and not Path(mask_path).exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load image
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), flag)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert color space and normalize
        if not grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Load and normalize mask if provided
        mask = None
        if mask_path:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
            mask = mask.astype(np.float32) / 255.0

        # Validate shapes
        if mask is not None and image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image shape {image.shape[:2]} does not match mask shape {mask.shape}"
            )

        logger.debug(
            f"Loaded image {Path(image_path).name} with shape {image.shape}"
            + (f" and mask {Path(mask_path).name}" if mask is not None else "")
        )
        return image, mask

    def save_result(
        self,
        result: Image,
        output_path: PathLike,
        original: Optional[Image] = None,
    ) -> None:
        """Save inpainting result, optionally with side-by-side comparison."""
        # Validate inputs
        if not isinstance(result, np.ndarray):
            raise TypeError("Result must be a numpy array")
        if original is not None and original.shape != result.shape:
            raise ValueError("Original and result shapes must match")

        # Convert to uint8 range [0, 255]
        result_uint8 = (result * 255).astype(np.uint8)

        # Create side-by-side comparison if original provided
        if original is not None:
            original_uint8 = (original * 255).astype(np.uint8)
            result_uint8 = np.hstack([original_uint8, result_uint8])

        # Convert RGB to BGR for OpenCV if needed
        if len(result_uint8.shape) == 3:
            result_uint8 = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        if not cv2.imwrite(str(output_path), result_uint8):
            raise IOError(f"Failed to save result to {output_path}")

        logger.info(f"Saved result to {output_path}")

    @abstractmethod
    def inpaint(
        self,
        image: Image,
        mask: Mask,
        **kwargs,
    ) -> Image:
        """Inpaint the masked region.

        Args:
            image: Input image normalized to [0, 1]
            mask: Binary mask where 1 indicates pixels to inpaint
            **kwargs: Algorithm-specific parameters

        Returns:
            Inpainted image normalized to [0, 1]
        """
        pass
