from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from loguru import logger

Image: TypeAlias = np.ndarray  # Shape: (H, W) or (H, W, C)
Mask: TypeAlias = np.ndarray  # Shape: (H, W), dtype: bool or uint8
PathLike: TypeAlias = str | Path


@dataclass
class InpaintingInputs:
    """Validated and preprocessed inputs for inpainting."""

    image: Image  # Normalized to [0, 1]
    mask: Mask  # Binary mask where True/1 indicates pixels to inpaint
    original_dtype: np.dtype
    original_range: tuple[float, float]


class InpaintingAlgorithm(ABC):
    """Base class for all inpainting algorithms."""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {self.name} algorithm")

    def validate_inputs(self, image: Image, mask: Mask) -> None:
        """Validate input image and mask."""
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and mask must have same spatial dimensions. "
                f"Got image shape {image.shape} and mask shape {mask.shape}"
            )

        if not np.issubdtype(mask.dtype, np.integer) and not mask.dtype == bool:
            raise ValueError(f"Mask must be integer or boolean type, got {mask.dtype}")

        if np.all(mask == 0):
            logger.warning("Empty mask, nothing to inpaint")
            return image

        if np.all(mask == 1):
            logger.warning("Full mask, nothing to inpaint")
            return image

    def preprocess_inputs(self, image: Image, mask: Mask) -> InpaintingInputs:
        """Preprocess inputs to standardized format."""
        # Store original properties for reconstruction
        original_dtype = image.dtype
        original_range = (float(image.min()), float(image.max()))

        # Normalize image to [0, 1]
        image_float = image.astype(np.float32)
        if original_range[1] > 1.0:
            image_float /= 255.0

        # Ensure mask is binary
        if mask.dtype == bool:
            mask_binary = mask
        else:
            # Important: standardize what 1 means!
            # Here we say 1/True means "pixels to inpaint"
            mask_binary = (mask > 0).astype(bool)

        # Validate after preprocessing
        self.validate_inputs(image_float, mask_binary)

        # Debug visualizations
        logger.debug(f"Mask statistics: {mask_binary.sum()}/{mask_binary.size} pixels to inpaint")
        logger.debug(
            f"Image range after normalization: [{image_float.min():.3f}, {image_float.max():.3f}]"
        )

        return InpaintingInputs(
            image=image_float,
            mask=mask_binary,
            original_dtype=original_dtype,
            original_range=original_range,
        )

    def postprocess_output(self, output: Image, inputs: InpaintingInputs) -> Image:
        """Convert output back to original format."""
        # Clip to valid range
        output_clipped = np.clip(output, 0, 1)

        # Scale back to original range if needed
        if inputs.original_range[1] > 1.0:
            output_clipped *= 255.0

        # Convert back to original dtype
        return output_clipped.astype(inputs.original_dtype)

    def inpaint(self, image: Image, mask: Mask, **kwargs) -> Image:
        """Inpaint the masked region.

        Args:
            image: Input image normalized to [0, 1]
            mask: Binary mask where True/1 indicates pixels to inpaint
            **kwargs: Algorithm-specific parameters

        Returns:
            Inpainted image normalized to [0, 1]
        """
        inputs = self.preprocess_inputs(image, mask)
        output = self._inpaint(inputs.image, inputs.mask, **kwargs)
        return self.postprocess_output(output, inputs)

    @abstractmethod
    def _inpaint(self, image: Image, mask: Mask, **kwargs) -> Image:
        """Inpainting logic for the specific algorithm."""
        pass

    def load_image(
        self,
        image_path: PathLike,
        mask_path: PathLike | None = None,
        grayscale: bool = False,
    ) -> tuple[Image, Mask | None]:
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
        original: Image | None = None,
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

    def run_example(self):
        import cv2
        import matplotlib.pyplot as plt

        # Load images
        image = cv2.imread("data/datasets/real/real_1227/image.png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread("data/datasets/real/real_1227/mask_brush.png", cv2.IMREAD_GRAYSCALE)
        print(image.shape, mask.shape)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # Create a copy of the image for scrambling
        scrambled_image = image.copy()

        # Get the masked region
        mask_region = mask > 0.5

        # Option 1: Fill with random noise
        # scrambled_image[mask_region] = np.random.uniform(0, 1, size=scrambled_image[mask_region].shape)

        # Option 2: Fill with mean + noise (more realistic)
        valid_pixels = image[~mask_region]
        mean_value = np.mean(valid_pixels)
        std_value = np.std(valid_pixels)
        scrambled_image[mask_region] = np.random.normal(
            mean_value, std_value, size=scrambled_image[mask_region].shape
        )
        scrambled_image = np.clip(scrambled_image, 0, 1)

        # Run inpainting on the scrambled image
        result = self.inpaint(scrambled_image, mask)

        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

        ax1.imshow(image, cmap="gray")
        ax1.set_title("Original")
        ax1.axis("off")

        ax2.imshow(scrambled_image, cmap="gray")
        ax2.set_title("Scrambled Input")
        ax2.axis("off")

        ax3.imshow(mask, cmap="gray")
        ax3.set_title("Mask")
        ax3.axis("off")

        ax4.imshow(result, cmap="gray")
        ax4.set_title("Result")
        ax4.axis("off")

        plt.tight_layout()
        plt.show()
