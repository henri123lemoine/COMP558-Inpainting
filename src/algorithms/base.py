from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeAlias

import cv2
import numpy as np
from loguru import logger

Image: TypeAlias = np.ndarray  # Shape: (H, W) or (H, W, C)
Mask: TypeAlias = np.ndarray  # Shape: (H, W), dtype: bool or uint8
PathLike: TypeAlias = str | Path


class InpaintingResult(NamedTuple):
    """Results from inpainting operation."""

    original: Image
    masked: Image
    mask: Mask
    output: Image


@dataclass
class InpaintingInputs:
    """Validated and preprocessed inputs for inpainting."""

    original: Image
    image: Image
    mask: Mask
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
        # Store original image before any processing
        original = image.copy()

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
            mask_binary = (mask > 0).astype(bool)

        # Replace masked pixels with np.nan
        image_float_masked = image_float.copy()
        image_float_masked[mask_binary] = np.nan

        # Validate after preprocessing
        self.validate_inputs(image_float_masked, mask_binary)

        logger.debug(f"Mask statistics: {mask_binary.sum()}/{mask_binary.size} pixels to inpaint")
        logger.debug(
            f"Image range after normalization: [{np.nanmin(image_float_masked):.3f}, {np.nanmax(image_float_masked):.3f}]"
        )

        return InpaintingInputs(
            original=original,
            image=image_float_masked,
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

    def inpaint(self, image: Image, mask: Mask, **kwargs) -> tuple[Image, Image, Mask, Image]:
        """Inpaint the masked region.

        Returns
        - original: Original input image
        - masked: Image with mask applied (np.nan in masked regions)
        - mask: Binary mask showing regions to inpaint
        - output: Final inpainted result

        ***IMPORTANT***: _inpaint() method must *never* see behind the mask!
        """
        inputs = self.preprocess_inputs(image, mask)
        output = self._inpaint(inputs.image, inputs.mask, **kwargs)
        processed_output = self.postprocess_output(output, inputs)

        return inputs.original, inputs.image, inputs.mask, processed_output

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

    def run_example(
        self,
        image_path: str | Path = Path("test-images/portrait.png"),
        mask_path: str | Path = Path("test-images/masks/portrait.png"),
        scale_factor=1.0,
        save_output=True,
    ):
        """Run inpainting experiment with various options."""
        import cv2
        import matplotlib.pyplot as plt

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Error: Could not read the image file at {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Error: Could not read the mask file at {mask_path}")

        if mask.shape != image.shape:
            print(f"Resizing mask from {mask.shape} to {image.shape}...")
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        if scale_factor != 1.0:
            new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        print(f"Working with image size: {image.shape}")

        original, masked, mask, inpainted = self.inpaint(image, mask)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(original, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(masked, cmap="gray")
        plt.title("Masked Image")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(inpainted, cmap="gray")
        plt.title("Inpainted Result")
        plt.axis("off")

        plt.tight_layout()

        if save_output:
            plt.savefig("comparison.png")

        plt.show()
