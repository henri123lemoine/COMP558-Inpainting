from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt
import torchvision.transforms as T
from loguru import logger
from torchvision.datasets import CIFAR10

from .masks import InpaintSample, MaskConfig, MaskGenerator
from .utils import ImageCategory


class ImageGenerator:
    """Manages generation of synthetic test images."""

    ## Basic Patterns ##

    def create_line_image(self, size: int) -> npt.NDArray[np.uint8]:
        """Create test image with straight lines."""
        image = np.zeros((size, size), dtype=np.uint8)
        cv2.line(image, (size // 4, size // 4), (3 * size // 4, size // 4), 255, 2)
        cv2.line(image, (size // 4, size // 4), (size // 4, 3 * size // 4), 255, 2)
        cv2.line(image, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, 2)
        return image

    def create_shape_image(self, size: int) -> npt.NDArray[np.uint8]:
        """Create test image with basic shapes."""
        image = np.zeros((size, size), dtype=np.uint8)

        cv2.rectangle(image, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, 2)
        cv2.circle(image, (size // 2, size // 2), size // 4, 255, 2)

        triangle_pts = np.array(
            [[size // 2, size // 4], [size // 4, 3 * size // 4], [3 * size // 4, 3 * size // 4]]
        )
        cv2.polylines(image, [triangle_pts], True, 255, 2)
        return image

    def create_curve_image(self, size: int) -> npt.NDArray[np.uint8]:
        """Create test image with curves."""
        image = np.zeros((size, size), dtype=np.uint8)

        x = np.linspace(0, 4 * np.pi, size)
        y = np.sin(x) * size // 4 + size // 2
        points = np.column_stack((np.linspace(0, size - 1, size), y)).astype(np.int32)

        for i in range(len(points) - 1):
            pt1 = tuple(map(int, points[i]))
            pt2 = tuple(map(int, points[i + 1]))
            cv2.line(image, pt1, pt2, 255, 2)
        return image

    def create_diagonal_pattern(self, size: int) -> np.ndarray:
        """Create a diagonal stripes pattern."""
        pattern = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            pattern[i, (i // 2) % (size // 4) :: size // 4] = 255
        return pattern

    def create_cross_lines(self, size: int) -> np.ndarray:
        """Create an image with crossing lines and a circle."""
        image = np.zeros((size, size), dtype=np.uint8)
        cv2.line(image, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, 2)
        cv2.line(image, (size // 4, 3 * size // 4), (3 * size // 4, size // 4), 255, 2)
        cv2.circle(image, (size // 2, size // 2), size // 4, 255, 2)
        return image

    def create_checkerboard(self, size: int) -> npt.NDArray[np.uint8]:
        """Create checkerboard pattern."""
        image = np.zeros((size, size), dtype=np.uint8)
        square_size = size // 8

        for i in range(0, size, square_size * 2):
            for j in range(0, size, square_size * 2):
                image[i : i + square_size, j : j + square_size] = 255
                image[
                    i + square_size : i + 2 * square_size, j + square_size : j + 2 * square_size
                ] = 255
        return image

    def create_checker_pattern(self, size: int) -> np.ndarray:
        """Create a checker pattern."""
        pattern = np.zeros((size, size), dtype=np.uint8)
        pattern[::4, ::4] = 255  # Set every 4th pixel starting at (0,0)
        pattern[1::4, 1::4] = 255  # Set every 4th pixel starting at (1,1)
        return pattern

    def create_textured_checker(self, size: int) -> np.ndarray:
        """Create a noisy checkerboard pattern."""
        image = np.zeros((size, size), dtype=np.uint8)
        square_size = 8
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i + j) // square_size % 2 == 0:
                    image[i : i + square_size, j : j + square_size] = 255

        noise = np.random.normal(0, 20, (size, size))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return image

    def create_dot_pattern(self, size: int) -> npt.NDArray[np.uint8]:
        """Create regular dot pattern."""
        image = np.zeros((size, size), dtype=np.uint8)
        spacing = size // 8

        for i in range(spacing, size - spacing, spacing):
            for j in range(spacing, size - spacing, spacing):
                cv2.circle(image, (i, j), 3, 255, -1)
        return image

    def create_noise_pattern(self, size: int) -> npt.NDArray[np.uint8]:
        """Create structured noise pattern."""
        noise = np.random.randint(0, 256, (size // 4, size // 4), dtype=np.uint8)
        return cv2.resize(noise, (size, size), interpolation=cv2.INTER_NEAREST)

    ## Gradients ##

    def create_linear_gradient(self, size: int) -> npt.NDArray[np.uint8]:
        """Create linear gradient."""
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        return ((xx + yy) / 2 * 255).astype(np.uint8)

    def create_radial_gradient(self, size: int) -> npt.NDArray[np.uint8]:
        """Create radial gradient."""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx**2 + yy**2)
        return ((1 - radius) * 255).clip(0, 255).astype(np.uint8)


class InpaintingDataset:
    """Dataset manager for inpainting experiments."""

    SYNTHETIC_IMAGES = {
        # Structure cases
        "lines": (ImageGenerator.create_line_image, ImageCategory.STRUCTURE),
        "shapes": (ImageGenerator.create_shape_image, ImageCategory.STRUCTURE),
        "curves": (ImageGenerator.create_curve_image, ImageCategory.STRUCTURE),
        "cross_lines": (ImageGenerator.create_cross_lines, ImageCategory.STRUCTURE),
        "diagonal": (ImageGenerator.create_diagonal_pattern, ImageCategory.STRUCTURE),
        # Texture cases
        "checkerboard": (ImageGenerator.create_checkerboard, ImageCategory.TEXTURE),
        "checker": (ImageGenerator.create_checker_pattern, ImageCategory.TEXTURE),
        "checkerboard_noisy": (ImageGenerator.create_textured_checker, ImageCategory.TEXTURE),
        "dots": (ImageGenerator.create_dot_pattern, ImageCategory.TEXTURE),
        "noise": (ImageGenerator.create_noise_pattern, ImageCategory.TEXTURE),
        # Gradient cases
        "linear_gradient": (ImageGenerator.create_linear_gradient, ImageCategory.GRADIENT),
        "radial_gradient": (ImageGenerator.create_radial_gradient, ImageCategory.GRADIENT),
    }

    def __init__(
        self, root_dir: Path | str, mask_config: MaskConfig | None = None, save_samples: bool = True
    ):
        """Initialize dataset manager."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.synthetic_dir = self.root_dir / "synthetic"
        self.synthetic_dir.mkdir(exist_ok=True)

        self.real_dir = self.root_dir / "real"
        self.real_dir.mkdir(exist_ok=True)

        self.mask_generator = MaskGenerator(mask_config or MaskConfig())
        self.image_generator = ImageGenerator()
        self.save_samples = save_samples

    def _apply_mask(
        self, image: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        """Apply mask to image, setting masked regions to NaN."""
        masked = image.astype(np.float32)
        masked[mask > 0] = np.nan
        return masked

    def generate_synthetic_dataset(
        self,
        size: int = 128,
        force_regenerate: bool = False,
        mask_types: List[str] = ["center", "random", "brush", "text"],
        selected_cases: list[str] | None = None,
    ) -> Dict[str, InpaintSample]:
        """Generate synthetic test cases."""
        dataset = {}

        cases_to_generate = (
            {k: v for k, v in self.SYNTHETIC_IMAGES.items() if k in selected_cases}
            if selected_cases
            else self.SYNTHETIC_IMAGES
        )

        for case_name, (gen_func, category) in cases_to_generate.items():
            case_dir = self.synthetic_dir / str(category.name).lower() / case_name
            if self.save_samples:
                case_dir.mkdir(parents=True, exist_ok=True)

            # Generate or load image
            image_path = case_dir / "image.png" if self.save_samples else None
            if not force_regenerate and image_path and image_path.exists():
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                logger.debug(f"Loaded existing {case_name} image")
            else:
                image = gen_func(self.image_generator, size)
                if self.save_samples:
                    cv2.imwrite(str(image_path), image)
                logger.debug(f"Generated {case_name} image")

            # Generate masks
            for mask_type in mask_types:
                mask_path = case_dir / f"mask_{mask_type}.png" if self.save_samples else None

                if not force_regenerate and mask_path and mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    logger.debug(f"Loaded existing {case_name} {mask_type} mask")
                else:
                    mask = self.mask_generator.generate(image, mask_type)
                    if self.save_samples:
                        cv2.imwrite(str(mask_path), mask)
                    logger.debug(f"Generated {case_name} {mask_type} mask")

                masked = self._apply_mask(image, mask)
                dataset[f"{case_name}_{mask_type}"] = InpaintSample(
                    original=image, masked=masked, mask=mask, category=category
                )

        return dataset

    def load_real_dataset(
        self,
        n_images: int = 50,
        size: int = 128,
        mask_types: List[str] = ["center", "random", "brush", "text"],
    ) -> Dict[str, InpaintSample]:
        """Load and prepare real images for inpainting.

        Args:
            n_images: Number of images to sample
            size: Size to resize images to
            mask_types: Types of masks to generate for each image

        Returns:
            Dictionary mapping case names to InpaintSample objects
        """
        transform = T.Compose([T.Resize((size, size)), T.Grayscale(), T.ToTensor()])

        dataset = CIFAR10(root=self.real_dir, train=True, transform=transform, download=True)
        indices = np.random.choice(len(dataset), n_images, replace=False)

        real_cases = {}
        for idx in indices:
            img_tensor, _ = dataset[idx]
            image = (img_tensor.numpy()[0] * 255).astype(np.uint8)

            case_name = f"real_{idx}"
            case_dir = self.real_dir / case_name
            if self.save_samples:
                case_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(case_dir / "image.png"), image)

            for mask_type in mask_types:
                mask = self.mask_generator.generate(image, mask_type)
                if self.save_samples:
                    cv2.imwrite(str(case_dir / f"mask_{mask_type}.png"), mask)

                masked = self._apply_mask(image, mask)
                real_cases[f"{case_name}_{mask_type}"] = InpaintSample(
                    original=image, masked=masked, mask=mask, category=ImageCategory.REAL
                )

            logger.debug(f"Processed {case_name}")

        return real_cases

    def load_custom_dataset(
        self,
        image_dir: Path | str = "test-images/",
        mask_dir: Path | str = "test-images/masks/",
        target_size: int = 256,
    ) -> dict[str, InpaintSample]:
        """Load custom image-mask pairs for inpainting.

        Images larger than target_size on their smallest dimension will be scaled down
        while preserving aspect ratio.

        Custom images are loaded in color, unlike other benchmark images.
        """
        masks = MaskGenerator.load_masks_from_directory(mask_dir)
        if not masks:
            raise ValueError(f"No masks found in {mask_dir}")

        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} does not exist")

        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(ext))

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        def resize_if_needed(image: np.ndarray, target_size: int) -> np.ndarray:
            """Resize image if its smallest dimension is larger than target_size."""
            height, width = image.shape[:2]
            min_dim = min(height, width)

            if min_dim > target_size:
                scale = target_size / min_dim
                new_height = int(height * scale)
                new_width = int(width * scale)
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return image

        samples = {}
        for image_file in image_files:
            # Read image in BGR format
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning(f"Could not read image file: {image_file}")
                continue

            # Convert to RGB and keep in [0, 255] range
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_if_needed(image, target_size)

            image_name = image_file.stem
            if image_name not in masks:
                logger.warning(f"No matching mask found for custom {image_name}")
                continue

            # Process mask and ensure it's in same range as image
            found_mask = masks[image_name]
            mask = found_mask.astype(np.uint8) * 255

            # Resize mask if needed
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            mask = mask.squeeze()

            # Create masked version (NaN in masked regions)
            masked = image.astype(np.float32)  # Convert to float32 for NaN support
            mask_bool = mask > 0
            for c in range(3):
                channel_masked = masked[..., c]
                channel_masked[mask_bool] = np.nan
                masked[..., c] = channel_masked

            assert len(mask.shape) == 2, f"Mask should be 2D, got shape {mask.shape}"

            samples[f"custom_{image_name}"] = InpaintSample(
                original=image,  # Keep original in [0, 255] range
                masked=masked,  # Contains NaN values
                mask=mask,  # In [0, 255] range
                category=ImageCategory.CUSTOM,
            )

            logger.debug(
                f"Processed custom pair '{image_name}' "
                f"(size: {image.shape}, mask coverage: {np.mean(mask > 0):.1%})"
            )

        logger.debug(f"Loaded {len(samples)} custom image-mask pairs")
        return samples

    def load_images_from_directory(
        self,
        image_dir: Path | str = "test-images/",
        target_size: tuple[int, int] | None = None,
        mask_types: List[str] = ["center", "random", "brush", "text"],
    ) -> Dict[str, InpaintSample]:
        """Load and prepare images from a directory for inpainting.

        Notes:
            - Supports common image formats (png, jpg, etc.)
            - Converts color images to grayscale
            - Automatically resizes images if target_size is provided
            - Generates specified mask types for each image
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} does not exist")

        # Find all image files
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(ext))

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        samples = {}
        for image_file in image_files:
            # Read and preprocess image
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Could not read image file: {image_file}")
                continue

            # Generate masks and create samples
            image_name = image_file.stem
            for mask_type in mask_types:
                mask = self.mask_generator.generate(image, mask_type)
                masked = self._apply_mask(image, mask)

                # Save samples if configured
                if self.save_samples:
                    case_dir = self.root_dir / "custom" / image_name
                    case_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(case_dir / "image.png"), image)
                    cv2.imwrite(str(case_dir / f"mask_{mask_type}.png"), mask)

                samples[f"{image_name}_{mask_type}"] = InpaintSample(
                    original=image,
                    masked=masked,
                    mask=mask,
                    category=ImageCategory.REAL,  # Custom images treated as real
                )

                logger.debug(
                    f"Processed '{image_name}' with {mask_type} mask "
                    f"(size: {image.shape}, mask coverage: {np.mean(mask > 0):.1%})"
                )

        logger.debug(f"Loaded and processed {len(image_files)} images from {image_dir}")
        return samples


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    from src.settings import DATA_PATH

    def plot_sample(sample: InpaintSample, title: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(sample.original, cmap="gray")
        axes[0].set_title(f"{title}\nOriginal")
        axes[0].axis("off")

        axes[1].imshow(sample.masked, cmap="gray")
        axes[1].set_title("Masked (NaN regions in white)")
        axes[1].axis("off")

        axes[2].imshow(sample.mask, cmap="gray")
        axes[2].set_title("Mask")
        axes[2].axis("off")

        plt.tight_layout()
        return fig

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Test different dataset functionalities")
    parser.add_argument(
        "test_type",
        type=str,
        choices=["synthetic", "custom", "all"],
        help="Type of test to run: synthetic, custom, or all",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Size of images to generate/resize to (default: 256)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save test outputs (default: DATA_PATH/test_images)",
    )
    args = parser.parse_args()

    # Setup save directory
    save_dir = Path(args.save_dir) if args.save_dir else DATA_PATH / "test_images"
    save_dir.mkdir(exist_ok=True)

    # Initialize dataset with common configuration
    dataset = InpaintingDataset(
        save_dir / "dataset",
        mask_config=MaskConfig(
            coverage_range=(0.1, 0.2), thickness_range=(2, 4), num_components=(2, 3)
        ),
        save_samples=True,
    )

    def test_synthetic():
        """Test synthetic image generation"""
        logger.debug("Testing synthetic image generation...")

        # Test individual image generation
        generator = ImageGenerator()

        for name, func in [
            ("Line", generator.create_line_image),
            ("Shape", generator.create_shape_image),
            ("Curve", generator.create_curve_image),
            ("Checker", generator.create_checker_image),
            ("Dots", generator.create_dot_pattern),
            ("Noise", generator.create_noise_pattern),
            ("Linear", generator.create_linear_gradient),
            ("Radial", generator.create_radial_gradient),
        ]:
            img = func(args.size)
            plt.figure(figsize=(4, 4))
            plt.imshow(img, cmap="gray")
            plt.title(f"{name} Pattern")
            plt.axis("off")
            plt.savefig(save_dir / f"test_{name.lower()}.png", bbox_inches="tight", dpi=150)
            plt.close()
            logger.debug(f"Generated {name} pattern")

        # Test full synthetic dataset generation
        synthetic_samples = dataset.generate_synthetic_dataset(
            size=args.size, force_regenerate=True, mask_types=["center", "random", "brush", "text"]
        )

        # Verify samples
        sample = next(iter(synthetic_samples.values()))
        assert np.any(np.isnan(sample.masked)), "Masked image should contain NaN values"
        assert not np.any(np.isnan(sample.original)), "Original image should not contain NaN values"

        logger.debug("Synthetic tests completed successfully!")

    def test_custom():
        """Test custom dataset loading"""
        logger.debug("Testing custom dataset loading with existing masks...")

        custom_samples = dataset.load_custom_dataset(
            image_dir="test-images",
            mask_dir="test-images/masks",
        )

        for case_name, sample in custom_samples.items():
            fig = plot_sample(sample, f"Custom: {case_name}")
            fig.savefig(save_dir / f"test_custom_{case_name}.png", bbox_inches="tight", dpi=300)
            plt.close()
            logger.debug(f"Generated visualization for custom sample {case_name}")

        logger.debug("Custom dataset tests completed successfully!")

    # Run selected tests
    if args.test_type == "synthetic" or args.test_type == "all":
        test_synthetic()

    if args.test_type == "custom" or args.test_type == "all":
        test_custom()
