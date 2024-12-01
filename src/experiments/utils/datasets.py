from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms as T
from loguru import logger
from torchvision.datasets import CIFAR10


class InpaintingDataset:
    """Dataset manager for inpainting experiments.

    Handles creation and loading of both synthetic test cases and real image datasets.
    Each test case consists of an image and one or more masks.
    """

    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different dataset types
        self.synthetic_dir = self.root_dir / "synthetic"
        self.synthetic_dir.mkdir(exist_ok=True)

        self.real_dir = self.root_dir / "real"
        self.real_dir.mkdir(exist_ok=True)

    def generate_synthetic_dataset(
        self,
        size: int = 128,  # Size of synthetic images (square)
        force_regenerate: bool = False,  # If True, regenerate even if files exist
    ) -> dict[str, dict]:
        """Generate or load synthetic test cases."""
        test_cases = {
            # Structure cases
            "lines": {
                "func": self._create_line_image,
                "masks": ["center", "random", "brush"],
                "category": "structure",
            },
            "shapes": {
                "func": self._create_shape_image,
                "masks": ["center", "random", "brush"],
                "category": "structure",
            },
            "curves": {
                "func": self._create_curve_image,
                "masks": ["center", "random", "brush"],
                "category": "structure",
            },
            # Texture cases
            "checkerboard": {
                "func": self._create_checker_image,
                "masks": ["center", "random", "brush"],
                "category": "texture",
            },
            "dots": {
                "func": self._create_dot_pattern,
                "masks": ["center", "random", "brush"],
                "category": "texture",
            },
            "noise": {
                "func": self._create_noise_pattern,
                "masks": ["center", "random", "brush"],
                "category": "texture",
            },
            # Gradient cases
            "linear_gradient": {
                "func": self._create_linear_gradient,
                "masks": ["center", "random", "brush"],
                "category": "gradient",
            },
            "radial_gradient": {
                "func": self._create_radial_gradient,
                "masks": ["center", "random", "brush"],
                "category": "gradient",
            },
        }

        dataset = {}

        for case_name, case_info in test_cases.items():
            case_dir = self.synthetic_dir / case_info["category"] / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            image_path = case_dir / "image.png"

            # Generate or load image
            if not image_path.exists() or force_regenerate:
                image = case_info["func"](size)
                cv2.imwrite(str(image_path), image)
                logger.info(f"Generated {case_name} image")
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                logger.info(f"Loaded existing {case_name} image")

            # Generate or load masks
            masks = {}
            for mask_type in case_info["masks"]:
                mask_path = case_dir / f"mask_{mask_type}.png"

                if not mask_path.exists() or force_regenerate:
                    mask = self._create_mask(image, mask_type)
                    cv2.imwrite(str(mask_path), mask)
                    logger.info(f"Generated {case_name} {mask_type} mask")
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    logger.info(f"Loaded existing {case_name} {mask_type} mask")

                masks[mask_type] = mask

            dataset[case_name] = {"image": image, "masks": masks, "category": case_info["category"]}

        return dataset

    def _create_mask(self, image: np.ndarray, mask_type: str) -> np.ndarray:
        """Create mask of specified type."""
        if mask_type == "center":
            return self._create_center_mask(image)
        elif mask_type == "random":
            return self._create_random_mask(image)
        elif mask_type == "brush":
            return self._create_brush_mask(image)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def _create_center_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a rectangular mask in the center."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Center rectangle covering ~25% of image
        y1, y2 = h // 3, 2 * h // 3
        x1, x2 = w // 3, 2 * w // 3
        mask[y1:y2, x1:x2] = 255

        return mask

    def _create_random_mask(self, image: np.ndarray) -> np.ndarray:
        """Create random elliptical masks.

        Ensures masks cover between 10-40% of the image area with reasonable distribution.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Target total coverage between 10-40%
        target_coverage = np.random.uniform(0.15, 0.35)  # Increased minimum
        total_area = h * w
        current_coverage = 0

        # Add 3-5 random ellipses for better coverage
        n_holes = np.random.randint(3, 6)
        max_attempts = 30  # More attempts to achieve target coverage

        for _ in range(n_holes):
            attempts = 0
            while attempts < max_attempts:
                # Generate center with padding from edges
                pad = max(w, h) // 8
                center = (np.random.randint(pad, w - pad), np.random.randint(pad, h - pad))

                # Make axes relative to image size but not too small
                max_axis = min(w, h) // 4  # Increased maximum size
                min_axis = max_axis // 3  # Ensure minimum size
                axes = (
                    np.random.randint(min_axis, max_axis),
                    np.random.randint(min_axis, max_axis),
                )

                # Create temporary mask to test coverage
                temp_mask = mask.copy()
                angle = np.random.randint(0, 180)
                cv2.ellipse(temp_mask, center, axes, angle, 0, 360, 255, -1)

                # Calculate new coverage
                new_coverage = np.sum(temp_mask > 0) / total_area

                # Accept if within reasonable bounds
                if new_coverage <= target_coverage:
                    mask = temp_mask
                    current_coverage = new_coverage
                    break

                attempts += 1

            if current_coverage >= target_coverage:
                break

        return mask

    def _create_brush_mask(self, image: np.ndarray) -> np.ndarray:
        """Create brush-stroke like mask using Bezier curves.

        Ensures strokes are reasonably sized and cover 15-35% of the image.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        total_area = h * w
        target_coverage = np.random.uniform(0.2, 0.4)  # Increased coverage targets
        current_coverage = 0

        n_strokes = np.random.randint(3, 5)  # More strokes
        max_attempts = 25  # More attempts to achieve coverage

        for _ in range(n_strokes):
            attempts = 0
            while attempts < max_attempts and current_coverage < target_coverage:
                # Create temporary mask for testing
                temp_mask = mask.copy()

                # Generate better-spaced control points
                x_start = np.random.randint(w // 4, 3 * w // 4)
                y_start = np.random.randint(h // 4, 3 * h // 4)

                # Ensure manageable stroke length even for small images
                max_offset = max(2, min(w, h) // 4)  # At least 2 pixels

                # Ensure points stay within image bounds even with tiny images
                x_mid = x_start + np.random.randint(-max_offset, max_offset + 1)
                y_mid = y_start + np.random.randint(-max_offset, max_offset + 1)

                x_end = x_mid + np.random.randint(-max_offset, max_offset + 1)
                y_end = y_mid + np.random.randint(-max_offset, max_offset + 1)

                points = np.array([[x_start, y_start], [x_mid, y_mid], [x_end, y_end]])
                points = np.clip(points, 0, [w - 1, h - 1])

                # Generate points along the curve
                t = np.linspace(0, 1, 50)  # Fewer points for more controlled strokes
                curve_points = []

                for t_val in t:
                    point = (
                        (1 - t_val) ** 2 * points[0]
                        + 2 * (1 - t_val) * t_val * points[1]
                        + t_val**2 * points[2]
                    ).astype(np.int32)
                    curve_points.append(point)

                # Scale thickness relative to image size but with minimum and maximum bounds
                min_thickness = max(
                    2, min(w // 16, h // 16)
                )  # At least 2 pixels, at most 1/16th of size
                max_thickness = max(
                    min_thickness + 1, min(w // 8, h // 8)
                )  # At least 1 more than min, at most 1/8th
                thickness = np.random.randint(min_thickness, max_thickness)

                # Ensure we don't get zero-sized or invalid strokes
                if max_thickness <= min_thickness:
                    thickness = min_thickness

                for i in range(len(curve_points) - 1):
                    pt1 = tuple(curve_points[i])
                    pt2 = tuple(curve_points[i + 1])
                    cv2.line(temp_mask, pt1, pt2, 255, thickness)

                # Check coverage
                new_coverage = np.sum(temp_mask > 0) / total_area

                if new_coverage <= target_coverage:
                    mask = temp_mask
                    current_coverage = new_coverage
                    break

                attempts += 1

        return mask

    # Image generation functions
    def _create_line_image(self, size: int) -> np.ndarray:
        """Create test image with straight lines."""
        image = np.zeros((size, size), dtype=np.uint8)

        # Add horizontal, vertical and diagonal lines
        cv2.line(image, (size // 4, size // 4), (3 * size // 4, size // 4), 255, 2)
        cv2.line(image, (size // 4, size // 4), (size // 4, 3 * size // 4), 255, 2)
        cv2.line(image, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, 2)

        return image

    def _create_shape_image(self, size: int) -> np.ndarray:
        """Create test image with basic shapes."""
        image = np.zeros((size, size), dtype=np.uint8)

        # Add rectangle, circle and triangle
        cv2.rectangle(image, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, 2)
        cv2.circle(image, (size // 2, size // 2), size // 4, 255, 2)

        triangle_pts = np.array(
            [[size // 2, size // 4], [size // 4, 3 * size // 4], [3 * size // 4, 3 * size // 4]]
        )
        cv2.polylines(image, [triangle_pts], True, 255, 2)

        return image

    def _create_curve_image(self, size: int) -> np.ndarray:
        """Create test image with curves."""
        image = np.zeros((size, size), dtype=np.uint8)

        # Create sine wave
        x = np.linspace(0, 4 * np.pi, size)
        y = np.sin(x) * size // 4 + size // 2
        points = np.column_stack((np.linspace(0, size - 1, size), y))
        points = points.astype(np.int32)  # Convert to int32

        # Draw the curve
        for i in range(len(points) - 1):
            pt1 = tuple(map(int, points[i]))  # Explicitly convert to int tuple
            pt2 = tuple(map(int, points[i + 1]))
            cv2.line(image, pt1, pt2, 255, 2)

        return image

    def _create_checker_image(self, size: int) -> np.ndarray:
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

    def _create_dot_pattern(self, size: int) -> np.ndarray:
        """Create regular dot pattern."""
        image = np.zeros((size, size), dtype=np.uint8)
        spacing = size // 8

        for i in range(spacing, size - spacing, spacing):
            for j in range(spacing, size - spacing, spacing):
                cv2.circle(image, (i, j), 3, 255, -1)

        return image

    def _create_noise_pattern(self, size: int) -> np.ndarray:
        """Create structured noise pattern."""
        # Start with random noise
        noise = np.random.randint(0, 256, (size // 4, size // 4), dtype=np.uint8)
        # Upscale to create structured pattern
        image = cv2.resize(noise, (size, size), interpolation=cv2.INTER_NEAREST)
        return image

    def _create_linear_gradient(self, size: int) -> np.ndarray:
        """Create linear gradient."""
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        gradient = ((xx + yy) / 2 * 255).astype(np.uint8)
        return gradient

    def _create_radial_gradient(self, size: int) -> np.ndarray:
        """Create radial gradient."""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx**2 + yy**2)
        gradient = ((1 - radius) * 255).clip(0, 255).astype(np.uint8)
        return gradient

    def load_real_dataset(
        self,
        n_images: int = 50,
        size: int = 128,
    ) -> dict[str, dict]:
        """Load real images from CIFAR-10 and prepare them for inpainting."""
        # Setup transforms
        transform = T.Compose([T.Resize((size, size)), T.Grayscale(), T.ToTensor()])

        # Load dataset (download automatically if needed)
        dataset = CIFAR10(root=self.real_dir, train=True, transform=transform, download=True)

        # Sample n_images randomly
        indices = np.random.choice(len(dataset), n_images, replace=False)

        real_cases = {}
        for idx in indices:
            img_tensor, _ = dataset[idx]
            # Convert to numpy array and scale to [0, 255]
            image = (img_tensor.numpy()[0] * 255).astype(np.uint8)

            # Generate masks
            masks = {
                "center": self._create_center_mask(image),
                "random": self._create_random_mask(image),
                "brush": self._create_brush_mask(image),
            }

            case_name = f"real_{idx}"
            real_cases[case_name] = {"image": image, "masks": masks, "category": "real"}

            # Save images and masks
            case_dir = self.real_dir / case_name
            case_dir.mkdir(exist_ok=True)

            cv2.imwrite(str(case_dir / "image.png"), image)
            for mask_type, mask in masks.items():
                cv2.imwrite(str(case_dir / f"mask_{mask_type}.png"), mask)

            logger.info(f"Processed and saved {case_name}")

        return real_cases


def validate_mask_coverage(
    mask: np.ndarray, name: str, min_coverage: float = 0.05, max_coverage: float = 0.5
) -> None:
    """Validate mask coverage and distribution."""
    coverage = np.mean(mask > 0)
    logger.info(f"{name} mask coverage: {coverage:.1%}")

    if coverage < min_coverage:
        raise ValueError(f"{name} mask coverage too low: {coverage:.1%}")
    if coverage > max_coverage:
        raise ValueError(f"{name} mask coverage too high: {coverage:.1%}")

    # Check for connectivity - masks shouldn't be too fragmented
    if name != "random":  # Skip for random masks which are intentionally separate
        labeled, num_features = ndimage.label(mask)
        if num_features > 5:
            logger.warning(f"{name} mask might be too fragmented: {num_features} separate regions")


def visualize_mask_distribution(
    dataset: InpaintingDataset, size: int = 128, num_samples: int = 10
) -> None:
    """Generate and visualize multiple masks to check distribution."""
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
    mask_types = ["random", "brush", "center"]

    coverage_stats = {mask_type: [] for mask_type in mask_types}

    for i in range(num_samples):
        img = np.zeros((size, size), dtype=np.uint8)

        for j, mask_type in enumerate(mask_types):
            mask = dataset._create_mask(img, mask_type)
            coverage = np.mean(mask > 0)
            coverage_stats[mask_type].append(coverage)

            axes[j, i].imshow(mask, cmap="gray")
            axes[j, i].axis("off")
            if i == 0:
                axes[j, i].set_ylabel(mask_type)

    plt.tight_layout()

    for mask_type in mask_types:
        coverages = coverage_stats[mask_type]
        logger.info(f"\n{mask_type} mask statistics:")
        logger.info(f"Mean coverage: {np.mean(coverages):.1%}")
        logger.info(f"Std coverage: {np.std(coverages):.1%}")
        logger.info(f"Min coverage: {np.min(coverages):.1%}")
        logger.info(f"Max coverage: {np.max(coverages):.1%}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import ndimage

    dataset = InpaintingDataset(Path("data/datasets"))

    # Generate synthetic dataset
    synthetic_cases = dataset.generate_synthetic_dataset(size=128, force_regenerate=True)

    # Load real dataset
    real_cases = dataset.load_real_dataset(n_images=10, size=128)

    # Combine all cases
    test_cases = {**synthetic_cases, **real_cases}

    def plot_case(image: np.ndarray, masks: dict[str, np.ndarray], title: str) -> None:
        n_masks = len(masks)
        fig, axes = plt.subplots(1, n_masks + 1, figsize=(4 * (n_masks + 1), 4))

        # Plot original image
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"{title}\nOriginal")
        axes[0].axis("off")

        # Plot masks
        for i, (mask_type, mask) in enumerate(masks.items(), 1):
            axes[i].imshow(mask, cmap="gray")
            axes[i].set_title(f"Mask ({mask_type})")
            axes[i].axis("off")

        plt.tight_layout()
        return fig

    # Plot one example from each category
    categories = {
        "structure": ["lines", "shapes", "curves"],
        "texture": ["checkerboard", "dots", "noise"],
        "gradient": ["linear_gradient", "radial_gradient"],
    }

    save_dir = Path("data/dataset_examples")
    save_dir.mkdir(parents=True, exist_ok=True)

    for category, cases in categories.items():
        for case in cases:
            if case in test_cases:
                data = test_cases[case]
                fig = plot_case(data["image"], data["masks"], f"{category}: {case}")
                fig.savefig(save_dir / f"{case}.png", bbox_inches="tight", dpi=150)
                plt.close(fig)

    logger.info("Generated and saved example visualizations to data/dataset_examples/")

    # If you want to run validation tests, uncomment these lines:
    test_image = np.zeros((16, 16), dtype=np.uint8)
    for mask_type in ["random", "brush", "center"]:
        mask = dataset._create_mask(test_image, mask_type)
        validate_mask_coverage(mask, mask_type)
    visualize_mask_distribution(dataset, size=16)
    plt.show()
