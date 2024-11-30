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
        self, size: int = 128, force_regenerate: bool = False
    ) -> dict[str, dict]:
        """Generate or load synthetic test cases.

        Args:
            size: Size of synthetic images (square)
            force_regenerate: If True, regenerate even if files exist

        Returns:
            Dictionary of test cases, each containing 'image' and 'masks'
        """
        test_cases = {
            # Structure cases
            "lines": {
                "func": self._create_line_image,
                "masks": ["center", "random"],
                "category": "structure",
            },
            "shapes": {
                "func": self._create_shape_image,
                "masks": ["center", "random", "brush"],
                "category": "structure",
            },
            "curves": {
                "func": self._create_curve_image,
                "masks": ["random", "brush"],
                "category": "structure",
            },
            # Texture cases
            "checkerboard": {
                "func": self._create_checker_image,
                "masks": ["center", "random"],
                "category": "texture",
            },
            "dots": {
                "func": self._create_dot_pattern,
                "masks": ["center", "random"],
                "category": "texture",
            },
            "noise": {
                "func": self._create_noise_pattern,
                "masks": ["center", "brush"],
                "category": "texture",
            },
            # Gradient cases
            "linear_gradient": {
                "func": self._create_linear_gradient,
                "masks": ["center", "random"],
                "category": "gradient",
            },
            "radial_gradient": {
                "func": self._create_radial_gradient,
                "masks": ["center", "brush"],
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
        """Create random elliptical masks."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Add 3-6 random ellipses
        n_holes = np.random.randint(3, 7)
        for _ in range(n_holes):
            # Generate center away from edges
            center = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))
            # Make axes relative to image size
            axes = (np.random.randint(w // 8, w // 4), np.random.randint(h // 8, h // 4))
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        return mask

    def _create_brush_mask(self, image: np.ndarray) -> np.ndarray:
        """Create brush-stroke like mask using Bezier curves."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        n_strokes = np.random.randint(2, 5)

        for _ in range(n_strokes):
            # Generate better-spaced control points
            x_start = np.random.randint(w // 4, 3 * w // 4)
            y_start = np.random.randint(h // 4, 3 * h // 4)

            # Ensure second point is some distance away
            x_mid = x_start + np.random.randint(-w // 3, w // 3)
            y_mid = y_start + np.random.randint(-h // 3, h // 3)

            # Ensure end point creates a meaningful curve
            x_end = x_mid + np.random.randint(-w // 3, w // 3)
            y_end = y_mid + np.random.randint(-h // 3, h // 3)

            points = np.array([[x_start, y_start], [x_mid, y_mid], [x_end, y_end]])

            # Clip points to ensure they're within image bounds
            points = np.clip(points, 0, [w - 1, h - 1])

            # Generate points along the curve
            t = np.linspace(0, 1, 100)
            curve_points = []

            for t_val in t:
                point = (
                    (1 - t_val) ** 2 * points[0]
                    + 2 * (1 - t_val) * t_val * points[1]
                    + t_val**2 * points[2]
                ).astype(np.int32)
                curve_points.append(point)

            # Draw the curve with larger thickness variation
            for i in range(len(curve_points) - 1):
                thickness = np.random.randint(w // 16, w // 8)  # Relative to image size
                pt1 = tuple(curve_points[i])
                pt2 = tuple(curve_points[i + 1])
                cv2.line(mask, pt1, pt2, 255, thickness)

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


if __name__ == "__main__":
    dataset = InpaintingDataset(Path("data/datasets"))

    # Generate synthetic dataset
    synthetic_cases = dataset.generate_synthetic_dataset(size=128, force_regenerate=True)

    # Load real dataset
    real_cases = dataset.load_real_dataset(n_images=10, size=128)

    # Combine all cases
    test_cases = {**synthetic_cases, **real_cases}

    import matplotlib.pyplot as plt

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
