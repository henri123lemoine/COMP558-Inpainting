from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.algorithms.base import InpaintingAlgorithm
from src.experiments.base import InpaintingExperiment
from src.experiments.utils.visualization import plot_multiple_results


class ComparisonExperiment(InpaintingExperiment):
    """Experiment for comparing multiple inpainting algorithms."""

    def __init__(
        self,
        name: str,
        algorithms: list[InpaintingAlgorithm],
        dataset_dir: str | Path | None = None,
    ):
        super().__init__(name, algorithms[0], dataset_dir)
        self.algorithms = algorithms
        logger.info(
            f"Initialized comparison of {len(algorithms)} algorithms: "
            f"{', '.join(alg.name for alg in algorithms)}"
        )

    def _create_test_dataset(self) -> None:
        """Create test dataset if it doesn't exist."""
        dataset_dir = self.dataset_dir / "comparison_test"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # List of test cases to create
        test_cases = {
            "simple_gradient": self._create_gradient_image,
            "texture_pattern": self._create_texture_image,
            "structure_test": self._create_structure_image,
        }

        # Different mask types
        mask_types = {
            "center": self._create_center_mask,
            "random": self._create_random_mask,
            "brush": self._create_brush_mask,
        }

        # Create test cases and masks
        for case_name, image_func in test_cases.items():
            for mask_name, mask_func in mask_types.items():
                image_path = dataset_dir / f"{case_name}_{mask_name}.png"
                mask_path = dataset_dir / f"{case_name}_{mask_name}_mask.png"

                if not image_path.exists() or not mask_path.exists():
                    logger.info(f"Creating test case: {case_name} with {mask_name} mask")
                    image = image_func(128)  # Fixed size for test images
                    mask = mask_func(image)

                    cv2.imwrite(str(image_path), image)
                    cv2.imwrite(str(mask_path), mask)

    def _create_gradient_image(self, size: int) -> np.ndarray:
        """Create a simple gradient test image."""
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        image = ((xx + yy) / 2 * 255).astype(np.uint8)
        return image

    def _create_texture_image(self, size: int) -> np.ndarray:
        """Create a textured test image."""
        image = np.zeros((size, size), dtype=np.uint8)

        # Create checkerboard pattern
        square_size = 8
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i + j) // square_size % 2 == 0:
                    image[i : i + square_size, j : j + square_size] = 255

        # Add some noise for texture
        noise = np.random.normal(0, 20, (size, size))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return image

    def _create_structure_image(self, size: int) -> np.ndarray:
        """Create a test image with clear structures."""
        image = np.zeros((size, size), dtype=np.uint8)

        # Add some lines
        cv2.line(image, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), 255, 2)
        cv2.line(image, (size // 4, 3 * size // 4), (3 * size // 4, size // 4), 255, 2)

        # Add a circle
        cv2.circle(image, (size // 2, size // 2), size // 4, 255, 2)

        return image

    def _create_center_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mask in the center of the image."""
        h, w = image.shape[:2]
        mask = np.zeros_like(image)
        y1, y2 = h // 3, 2 * h // 3
        x1, x2 = w // 3, 2 * w // 3
        mask[y1:y2, x1:x2] = 255
        return mask

    def _create_random_mask(self, image: np.ndarray) -> np.ndarray:
        """Create random masks."""
        mask = np.zeros_like(image)
        n_holes = np.random.randint(3, 7)
        h, w = image.shape[:2]

        for _ in range(n_holes):
            # Random ellipse parameters
            center = (np.random.randint(0, w), np.random.randint(0, h))
            axes = (np.random.randint(10, 30), np.random.randint(10, 30))
            angle = np.random.randint(0, 180)

            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        return mask

    def _create_brush_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a brush-stroke like mask."""
        mask = np.zeros_like(image)
        h, w = image.shape[:2]
        n_strokes = np.random.randint(2, 5)

        for _ in range(n_strokes):
            # Create random control points for the curve
            control_points = np.array(
                [
                    [np.random.randint(0, w), np.random.randint(0, h)],
                    [np.random.randint(0, w), np.random.randint(0, h)],
                    [np.random.randint(0, w), np.random.randint(0, h)],
                ]
            )

            # Generate points along the bezier curve
            t = np.linspace(0, 1, 100)
            curve_points = []
            for t_val in t:
                # Quadratic Bezier curve formula
                point = (
                    (1 - t_val) ** 2 * control_points[0]
                    + 2 * (1 - t_val) * t_val * control_points[1]
                    + t_val**2 * control_points[2]
                ).astype(np.int32)
                curve_points.append(point)

            # Draw the curve with varying thickness
            for i in range(len(curve_points) - 1):
                thickness = np.random.randint(5, 15)
                pt1 = tuple(map(int, curve_points[i]))
                pt2 = tuple(map(int, curve_points[i + 1]))
                cv2.line(mask, pt1, pt2, 255, thickness)

        return mask

    def _load_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Load or create comparison test dataset."""
        # Create test dataset if needed
        self._create_test_dataset()

        dataset_dir = self.dataset_dir / "comparison_test"
        dataset = {}

        # Load all image-mask pairs
        for image_path in dataset_dir.glob("*.png"):
            if "_mask" not in image_path.stem:
                mask_path = dataset_dir / f"{image_path.stem}_mask.png"
                if mask_path.exists():
                    image, mask = self.algorithms[0].load_image(
                        image_path, mask_path, grayscale=True
                    )
                    dataset[image_path.stem] = {"image": image, "mask": mask}

        return dataset

    def run(self, save_visualizations: bool = True) -> None:
        """Run the comparison experiment."""
        logger.info(f"Running comparison experiment: {self.name}")

        # Load dataset
        dataset = self.load_dataset()
        logger.info(f"Loaded dataset with {len(dataset)} test cases")

        # Process each test case
        for image_name, data in dataset.items():
            logger.info(f"Processing test case: {image_name}")

            # Store results for each algorithm
            results_dict = {}

            # Run each algorithm
            for algorithm in self.algorithms:
                logger.info(f"Running {algorithm.name}")
                result = algorithm.inpaint(data["image"], data["mask"])

                # Save individual result
                output_path = self.output_dir / f"{image_name}_{algorithm.name}_result.png"
                algorithm.save_result(result, output_path)

                # Store for comparison visualization
                results_dict[algorithm.name] = {
                    "original": data["image"],
                    "mask": data["mask"],
                    "result": result,
                }

            # Save comparison visualization
            if save_visualizations:
                vis_path = self.output_dir / f"{image_name}_comparison.png"
                plot_multiple_results(results_dict, save_path=vis_path)

        logger.info("Comparison experiment completed")
