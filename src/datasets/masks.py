from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from .utils import ImageCategory


@dataclass
class MaskConfig:
    """Configuration for mask generation."""

    coverage_range: tuple[float, float] = (0.05, 0.15)
    thickness_range: tuple[int, int] | None = (1, 2)
    num_components: tuple[int, int] | None = (3, 5)


@dataclass
class InpaintSample:
    """Single sample for inpainting tasks."""

    original: npt.NDArray[np.uint8]
    masked: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    category: ImageCategory


class MaskGenerator:
    """Handles generation of various mask types for inpainting."""

    def __init__(self, config: MaskConfig):
        self.config = config

    def generate(
        self, image: npt.NDArray[np.uint8], mask_type: str, **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Generate mask of specified type with given parameters."""
        generator_map = {
            "center": self._create_center_mask,
            "random": self._create_random_mask,
            "brush": self._create_brush_mask,
            "text": self._create_text_mask,
        }

        if mask_type not in generator_map:
            raise ValueError(f"Unknown mask type: {mask_type}")

        return generator_map[mask_type](image, **kwargs)

    def _create_center_mask(
        self,
        image: npt.NDArray[np.uint8],
        width_scale: float = 0.2,
        height_scale: float = 0.2,
    ) -> npt.NDArray[np.uint8]:
        """Create a rectangular mask in the center with configurable size."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        y_size = int(h * height_scale)
        x_size = int(w * width_scale)

        y1 = (h - y_size) // 2
        y2 = y1 + y_size
        x1 = (w - x_size) // 2
        x2 = x1 + x_size

        mask[y1:y2, x1:x2] = 255
        return mask

    def _create_random_mask(
        self,
        image: npt.NDArray[np.uint8],
        n_components: int | None = None,
        min_size: float = 0.05,
    ) -> npt.NDArray[np.uint8]:
        """Create random elliptical masks with configurable parameters."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if n_components is None:
            n_components = np.random.randint(
                *self.config.num_components if self.config.num_components else (3, 6)
            )

        min_size = int(min(h, w) * min_size)
        max_size = int(min(h, w) * 0.15)  # Max 15% of image size

        target_coverage = np.random.uniform(*self.config.coverage_range)
        total_area = h * w
        current_coverage = 0

        for _ in range(n_components):
            if current_coverage >= target_coverage:
                break

            center = (
                np.random.randint(min_size, w - min_size),
                np.random.randint(min_size, h - min_size),
            )
            axes = (np.random.randint(min_size, max_size), np.random.randint(min_size, max_size))
            angle = np.random.randint(0, 180)

            temp_mask = mask.copy()
            cv2.ellipse(temp_mask, center, axes, angle, 0, 360, 255, -1)

            new_coverage = np.sum(temp_mask > 0) / total_area
            if new_coverage <= target_coverage:
                mask = temp_mask
                current_coverage = new_coverage

        return mask

    def _create_brush_mask(
        self,
        image: npt.NDArray[np.uint8],
        n_strokes: int | None = None,
        stroke_width: int | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Create brush-stroke mask with configurable parameters."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if n_strokes is None:
            n_strokes = np.random.randint(
                *self.config.num_components if self.config.num_components else (3, 5)
            )

        if stroke_width is None and self.config.thickness_range:
            stroke_width = np.random.randint(*self.config.thickness_range)
        elif stroke_width is None:
            stroke_width = 1

        target_coverage = np.random.uniform(*self.config.coverage_range)
        total_area = h * w
        current_coverage = 0

        for _ in range(n_strokes):
            if current_coverage >= target_coverage:
                break

            points = self._generate_bezier_points(w, h)
            temp_mask = mask.copy()

            for i in range(len(points) - 1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i + 1]))
                cv2.line(temp_mask, pt1, pt2, 255, stroke_width)

            new_coverage = np.sum(temp_mask > 0) / total_area
            if new_coverage <= target_coverage:
                mask = temp_mask
                current_coverage = new_coverage

        return mask

    def _create_text_mask(
        self,
        image: npt.NDArray[np.uint8],
        text: str = "COMP558 IS THE BEST CLASS",
        scale: float = 0.8,
        thickness: int = 1,
    ) -> npt.NDArray[np.uint8]:
        """Create mask from text, automatically wrapping to fit image width."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = scale * min(w, h) / 500  # Scale relative to image size

        # First measure each word
        words = text.split()
        word_sizes = [cv2.getTextSize(word, font, font_scale, thickness)[0] for word in words]

        # Build lines that fit within the image width
        lines = []
        current_line = []
        current_width = 0
        space_width = cv2.getTextSize(" ", font, font_scale, thickness)[0][0]

        for word, (word_width, _) in zip(words, word_sizes):
            # Check if adding this word would exceed image width
            if current_line and current_width + space_width + word_width > w * 0.9:  # 90% of width
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                if current_line:
                    current_width += space_width + word_width
                else:
                    current_width = word_width
                current_line.append(word)

        if current_line:
            lines.append(" ".join(current_line))

        # Calculate total height needed
        line_height = cv2.getTextSize("X", font, font_scale, thickness)[0][1]
        line_spacing = line_height + 10  # Add some padding between lines
        total_height = line_spacing * len(lines)

        # Start position (centered vertically)
        y_start = (h - total_height) // 2 + line_height

        # Draw each line
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x = (w - text_size[0]) // 2  # Center horizontally
            y = y_start + i * line_spacing
            cv2.putText(mask, line, (x, y), font, font_scale, 255, thickness)

        return mask

    ## Helpers ##

    def _generate_bezier_points(
        self, w: int, h: int, n_points: int = 50
    ) -> list[npt.NDArray[np.int32]]:
        """Generate points along a Bezier curve for brush strokes."""
        x_start = np.random.randint(w // 4, 3 * w // 4)
        y_start = np.random.randint(h // 4, 3 * h // 4)

        max_offset = min(w, h) // 4
        x_mid = x_start + np.random.randint(-max_offset, max_offset + 1)
        y_mid = y_start + np.random.randint(-max_offset, max_offset + 1)

        x_end = x_mid + np.random.randint(-max_offset, max_offset + 1)
        y_end = y_mid + np.random.randint(-max_offset, max_offset + 1)

        points = np.array([[x_start, y_start], [x_mid, y_mid], [x_end, y_end]])
        points = np.clip(points, 0, [w - 1, h - 1])

        t = np.linspace(0, 1, n_points)
        curve_points = []

        for t_val in t:
            point = (
                (1 - t_val) ** 2 * points[0]
                + 2 * (1 - t_val) * t_val * points[1]
                + t_val**2 * points[2]
            ).astype(np.int32)
            curve_points.append(point)

        return curve_points

    @staticmethod
    def load_masks_from_directory(
        mask_dir: Path | str = "test-images/masks/",
        target_size: tuple[int, int] | None = None,
        threshold: int = 127,
        recursive: bool = True,
    ) -> dict[str, npt.NDArray[np.bool_]]:
        """Load mask images from a directory.

        Notes:
            - Supports common image formats (png, jpg, etc.)
            - Converts color images to grayscale
            - Thresholds grayscale images to create binary masks
            - Automatically resizes masks if target_size is provided
        """
        mask_dir = Path(mask_dir)
        if not mask_dir.exists():
            raise ValueError(f"Mask directory {mask_dir} does not exist")

        # Find all image files
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        mask_files = []
        for ext in extensions:
            if recursive:
                mask_files.extend(mask_dir.rglob(ext))
            else:
                mask_files.extend(mask_dir.glob(ext))

        if not mask_files:
            raise ValueError(f"No mask images found in {mask_dir}")

        masks = {}
        for mask_file in mask_files:
            # Read image in grayscale
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not read mask file: {mask_file}")
                continue

            # Resize if needed
            if target_size is not None:
                mask = cv2.resize(mask, (target_size[1], target_size[0]))  # cv2 uses (w,h)

            # Threshold to create binary mask
            _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

            # Convert to boolean
            mask = mask.astype(bool)

            # Use stem (filename without extension) as key
            mask_name = mask_file.stem
            masks[mask_name] = mask

            logger.debug(
                f"Loaded mask '{mask_name}' from {mask_file.name} "
                f"(size: {mask.shape}, coverage: {np.mean(mask):.1%})"
            )

        logger.debug(f"Loaded {len(masks)} masks from {mask_dir}")
        return masks


if __name__ == "__main__":
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage

    # Test mask generation with different configurations
    test_configs = [
        MaskConfig(coverage_range=(0.1, 0.2), thickness_range=(2, 4), num_components=(2, 3)),
        MaskConfig(coverage_range=(0.3, 0.4), thickness_range=(4, 8), num_components=(4, 6)),
    ]

    def plot_masks(
        image: npt.NDArray[np.uint8], masks: list[tuple[str, npt.NDArray[np.uint8]]], title: str
    ) -> None:
        fig, axes = plt.subplots(1, len(masks), figsize=(4 * len(masks), 4))
        if len(masks) == 1:
            axes = [axes]

        for ax, (name, mask) in zip(axes, masks):
            ax.imshow(mask, cmap="gray")
            ax.set_title(f"{title}\n{name}")
            ax.axis("off")

        plt.tight_layout()
        return fig

    # Test each configuration
    test_image = np.zeros((128, 128), dtype=np.uint8)

    for i, config in enumerate(test_configs):
        mask_gen = MaskGenerator(config)
        masks = []

        # Generate all mask types
        for mask_type in ["center", "random", "brush", "text"]:
            mask = mask_gen.generate(test_image, mask_type)
            coverage = np.mean(mask > 0)
            logger.debug(f"Config {i+1}, {mask_type} mask coverage: {coverage:.1%}")

            if mask_type != "center":  # Skip connectivity check for center mask
                labeled, num_features = ndimage.label(mask)
                logger.debug(f"Config {i+1}, {mask_type} mask components: {num_features}")

            masks.append((mask_type, mask))

        # Plot results
        fig = plot_masks(test_image, masks, f"Config {i+1}")
        plt.show()

    # Test special cases
    mask_gen = MaskGenerator(MaskConfig())

    # Test text mask with different parameters
    text_masks = [
        ("Default", mask_gen.generate(test_image, "text")),
        ("Large", mask_gen.generate(test_image, "text", scale=2.0)),
        ("Custom", mask_gen.generate(test_image, "text", text="TEST", thickness=1)),
    ]
    fig = plot_masks(test_image, text_masks, "Text Masks")
    plt.show()

    # Test brush mask with different strokes
    brush_masks = [
        ("Few strokes", mask_gen.generate(test_image, "brush", n_strokes=2)),
        ("Thin", mask_gen.generate(test_image, "brush", stroke_width=1)),
        ("Thick", mask_gen.generate(test_image, "brush", stroke_width=8)),
    ]
    fig = plot_masks(test_image, brush_masks, "Brush Masks")
    plt.show()

    # Test loading custom masks
    test_mask_dir = Path("data/test_outputs/custom_masks")
    test_mask_dir.mkdir(parents=True, exist_ok=True)

    # Create some test masks and save them
    test_cases = [
        ("custom_circle", cv2.circle(np.zeros((64, 64), dtype=np.uint8), (32, 32), 20, 255, -1)),
        (
            "custom_rect",
            cv2.rectangle(np.zeros((64, 64), dtype=np.uint8), (10, 10), (50, 50), 255, -1),
        ),
    ]

    for name, mask in test_cases:
        cv2.imwrite(str(test_mask_dir / f"{name}.png"), mask)

    # Test loading them back
    loaded_masks = MaskGenerator.load_masks_from_directory(
        test_mask_dir,
        target_size=(128, 128),  # Test resizing
    )

    # Visualize loaded masks
    fig = plot_masks(
        test_image,
        [(name, mask.astype(np.uint8) * 255) for name, mask in loaded_masks.items()],
        "Loaded Custom Masks",
    )
    plt.show()

    ## Mask Generation For Specific Image
    image = cv2.imread("test-images/the_scream.jpg")
    if image is None:
        raise ValueError("Could not read the_scream.jpg")

    config = MaskConfig(
        coverage_range=(0.14, 0.16),  # Target ~15% coverage
        thickness_range=(1, 2),  # Thin strokes
        num_components=(15, 20),  # More strokes to increase coverage
    )

    mask_gen = MaskGenerator(config)
    mask = mask_gen.generate(image, "brush", n_strokes=50, stroke_width=2)

    cv2.imwrite("mask.png", mask)
    coverage = np.mean(mask > 0)
    print(f"Mask coverage: {coverage:.1%}")

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Mask (coverage: {coverage:.1%})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
