from dataclasses import dataclass

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import InpaintingAlgorithm


@dataclass(frozen=True)
class EfrosLeungParams:
    """
    Parameters for the Efros-Leung texture synthesis algorithm.

    window_size: Size of the neighborhood window (must be odd)
    error_threshold: Maximum allowed error for matching
    sigma: Standard deviation for Gaussian weighting
    n_candidates: Number of candidate matches to randomly choose from
    """

    window_size: int = 11
    error_threshold: float = 0.1
    sigma: float = 1.0
    n_candidates: int = 10
    search_step: int = 2
    batch_size: int = 4

    def __post_init__(self):
        if self.window_size % 2 == 0:
            raise ValueError("Window size must be odd")


class EfrosLeungInpainting(InpaintingAlgorithm):
    """Efros-Leung texture synthesis algorithm adapted for inpainting.

    Efros, Alexei A., and Thomas K. Leung. "Texture synthesis by non-parametric sampling."
    In Proceedings of the seventh IEEE international conference on computer vision,
    vol. 2, pp. 1033-1038. IEEE, 1999.
    `https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf`
    """

    def __init__(
        self,
        window_size: int = 11,
        error_threshold: float = 0.1,
        sigma: float = 1.0,
        n_candidates: int = 10,
        search_step: int = 2,
        batch_size: int = 4,
    ):
        super().__init__("EfrosLeung")

        self.params = EfrosLeungParams(
            window_size=window_size,
            error_threshold=error_threshold,
            sigma=sigma,
            n_candidates=n_candidates,
            search_step=search_step,
            batch_size=batch_size,
        )

        self.weights = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self) -> np.ndarray:
        """Create a Gaussian weighting kernel for the window."""
        kernel_size = self.params.window_size
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(dist**2) / (2 * self.params.sigma**2))

        return kernel / kernel.sum()

    def _get_neighborhood(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pos: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the valid neighborhood around a position."""
        half_window = self.params.window_size // 2
        y, x = pos  # position of the center pixel

        # Extract neighborhood and corresponding mask
        neighborhood = image[
            y - half_window : y + half_window + 1, x - half_window : x + half_window + 1
        ]

        mask_region = mask[
            y - half_window : y + half_window + 1, x - half_window : x + half_window + 1
        ]

        # Valid pixels are those that don't need inpainting (mask != 1)
        validity_mask = mask_region < 0.5

        return neighborhood, validity_mask

    def _find_best_match(
        self,
        target: np.ndarray,
        valid_mask: np.ndarray,
        image: np.ndarray,
        exclude_pos: tuple[int, int],
    ) -> tuple[float, tuple[int, int]]:
        """Find the best matching patch in the image.

        Args:
            target: Target neighborhood
            valid_mask: Mask indicating valid pixels in target
            image: Source image to search in
            exclude_pos: Position to exclude from search
        """
        half_window = self.params.window_size // 2
        height, width = image.shape[:2]

        # Prepare valid positions for searching
        y_range = range(half_window, height - half_window)
        x_range = range(half_window, width - half_window)

        best_error = float("inf")
        best_pos = None
        candidates = []

        # Search through all possible positions
        for y in y_range:
            for x in x_range:
                # Skip the excluded position
                if (y, x) == exclude_pos:
                    continue

                # Get neighborhood at this position
                neighborhood = image[
                    y - half_window : y + half_window + 1, x - half_window : x + half_window + 1
                ]

                # Calculate error for valid pixels only
                error = np.sum(self.weights * valid_mask * (target - neighborhood) ** 2) / np.sum(
                    self.weights * valid_mask
                )

                if error < self.params.error_threshold:
                    candidates.append((error, (y, x)))
                    if error < best_error:
                        best_error = error
                        best_pos = (y, x)

        # If we found candidates, randomly choose from the best ones
        if candidates:
            candidates.sort(key=lambda x: x[0])
            n_select = min(self.params.n_candidates, len(candidates))
            error, pos = candidates[np.random.randint(n_select)]
            return error, pos

        return best_error, best_pos

    def inpaint(self, image: np.ndarray, mask: np.ndarray, max_steps: int = None) -> np.ndarray:
        """Efros-Leung inpainting."""
        if len(image.shape) != 2:
            raise ValueError("Only grayscale images are supported")

        # Normalize mask to binary 0/1
        mask = (mask > 0.5).astype(np.float32)

        if np.all(mask == 0):
            logger.warning("Empty mask, nothing to inpaint")
            return image
        elif np.all(mask == 1):
            logger.warning("Full mask, nothing to inpaint")
            return image

        result = image.copy()
        remaining_mask = mask.copy()
        half_window = self.params.window_size // 2

        # Get total number of pixels to fill
        n_pixels = int(np.sum(mask > 0.5))  # Cast to int for tqdm
        if max_steps is None:
            max_steps = n_pixels

        logger.info(f"Starting inpainting of {n_pixels} pixels")

        # Pre-compute list of unfilled positions
        unfilled_positions = [
            (y, x)
            for y in range(half_window, image.shape[0] - half_window)
            for x in range(half_window, image.shape[1] - half_window)
            if remaining_mask[y, x] == 1
        ]

        filled_pixels = 0
        total_pixels = min(n_pixels, max_steps)

        with tqdm(total=total_pixels, desc="Inpainting") as pbar:
            while filled_pixels < max_steps and unfilled_positions:
                # Find unfilled pixel with most filled neighbors
                max_filled = 0
                best_pos = None

                for y, x in unfilled_positions:
                    # Count filled neighbors in window
                    window_mask = remaining_mask[
                        y - half_window : y + half_window + 1,
                        x - half_window : x + half_window + 1,
                    ]
                    n_filled = np.sum(window_mask == 0)

                    if n_filled > max_filled:
                        max_filled = n_filled
                        best_pos = (y, x)

                if best_pos is None:
                    logger.info("No more pixels to fill")
                    break

                # Get neighborhood of selected pixel
                neighborhood, valid_mask = self._get_neighborhood(result, remaining_mask, best_pos)

                # Find best matching patch
                _, match_pos = self._find_best_match(neighborhood, valid_mask, image, best_pos)

                if match_pos is None:
                    logger.warning(f"No valid match found for position {best_pos}")
                    # Fill with average of valid neighbors as fallback
                    valid_values = neighborhood[valid_mask]
                    result[best_pos] = np.mean(valid_values) if len(valid_values) > 0 else 0.5
                else:
                    # Copy the center pixel from the best match
                    result[best_pos] = image[match_pos]

                # Mark pixel as filled and update tracking
                remaining_mask[best_pos] = 0
                filled_pixels += 1
                unfilled_positions.remove(best_pos)
                pbar.update(1)

        logger.info(f"Completed inpainting of {filled_pixels} pixels")
        return result


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    inpainter = EfrosLeungInpainting()

    image = cv2.imread("data/datasets/real/real_1227/image.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("data/datasets/real/real_1227/mask_brush.png", cv2.IMREAD_GRAYSCALE)
    print(image.shape, mask.shape)

    plt.imshow(image, cmap="gray")

    image = image.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    result = inpainter.inpaint(image, mask)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Mask")
    ax2.axis("off")

    ax3.imshow(result, cmap="gray")
    ax3.set_title("Result")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()
