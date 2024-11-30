import numpy as np
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import InpaintingAlgorithm


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
    ):
        """Initialize the algorithm.

        Args:
            window_size: Size of the neighborhood window (must be odd)
            error_threshold: Maximum allowed error for matching
            sigma: Standard deviation for Gaussian weighting
            n_candidates: Number of candidate matches to randomly choose from
        """
        super().__init__("EfrosLeung")

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")

        self.window_size = window_size
        self.error_threshold = error_threshold
        self.sigma = sigma
        self.n_candidates = n_candidates

        # Create Gaussian weighting kernel
        self.weights = self._create_gaussian_kernel()
        logger.debug(
            f"Initialized with window_size={window_size}, "
            f"error_threshold={error_threshold}, sigma={sigma}"
        )

    def _create_gaussian_kernel(self) -> np.ndarray:
        """Create a Gaussian weighting kernel for the window."""
        kernel_size = self.window_size
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(dist**2) / (2 * self.sigma**2))

        return kernel / kernel.sum()

    def _get_neighborhood(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pos: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the valid neighborhood around a position.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint
            pos: (y, x) position of the center pixel

        Returns:
            Tuple of (neighborhood, validity_mask)
        """
        half_window = self.window_size // 2
        y, x = pos

        # Extract neighborhood and corresponding mask
        neighborhood = image[
            y - half_window : y + half_window + 1, x - half_window : x + half_window + 1
        ]

        mask_region = mask[
            y - half_window : y + half_window + 1, x - half_window : x + half_window + 1
        ]

        # Valid pixels are those that are already filled (mask == 0)
        validity_mask = mask_region == 0

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

        Returns:
            Tuple of (error, (y, x) position)
        """
        half_window = self.window_size // 2
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

                if error < self.error_threshold:
                    candidates.append((error, (y, x)))
                    if error < best_error:
                        best_error = error
                        best_pos = (y, x)

        # If we found candidates, randomly choose from the best ones
        if candidates:
            candidates.sort(key=lambda x: x[0])
            n_select = min(self.n_candidates, len(candidates))
            error, pos = candidates[np.random.randint(n_select)]
            return error, pos

        return best_error, best_pos

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        max_steps: int = None,
        **kwargs,
    ) -> np.ndarray:
        """Inpaint the masked region using texture synthesis.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint
            max_steps: Maximum number of pixels to fill (default: all masked pixels)

        Returns:
            Inpainted image
        """
        if len(image.shape) != 2:
            raise ValueError("Only grayscale images are supported")

        # Make copies to work with
        result = image.copy()
        remaining_mask = mask.copy()

        # Get total number of pixels to fill
        n_pixels = np.sum(mask > 0)
        if max_steps is None:
            max_steps = n_pixels

        logger.info(f"Starting inpainting of {n_pixels} pixels")

        # Initialize progress bar
        pbar = tqdm(total=min(n_pixels, max_steps), desc="Inpainting")

        filled_pixels = 0
        half_window = self.window_size // 2

        while filled_pixels < max_steps:
            # Find unfilled pixel with most filled neighbors
            max_filled = 0
            best_pos = None

            for y in range(half_window, image.shape[0] - half_window):
                for x in range(half_window, image.shape[1] - half_window):
                    if remaining_mask[y, x] == 1:  # If pixel needs to be filled
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
                if len(valid_values) > 0:
                    result[best_pos] = np.mean(valid_values)
                else:
                    result[best_pos] = 0.5  # Default to mid-gray if no valid neighbors
            else:
                # Copy the center pixel from the best match
                result[best_pos] = image[match_pos]

            # Mark pixel as filled
            remaining_mask[best_pos] = 0
            filled_pixels += 1
            pbar.update(1)

        pbar.close()
        logger.info(f"Completed inpainting of {filled_pixels} pixels")
        return result
