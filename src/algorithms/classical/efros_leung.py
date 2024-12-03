from dataclasses import dataclass

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


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
    ):
        super().__init__("EfrosLeung")

        self.params = EfrosLeungParams(
            window_size=window_size,
            error_threshold=error_threshold,
            sigma=sigma,
            n_candidates=n_candidates,
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
        image: Image,
        mask: Mask,
        pos: tuple[int, int],
    ) -> tuple[np.ndarray, Mask]:
        """Get the valid neighborhood around a position."""
        half_window = self.params.window_size // 2
        y, x = pos  # position of the center pixel
        window_size = self.params.window_size

        # Create padded versions of image and mask
        pad_top = max(0, half_window - y)
        pad_bottom = max(0, y + half_window + 1 - image.shape[0])
        pad_left = max(0, half_window - x)
        pad_right = max(0, x + half_window + 1 - image.shape[1])

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            padded_image = np.pad(
                image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
            )
            padded_mask = np.pad(
                mask,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=True,
            )

            # Adjust position for padded arrays
            y_pad = y + pad_top
            x_pad = x + pad_left

            # Extract exactly window_size x window_size neighborhood
            neighborhood = padded_image[
                y_pad - half_window : y_pad - half_window + window_size,
                x_pad - half_window : x_pad - half_window + window_size,
            ]
            validity_mask = ~padded_mask[
                y_pad - half_window : y_pad - half_window + window_size,
                x_pad - half_window : x_pad - half_window + window_size,
            ]
        else:
            # Extract exactly window_size x window_size neighborhood
            neighborhood = image[
                y - half_window : y - half_window + window_size,
                x - half_window : x - half_window + window_size,
            ]
            validity_mask = ~mask[
                y - half_window : y - half_window + window_size,
                x - half_window : x - half_window + window_size,
            ]

        # Double check we got the right size
        assert neighborhood.shape == (
            window_size,
            window_size,
        ), f"Got shape {neighborhood.shape}, expected ({window_size}, {window_size})"
        assert validity_mask.shape == (
            window_size,
            window_size,
        ), f"Got shape {validity_mask.shape}, expected ({window_size}, {window_size})"

        return neighborhood, validity_mask

    def _find_best_match(
        self,
        target: np.ndarray,  # Target neighborhood
        valid_mask: Mask,  # Mask indicating valid pixels in target
        image: Image,  # Source image to search in
        mask: Mask,  # Full image mask
        exclude_pos: tuple[int, int],  # Position to exclude from search
    ) -> tuple[float, tuple[int, int]]:
        """Find the best matching patch in the image."""
        half_window = self.params.window_size // 2
        window_size = self.params.window_size
        height, width = image.shape[:2]

        search_positions = list(zip(*np.where(~mask)))

        if not search_positions:
            raise ValueError("No unmasked pixels available to sample from")

        # Initialize with first valid position to ensure we always have a match
        best_error = float("inf")
        best_pos = None
        candidates = []

        # Find first valid position for initialization
        for y, x in search_positions:
            if (
                y < half_window
                or y >= height - half_window
                or x < half_window
                or x >= width - half_window
                or (y, x) == exclude_pos
            ):
                continue

            best_pos = (y, x)  # Initialize with first valid position
            break

        if best_pos is None:
            raise ValueError("No valid positions found within window constraints")

        # Now search through all positions for best match
        for y, x in search_positions:
            if (
                y < half_window
                or y >= height - half_window
                or x < half_window
                or x >= width - half_window
            ):
                continue

            if (y, x) == exclude_pos:
                continue

            # Get neighborhood at this position - use same window extraction as _get_neighborhood
            neighborhood = image[
                y - half_window : y - half_window + window_size,
                x - half_window : x - half_window + window_size,
            ]

            # Get the mask for this neighborhood
            neighborhood_mask = ~mask[
                y - half_window : y - half_window + window_size,
                x - half_window : x - half_window + window_size,
            ]

            # Only compare pixels that are valid in BOTH neighborhoods
            combined_valid_mask = valid_mask & neighborhood_mask

            # Skip if there are no valid pixels to compare
            if np.sum(combined_valid_mask) == 0:
                continue

            # Calculate error only for mutually valid pixels
            error = np.sum(self.weights * combined_valid_mask * (target - neighborhood) ** 2) / (
                np.sum(self.weights * combined_valid_mask) + 1e-10
            )

            # Always keep track of the best match
            if error < best_error:
                best_error = error
                best_pos = (y, x)

            # Only add to candidates if below threshold
            if error < self.params.error_threshold:
                candidates.append((error, (y, x)))

        # If we found candidates below threshold, randomly choose from the best ones
        if candidates:
            candidates.sort(key=lambda x: x[0])
            n_select = min(self.params.n_candidates, len(candidates))
            error, pos = candidates[np.random.randint(n_select)]
            return error, pos

        # If no candidates below threshold, return the best match we found
        assert best_pos is not None
        return best_error, best_pos

    def _inpaint(self, image: np.ndarray, mask: np.ndarray, max_steps: int = None) -> np.ndarray:
        """Efros-Leung inpainting."""
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

        with tqdm(total=total_pixels, desc="Efros-Leung") as pbar:
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
                    n_filled = np.sum(~window_mask == 0)

                    if n_filled > max_filled:
                        max_filled = n_filled
                        best_pos = (y, x)

                if best_pos is None:
                    logger.info("No more pixels to fill")
                    break

                # Get neighborhood of selected pixel
                neighborhood, valid_mask = self._get_neighborhood(result, remaining_mask, best_pos)

                try:
                    # Find best matching patch
                    _, match_pos = self._find_best_match(
                        neighborhood, valid_mask, image, mask, best_pos
                    )

                    # Get the matched pixel value, ensure it's a valid number
                    matched_value = image[match_pos]
                    if matched_value is None or np.isnan(matched_value):
                        raise ValueError("Invalid matched pixel value")

                    result[best_pos] = matched_value

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error finding match at {best_pos}: {e}")
                    # Fall back to average of valid neighbors, ensuring we always set a valid value
                    valid_values = neighborhood[valid_mask]
                    if len(valid_values) > 0:
                        valid_values = valid_values[
                            ~np.isnan(valid_values)
                        ]  # Remove any NaN values
                        if len(valid_values) > 0:
                            result[best_pos] = np.mean(valid_values)
                        else:
                            result[best_pos] = 0.5  # Fallback if no valid values
                    else:
                        result[best_pos] = 0.5  # Fallback if no valid neighbors

                # Verify we set a valid value
                assert result[best_pos] is not None and not np.isnan(
                    result[best_pos]
                ), f"Invalid value set at position {best_pos}"

                # Mark pixel as filled and update tracking
                remaining_mask[best_pos] = 0
                filled_pixels += 1
                unfilled_positions.remove(best_pos)
                pbar.update(1)

        # Final verification of output
        assert not np.any(np.isnan(result)), "NaN values found in result"
        assert not np.any([x is None for x in result.flat]), "None values found in result"

        return result


if __name__ == "__main__":
    inpainter = EfrosLeungInpainting()
    inpainter.run_example(scale_factor=0.25)
