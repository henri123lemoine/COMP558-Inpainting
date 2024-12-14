from dataclasses import dataclass

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, InpaintingParams, Mask


@dataclass(frozen=True)
class EfrosLeungParams(InpaintingParams):
    """
    Parameters for the Efros-Leung texture synthesis algorithm.

    window_size: Size of the neighborhood window (must be odd)
    error_threshold: Maximum allowed error for matching
    sigma: Standard deviation for Gaussian weighting
    """

    window_size: int = 13
    error_threshold: float = 0.25
    sigma: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        if self.window_size % 2 == 0:
            raise ValueError("Window size must be odd")


class EfrosLeungInpainting(InpaintingAlgorithm):
    """Efros-Leung texture synthesis algorithm adapted for inpainting.

    Efros, Alexei A., and Thomas K. Leung. "Texture synthesis by non-parametric sampling."
    In Proceedings of the seventh IEEE international conference on computer vision,
    vol. 2, pp. 1033-1038. IEEE, 1999.
    `https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf`
    """

    params_class = EfrosLeungParams

    def __init__(self, **kwargs):
        super().__init__(name="EfrosLeung", **kwargs)
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
        pad_width = [
            (pad_top, pad_bottom)
            for pad_top, pad_bottom in [
                (max(0, half_window - y), max(0, y + half_window + 1 - image.shape[0])),
                (max(0, half_window - x), max(0, x + half_window + 1 - image.shape[1])),
            ]
        ]

        # Add channel dimension padding if needed
        if len(image.shape) == 3:
            pad_width.append((0, 0))

        if any([sum(pads) > 0 for pads in pad_width]):
            padded_image = np.pad(image, pad_width, mode="reflect")
            padded_mask = np.pad(
                mask,
                pad_width[:2],  # Only pad spatial dimensions for mask
                mode="constant",
                constant_values=True,
            )

            # Adjust position for padded arrays
            y_pad = y + pad_width[0][0]
            x_pad = x + pad_width[1][0]

            # Extract exactly window_size x window_size neighborhood
            if len(image.shape) == 3:
                neighborhood = padded_image[
                    y_pad - half_window : y_pad - half_window + window_size,
                    x_pad - half_window : x_pad - half_window + window_size,
                    :,
                ]
            else:
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
            if len(image.shape) == 3:
                neighborhood = image[
                    y - half_window : y - half_window + window_size,
                    x - half_window : x - half_window + window_size,
                    :,
                ]
            else:
                neighborhood = image[
                    y - half_window : y - half_window + window_size,
                    x - half_window : x - half_window + window_size,
                ]
            validity_mask = ~mask[
                y - half_window : y - half_window + window_size,
                x - half_window : x - half_window + window_size,
            ]

        return neighborhood, validity_mask

    def _find_best_match(
        self,
        target: np.ndarray,
        valid_mask: Mask,
        masked_image: Image,
        mask: Mask,
        exclude_pos: tuple[int, int],
    ) -> tuple[float, tuple[int, int]]:
        """Find best match with neighborhood averaging fallback."""
        half_window = self.params.window_size // 2
        height, width = masked_image.shape[:2]

        # Get valid source regions
        filled_mask = ~mask & ~np.isnan(masked_image[..., 0])
        search_positions = list(zip(*np.where(filled_mask)))

        if not search_positions:
            # Fallback: use neighborhood average
            y, x = exclude_pos
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not np.any(np.isnan(masked_image[ny, nx]))
                        and not mask[ny, nx]
                    ):
                        neighbors.append(masked_image[ny, nx])

            if neighbors:
                return 0.0, exclude_pos
            raise ValueError("No valid neighbors found")

        best_error = float("inf")
        best_pos = None
        candidates = []

        # Try to find best matches
        for y, x in search_positions:
            if (
                y < half_window
                or y >= height - half_window
                or x < half_window
                or x >= width - half_window
                or (y, x) == exclude_pos
            ):
                continue

            neighborhood = masked_image[
                y - half_window : y + half_window + 1,
                x - half_window : x + half_window + 1,
            ]

            # Only compare non-nan pixels that are valid in both patches
            valid_comparison = valid_mask[..., None] & ~np.isnan(neighborhood)
            n_valid = np.sum(valid_comparison)

            if n_valid < 4:  # Require at least 4 valid pixels for comparison
                continue

            # Compute weighted error
            diff = (target - neighborhood) ** 2
            error = np.sum(self.weights[..., None] * valid_comparison * diff)
            error = error / (np.sum(self.weights[..., None] * valid_comparison) + 1e-10)

            if error < best_error:
                best_error = error
                best_pos = (y, x)

            if error < self.params.error_threshold:
                candidates.append((error, (y, x)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            n_best = min(3, len(candidates))
            idx = np.random.randint(n_best)
            return candidates[idx]

        if best_pos is not None:
            return best_error, best_pos

        # Last resort: neighborhood average
        y, x = exclude_pos
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and not np.any(np.isnan(masked_image[ny, nx]))
                    and not mask[ny, nx]
                ):
                    neighbors.append(masked_image[ny, nx])

        if neighbors:
            return 0.0, exclude_pos

        raise ValueError("No valid matches or neighbors found")

    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Efros-Leung inpainting with guaranteed filling of all masked pixels."""
        result = image.copy()
        remaining_mask = mask.copy()
        half_window = self.params.window_size // 2

        # Get total number of pixels to fill
        total_pixels = int(np.sum(mask > 0.5))
        filled_pixels = 0

        logger.debug(f"Starting inpainting of {total_pixels} pixels")

        with tqdm(total=total_pixels, desc="Efros-Leung") as pbar:
            while filled_pixels < total_pixels:
                # Find pixels that still need to be filled (contain NaN)
                # For 3D images, check any channel since NaN applies to all channels
                if len(result.shape) == 3:
                    unfilled_positions = np.where(np.isnan(result[..., 0]))
                else:
                    unfilled_positions = np.where(np.isnan(result))

                unfilled_y, unfilled_x = unfilled_positions[0], unfilled_positions[1]

                if len(unfilled_y) == 0:
                    break

                # Find pixel with most valid neighbors
                best_pos = None
                max_valid_neighbors = -1

                for i in range(len(unfilled_y)):
                    y, x = unfilled_y[i], unfilled_x[i]

                    # Skip pixels too close to border
                    if (
                        y < half_window
                        or y >= result.shape[0] - half_window
                        or x < half_window
                        or x >= result.shape[1] - half_window
                    ):
                        continue

                    # Count valid (non-NaN) neighbors
                    if len(result.shape) == 3:
                        window = result[y - 1 : y + 2, x - 1 : x + 2, 0]  # Check first channel
                    else:
                        window = result[y - 1 : y + 2, x - 1 : x + 2]
                    valid_neighbors = np.sum(~np.isnan(window))

                    if valid_neighbors > max_valid_neighbors:
                        max_valid_neighbors = valid_neighbors
                        best_pos = (y, x)

                # If no position found with the window constraint, take any unfilled pixel
                if best_pos is None:
                    for i in range(len(unfilled_y)):
                        y, x = unfilled_y[i], unfilled_x[i]
                        if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
                            best_pos = (y, x)
                            break

                if best_pos is None:
                    logger.error("No pixels to fill but NaN values remain!")
                    # Emergency fill of all remaining NaN values
                    nan_mask = np.isnan(result)
                    result[nan_mask] = 0.5
                    break

                y, x = best_pos
                try:
                    # Try to get neighborhood and find best match
                    neighborhood, valid_mask = self._get_neighborhood(
                        result, remaining_mask, best_pos
                    )
                    _, match_pos = self._find_best_match(
                        neighborhood, valid_mask, result, remaining_mask, best_pos
                    )
                    result[y, x] = result[match_pos]

                    # Verify we didn't set a NaN value
                    if np.any(np.isnan(result[y, x])):
                        raise ValueError("Matched to NaN value")

                except (ValueError, IndexError) as e:
                    # Fallback: use mean of valid neighbors or default value
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (
                                0 <= ny < result.shape[0]
                                and 0 <= nx < result.shape[1]
                                and not np.any(np.isnan(result[ny, nx]))
                            ):
                                neighbors.append(result[ny, nx])

                    if neighbors:
                        result[y, x] = np.mean(neighbors, axis=0)
                    else:
                        # If no valid neighbors, use the global mean of non-masked pixels
                        if len(result.shape) == 3:
                            valid_mask = ~np.isnan(result[..., 0])
                            result[y, x] = np.mean(result[valid_mask], axis=0)
                        else:
                            valid_mask = ~np.isnan(result)
                            result[y, x] = np.mean(result[valid_mask])

                remaining_mask[y, x] = 0
                filled_pixels += 1
                pbar.update(1)

        # Final safety check: fill any remaining NaN values
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            logger.warning(f"Filling {np.sum(nan_mask)} remaining NaN values with default value")
            result[nan_mask] = 0.5

        assert not np.any(np.isnan(result)), "NaN values found in result"
        assert not np.any([x is None for x in result.flat]), "None values found in result"

        return result


if __name__ == "__main__":
    inpainter = EfrosLeungInpainting()
    inpainter.run()
