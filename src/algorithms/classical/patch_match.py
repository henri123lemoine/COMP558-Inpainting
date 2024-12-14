from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, InpaintingParams, Mask

Patch: TypeAlias = np.ndarray  # Shape: (patch_size, patch_size)
PatchMask: TypeAlias = np.ndarray  # Shape: (patch_size, patch_size), dtype: bool
NNField: TypeAlias = np.ndarray  # Shape: (H, W, 2), dtype: int32


@dataclass(frozen=True)
class PatchMatchParams(InpaintingParams):
    """Parameters for PatchMatch algorithm."""

    patch_size: int = 13
    num_iterations: int = 5
    search_ratio: float = 0.5
    alpha: float = 0.15

    def __post_init__(self) -> None:
        """Validate parameters."""
        super().__post_init__()
        if self.patch_size % 2 == 0:
            raise ValueError("Patch size must be odd")
        if self.patch_size < 3:
            raise ValueError("Patch size must be at least 3")
        if self.num_iterations < 1:
            raise ValueError("Number of iterations must be positive")
        if not 0 < self.search_ratio < 1:
            raise ValueError("Search ratio must be between 0 and 1")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")


class PatchMatchInpainting(InpaintingAlgorithm):
    """PatchMatch-based inpainting algorithm.

    Barnes, Connelly, et al. "PatchMatch: A randomized correspondence algorithm
    for structural image editing." ACM Transactions on Graphics (ToG) 28.3 (2009): 24.
    `https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php`
    """

    params_class = PatchMatchParams

    # Constants
    MIN_VALID_RATIO: Final[float] = 0.1  # Minimum ratio of valid pixels in patch
    MAX_RANDOM_SAMPLES: Final[int] = 25  # Maximum random samples per position

    def __init__(self, **kwargs) -> None:
        """Initialize PatchMatch algorithm with given parameters."""
        super().__init__("PatchMatch", **kwargs)
        self.half_patch = self.params.patch_size // 2
        self.weights = self._create_gaussian_weights()

    def _create_gaussian_weights(self) -> np.ndarray:
        """Create Gaussian weighting kernel for patch comparison."""
        x = np.linspace(-1, 1, self.params.patch_size)
        y = np.linspace(-1, 1, self.params.patch_size)
        xx, yy = np.meshgrid(x, y)
        weights = np.exp(-(xx**2 + yy**2) / 0.5)
        return weights / weights.sum()

    def _get_patch(
        self,
        image: Image,
        mask: Mask,
        y: int,
        x: int,
    ) -> tuple[Patch, PatchMask]:
        """Extract patch and validity mask centered at (y, x)."""
        h, w = image.shape[:2]  # Get spatial dimensions only
        y1 = max(0, y - self.half_patch)
        y2 = min(h, y + self.half_patch + 1)
        x1 = max(0, x - self.half_patch)
        x2 = min(w, x + self.half_patch + 1)

        patch = image[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] == 0

        # Handle boundary cases with padding
        if patch.shape[:2] != (self.params.patch_size, self.params.patch_size):
            pad_y1 = self.half_patch - (y - y1)
            pad_y2 = self.half_patch - (y2 - y - 1)
            pad_x1 = self.half_patch - (x - x1)
            pad_x2 = self.half_patch - (x2 - x - 1)

            padding = ((max(0, pad_y1), max(0, pad_y2)), (max(0, pad_x1), max(0, pad_x2)))
            if len(image.shape) == 3:
                padding += ((0, 0),)

            patch = np.pad(patch, padding, mode="edge")
            valid = np.pad(valid, padding[:2], mode="constant", constant_values=False)

        if len(image.shape) == 3:
            assert patch.shape == (self.params.patch_size, self.params.patch_size, image.shape[2])
        else:
            assert patch.shape == (self.params.patch_size, self.params.patch_size)
        assert valid.shape == (self.params.patch_size, self.params.patch_size)

        if np.any(np.isnan(patch)):
            logger.warning(f"NaN values in patch at ({y}, {x})")
        if np.min(patch) < 0 or np.max(patch) > 1:
            logger.warning(
                f"Patch values out of range at ({y}, {x}): [{np.min(patch):.3f}, {np.max(patch):.3f}]"
            )

        return patch, valid

    def _patch_distance(
        self,
        patch1: Patch,
        patch2: Patch,
        valid1: PatchMask,
        valid2: PatchMask,
        early_exit: float | None = None,
    ) -> float:
        """Compute weighted SSD between valid regions of two patches."""
        # Only consider pixels valid in both patches
        valid = valid1 & valid2
        valid_ratio = valid.mean()

        # Skip if too few valid pixels
        if valid_ratio < self.MIN_VALID_RATIO:
            return float("inf")

        # Ensure patches are normalized and valid
        if np.any(np.isnan(patch1)) or np.any(np.isnan(patch2)):
            return float("inf")

        patch1_valid = np.clip(patch1, 0, 1)
        patch2_valid = np.clip(patch2, 0, 1)

        # Compute weighted SSD
        diff = (patch1_valid - patch2_valid) ** 2

        if len(patch1.shape) == 3:
            # For color images, sum across channels and weight spatially
            weighted_diff = np.sum(diff, axis=2) * self.weights * valid
        else:
            weighted_diff = diff * self.weights * valid

        # Early exit if partial sum exceeds threshold
        if early_exit is not None:
            partial_sum = np.sum(weighted_diff)
            if partial_sum > early_exit:
                return float("inf")

        # Normalize by total weight of valid pixels
        valid_weights = np.sum(self.weights * valid)
        if valid_weights < 1e-8:
            return float("inf")

        return np.sum(weighted_diff) / valid_weights

    def _initialize_nn_field(
        self,
        image: Image,
        mask: Mask,
    ) -> NNField:
        """Initialize nearest-neighbor field with multiple initialization strategies."""
        h, w = image.shape
        nn_field = np.zeros((h, w, 2), dtype=np.int32)

        # Get valid source regions
        source_y, source_x = np.where(mask == 0)
        if len(source_y) == 0:
            raise ValueError("No valid source regions found in mask")

        # Initialize target pixels
        target_y, target_x = np.where(mask > 0)

        for y, x in zip(target_y, target_x):
            # Strategy 1: Try closest valid pixels first
            best_dist = float("inf")
            best_pos = None

            # Check immediate neighbors first
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                        dist = abs(dy) + abs(dx)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = [ny, nx]

            if best_pos is not None:
                nn_field[y, x] = best_pos
                continue

            # Strategy 2: Sample random positions with distance weighting
            candidates = []
            for _ in range(min(self.MAX_RANDOM_SAMPLES, len(source_y))):
                idx = np.random.randint(len(source_y))
                sy, sx = source_y[idx], source_x[idx]
                dist = np.sqrt((y - sy) ** 2 + (x - sx) ** 2)
                candidates.append((dist, [sy, sx]))

            # Take the closest among random samples
            if candidates:
                candidates.sort(key=lambda x: x[0])
                nn_field[y, x] = candidates[0][1]
            else:
                # Fallback to first valid pixel
                nn_field[y, x] = [source_y[0], source_x[0]]

        return nn_field

    def _propagate(
        self,
        image: Image,
        mask: Mask,
        nn_field: NNField,
        reverse: bool = False,
    ) -> None:
        """Propagate good matches to neighboring pixels, checking all directions."""
        h, w = image.shape
        y_range = range(h - 1, -1, -1) if reverse else range(h)
        x_range = range(w - 1, -1, -1) if reverse else range(w)

        for y in y_range:
            for x in x_range:
                if not mask[y, x]:  # Skip source pixels
                    continue

                current_patch, current_valid = self._get_patch(image, mask, y, x)
                best_nn = nn_field[y, x]
                best_dist = float("inf")

                # Try current best match
                if all(best_nn >= 0):
                    source_patch, source_valid = self._get_patch(
                        image, mask, best_nn[0], best_nn[1]
                    )
                    best_dist = self._patch_distance(
                        current_patch,
                        source_patch,
                        current_valid,
                        source_valid,
                    )

                # Check all four directions
                neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if reverse:
                    neighbors = [(-dy, -dx) for dy, dx in neighbors]

                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        prop_y = nn_field[ny, nx][0] - dy
                        prop_x = nn_field[ny, nx][1] - dx

                        if 0 <= prop_y < h and 0 <= prop_x < w and mask[prop_y, prop_x] == 0:
                            source_patch, source_valid = self._get_patch(
                                image, mask, prop_y, prop_x
                            )
                            dist = self._patch_distance(
                                current_patch,
                                source_patch,
                                current_valid,
                                source_valid,
                                early_exit=best_dist,
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_nn = [prop_y, prop_x]

                nn_field[y, x] = best_nn

    def _random_search(
        self,
        image: Image,
        mask: Mask,
        nn_field: NNField,
    ) -> None:
        """Perform random search around current best match."""
        h, w = image.shape
        max_radius = max(h, w)

        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0:  # Skip source pixels
                    continue

                current_patch, current_valid = self._get_patch(image, mask, y, x)
                best_nn = nn_field[y, x]
                best_dist = float("inf")

                if all(best_nn >= 0):
                    source_patch, source_valid = self._get_patch(
                        image, mask, best_nn[0], best_nn[1]
                    )
                    best_dist = self._patch_distance(
                        current_patch, source_patch, current_valid, source_valid
                    )

                radius = max_radius
                while radius >= 1:
                    # Try random offsets within current search radius
                    for _ in range(3):
                        dy = np.random.randint(-radius, radius + 1)
                        dx = np.random.randint(-radius, radius + 1)

                        ny = best_nn[0] + dy
                        nx = best_nn[1] + dx

                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                            source_patch, source_valid = self._get_patch(image, mask, ny, nx)
                            dist = self._patch_distance(
                                current_patch,
                                source_patch,
                                current_valid,
                                source_valid,
                                early_exit=best_dist,
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_nn = [ny, nx]

                    radius = int(radius * self.params.search_ratio)

                nn_field[y, x] = best_nn

    def _inpaint(
        self,
        image: Image,
        mask: Mask,
    ) -> Image:
        """Inpaint using enhanced PatchMatch algorithm."""
        h, w = image.shape[:2]
        if h < self.params.patch_size or w < self.params.patch_size:
            raise ValueError(
                f"Image dimensions ({h}, {w}) must be larger than "
                f"patch size {self.params.patch_size}"
            )

        # Debug prints to understand input values
        logger.debug(f"Initial image range: [{np.min(image):.3f}, {np.max(image):.3f}]")
        logger.debug(f"Mask range: [{np.min(mask):.3f}, {np.max(mask):.3f}]")
        logger.debug(f"Mask sum: {np.sum(mask > 0)} pixels to inpaint")

        # Initialize result image - ensure we start with the correct values
        result = image.copy()
        # Explicitly set masked regions to image mean as initial guess
        if len(image.shape) == 3:
            initial_guess = np.mean(image[mask == 0], axis=0)
            result[mask > 0] = initial_guess[None, None, :]
        else:
            initial_guess = np.mean(image[mask == 0])
            result[mask > 0] = initial_guess

        current_mask = mask.copy()  # mask > 0 indicates regions to inpaint

        # Main PatchMatch iterations
        for iter_idx in tqdm(range(self.params.num_iterations), desc="PatchMatch"):
            # Debug: check result range before updating
            logger.debug(
                f"Iteration {iter_idx}, result range: [{np.min(result):.3f}, {np.max(result):.3f}]"
            )

            # Initialize NN field
            nn_field = self._initialize_nn_field(result, current_mask)

            # Multiple refinement iterations
            for _ in range(3):
                self._propagate(result, current_mask, nn_field, reverse=False)
                self._random_search(result, current_mask, nn_field)
                self._propagate(result, current_mask, nn_field, reverse=True)

            # Update image using current NN field
            new_result = result.copy()
            masked_coords = np.where(current_mask > 0)

            for y, x in zip(*masked_coords):
                nn_y, nn_x = nn_field[y, x]
                if not current_mask[nn_y, nn_x]:
                    new_result[y, x] = result[nn_y, nn_x]
                else:
                    logger.warning(f"Tried to copy from masked pixel at ({nn_y}, {nn_x})")

            # Debug: check new_result values
            logger.debug(f"New result range: [{np.min(new_result):.3f}, {np.max(new_result):.3f}]")

            # Only blend masked regions
            blend_mask = current_mask.astype(float)
            blend_mask = gaussian_filter(blend_mask, sigma=1.0)
            blend_mask = np.clip(blend_mask, 0, 1)

            # Debug: check blend mask
            logger.debug(f"Blend mask range: [{np.min(blend_mask):.3f}, {np.max(blend_mask):.3f}]")

            # Blend for all channels if color image
            if len(image.shape) == 3:
                blend_mask = blend_mask[..., None]  # Add channel dimension for broadcasting

            # Only update pixels in the masked region
            result = result * (1 - blend_mask) + new_result * blend_mask

            # Update mask for next iteration, but only in the original masked region
            if iter_idx < self.params.num_iterations - 1:
                new_mask = gaussian_filter(current_mask.astype(float), sigma=0.5) > 0.5
                # Only update within original masked region
                new_mask = new_mask & (mask > 0)
                current_mask = new_mask

        # Final verification
        result = np.clip(result, 0, 1)
        # Ensure we haven't modified unmasked regions
        result[mask == 0] = image[mask == 0]

        logger.debug("PatchMatch inpainting completed")
        logger.debug(f"Final result range: [{np.min(result):.3f}, {np.max(result):.3f}]")
        return result


if __name__ == "__main__":
    inpainter = PatchMatchInpainting()
    inpainter.run()
