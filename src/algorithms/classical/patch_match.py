from dataclasses import dataclass
from typing import Final, Optional, TypeAlias

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask

Patch: TypeAlias = np.ndarray  # Shape: (patch_size, patch_size)
PatchMask: TypeAlias = np.ndarray  # Shape: (patch_size, patch_size), dtype: bool
NNField: TypeAlias = np.ndarray  # Shape: (H, W, 2), dtype: int32


@dataclass(frozen=True)
class PatchMatchParams:
    """Parameters for PatchMatch algorithm."""

    patch_size: int
    num_iterations: int
    search_ratio: float
    alpha: float

    def __post_init__(self) -> None:
        """Validate parameters."""
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

    # Constants
    MIN_VALID_RATIO: Final[float] = 0.3  # Minimum ratio of valid pixels in patch
    MAX_RANDOM_SAMPLES: Final[int] = 10  # Maximum random samples per position

    def __init__(
        self,
        patch_size: int = 9,
        num_iterations: int = 20,
        search_ratio: float = 0.5,
        alpha: float = 0.1,
    ) -> None:
        """Initialize PatchMatch algorithm with given parameters."""
        super().__init__("PatchMatch")

        self.params = PatchMatchParams(
            patch_size=patch_size,
            num_iterations=num_iterations,
            search_ratio=search_ratio,
            alpha=alpha,
        )

        self.half_patch = patch_size // 2
        # Pre-compute Gaussian weights for patch comparison
        self.weights = self._create_gaussian_weights()

        logger.debug(
            f"Initialized PatchMatch with: patch_size={patch_size}, "
            f"num_iterations={num_iterations}, search_ratio={search_ratio}"
        )

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
        h, w = image.shape
        y1 = max(0, y - self.half_patch)
        y2 = min(h, y + self.half_patch + 1)
        x1 = max(0, x - self.half_patch)
        x2 = min(w, x + self.half_patch + 1)

        patch = image[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] == 0  # Valid pixels are not masked

        # Handle boundary cases with padding
        if patch.shape != (self.params.patch_size, self.params.patch_size):
            pad_y1 = self.half_patch - (y - y1)
            pad_y2 = self.half_patch - (y2 - y - 1)
            pad_x1 = self.half_patch - (x - x1)
            pad_x2 = self.half_patch - (x2 - x - 1)

            padding = ((max(0, pad_y1), max(0, pad_y2)), (max(0, pad_x1), max(0, pad_x2)))
            patch = np.pad(patch, padding, mode="edge")
            valid = np.pad(valid, padding, mode="constant", constant_values=False)

        return patch, valid

    def _patch_distance(
        self,
        patch1: Patch,
        patch2: Patch,
        valid1: PatchMask,
        valid2: PatchMask,
        early_exit: Optional[float] = None,
    ) -> float:
        """Compute weighted SSD between valid regions of two patches."""
        # Only consider pixels valid in both patches
        valid = valid1 & valid2
        valid_ratio = valid.mean()

        # Skip if too few valid pixels
        if valid_ratio < self.MIN_VALID_RATIO:
            return float("inf")

        # Compute weighted SSD
        diff = (patch1 - patch2) ** 2
        weighted_diff = diff * self.weights * valid

        # Early exit if partial sum exceeds threshold
        if early_exit is not None:
            partial_sum = np.sum(weighted_diff)
            if partial_sum > early_exit:
                return float("inf")

        return np.sum(weighted_diff) / (np.sum(self.weights * valid) + 1e-8)

    def _initialize_nn_field(
        self,
        image: Image,
        mask: Mask,
    ) -> NNField:
        """Initialize nearest-neighbor field with good candidates."""
        h, w = image.shape
        nn_field = np.zeros((h, w, 2), dtype=np.int32)

        # Get valid source regions
        source_y, source_x = np.where(mask == 0)
        if len(source_y) == 0:
            raise ValueError("No valid source regions found in mask")

        # Pre-compute source patches for efficiency
        source_patches = []
        source_valids = []
        for y, x in zip(source_y, source_x):
            patch, valid = self._get_patch(image, mask, y, x)
            source_patches.append(patch)
            source_valids.append(valid)

        # Initialize target pixels
        target_y, target_x = np.where(mask > 0)

        for y, x in tqdm(
            zip(target_y, target_x),
            desc="Initializing NN field",
            total=len(target_y),
        ):
            target_patch, target_valid = self._get_patch(image, mask, y, x)

            # Try random candidates
            best_dist = float("inf")
            best_idx = None

            # Randomly sample source patches
            sample_indices = np.random.choice(
                len(source_y),
                size=min(self.MAX_RANDOM_SAMPLES, len(source_y)),
                replace=False,
            )

            for idx in sample_indices:
                dist = self._patch_distance(
                    target_patch,
                    source_patches[idx],
                    target_valid,
                    source_valids[idx],
                    early_exit=best_dist,
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                nn_field[y, x] = [source_y[best_idx], source_x[best_idx]]
            else:
                # Fallback to closest valid pixel
                nn_field[y, x] = [source_y[0], source_x[0]]
                # logger.warning(f"No good match found for pixel ({y}, {x})")

        return nn_field

    def _propagate(
        self,
        image: Image,
        mask: Mask,
        nn_field: NNField,
        reverse: bool = False,
    ) -> None:
        """Propagate good matches to neighboring pixels."""
        h, w = image.shape
        y_range = range(h - 1, -1, -1) if reverse else range(h)
        x_range = range(w - 1, -1, -1) if reverse else range(w)

        for y in y_range:
            for x in x_range:
                if mask[y, x] == 0:  # Skip source pixels
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

                # Try propagating from neighbors
                for dy, dx in [(0, -1), (-1, 0)] if not reverse else [(0, 1), (1, 0)]:
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

    def inpaint(
        self,
        image: Image,
        mask: Mask,
        **kwargs,
    ) -> Image:
        """Inpaint using PatchMatch algorithm."""
        # Validate inputs
        if len(image.shape) != 2:
            raise ValueError("Only grayscale images are supported")

        h, w = image.shape
        if h < self.params.patch_size or w < self.params.patch_size:
            raise ValueError(
                f"Image dimensions ({h}, {w}) must be larger than "
                f"patch size {self.params.patch_size}"
            )

        logger.info(
            f"Starting PatchMatch inpainting: {h}x{w} image, "
            f"{np.sum(mask > 0)} pixels to inpaint"
        )

        # Initialize result image
        result = image.copy()
        current_mask = mask.copy()

        # Main PatchMatch iterations
        for iter_idx in tqdm(range(self.params.num_iterations), desc="PatchMatch"):
            # Initialize NN field
            nn_field = self._initialize_nn_field(result, current_mask)

            # PatchMatch iterations
            for _ in range(2):  # Inner iterations for refinement
                # Forward and backward propagation
                self._propagate(result, current_mask, nn_field, reverse=False)
                self._random_search(result, current_mask, nn_field)
                self._propagate(result, current_mask, nn_field, reverse=True)

            # Update image using current NN field
            new_result = result.copy()
            masked_coords = np.where(current_mask > 0)
            for y, x in zip(*masked_coords):
                nn_y, nn_x = nn_field[y, x]
                new_result[y, x] = result[nn_y, nn_x]

            # Blend results with alpha
            result = (1 - self.params.alpha) * result + self.params.alpha * new_result

            # Update mask for next iteration
            if iter_idx < self.params.num_iterations - 1:
                current_mask = gaussian_filter(current_mask, sigma=0.5) > 0.5

        logger.info("PatchMatch inpainting completed")
        return result
