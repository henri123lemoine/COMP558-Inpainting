import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.algorithms.base import InpaintingAlgorithm


class PatchMatchInpainting(InpaintingAlgorithm):
    """PatchMatch-based inpainting algorithm.

    Barnes, Connelly, et al. "PatchMatch: A randomized correspondence algorithm
    for structural image editing." ACM Transactions on Graphics (ToG) 28.3 (2009): 24.
    `https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php`
    """

    def __init__(
        self,
        patch_size: int = 7,  # must be odd
        num_iterations: int = 5,
        search_ratio: float = 0.5,  # ratio of random search radius reduction
        alpha: float = 0.1,  # blend factor for overlapping patches
    ):
        super().__init__("PatchMatch")

        if patch_size % 2 == 0:
            raise ValueError("Patch size must be odd")

        self.patch_size = patch_size
        self.num_iterations = num_iterations
        self.search_ratio = search_ratio
        self.alpha = alpha
        self.half_patch = patch_size // 2

        logger.debug(
            f"Initialized with patch_size={patch_size}, "
            f"num_iterations={num_iterations}, search_ratio={search_ratio}"
        )

    def _get_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        y: int,
        x: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract patch and its validity mask."""
        h, w = image.shape
        y1 = max(0, y - self.half_patch)
        y2 = min(h, y + self.half_patch + 1)
        x1 = max(0, x - self.half_patch)
        x2 = min(w, x + self.half_patch + 1)

        patch = image[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] == 0  # Valid pixels are not masked

        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            # Pad patch and validity mask if necessary
            dy1 = self.half_patch - (y - y1)
            dy2 = self.half_patch - (y2 - y - 1)
            dx1 = self.half_patch - (x - x1)
            dx2 = self.half_patch - (x2 - x - 1)

            patch = np.pad(
                patch, ((max(0, dy1), max(0, dy2)), (max(0, dx1), max(0, dx2))), mode="edge"
            )
            valid = np.pad(
                valid,
                ((max(0, dy1), max(0, dy2)), (max(0, dx1), max(0, dx2))),
                mode="constant",
                constant_values=False,
            )

        return patch, valid

    def _patch_distance(
        self,
        patch1: np.ndarray,
        patch2: np.ndarray,
        valid1: np.ndarray,
        valid2: np.ndarray,
    ) -> float:
        """Compute weighted SSD between valid regions of two patches."""
        # Only consider pixels valid in both patches
        valid = valid1 & valid2
        if not np.any(valid):
            return float("inf")

        diff = (patch1 - patch2) ** 2
        weights = gaussian_filter(valid.astype(float), sigma=1.0)
        weighted_diff = diff * weights

        return np.sum(weighted_diff * valid) / (np.sum(weights * valid) + 1e-10)

    def _initialize_nn_field(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Initialize nearest-neighbor field with good candidates."""
        h, w = image.shape
        nn_field = np.zeros((h, w, 2), dtype=np.int32)

        # Get valid source regions
        source_y, source_x = np.where(mask == 0)
        if len(source_y) == 0:
            raise ValueError("No valid source regions found")

        # For each target pixel
        target_y, target_x = np.where(mask > 0)

        for y, x in zip(target_y, target_x):
            # Get target patch
            target_patch, target_valid = self._get_patch(image, mask, y, x)

            # Try several random candidates
            best_dist = float("inf")
            best_pos = None

            for _ in range(10):  # Try 10 random positions
                idx = np.random.randint(len(source_y))
                sy, sx = source_y[idx], source_x[idx]

                # Only consider if the patch would be mostly valid
                source_patch, source_valid = self._get_patch(image, mask, sy, sx)
                dist = self._patch_distance(target_patch, source_patch, target_valid, source_valid)

                if dist < best_dist:
                    best_dist = dist
                    best_pos = [sy, sx]

            if best_pos is not None:
                nn_field[y, x] = best_pos
            else:
                # Fallback to random valid position
                idx = np.random.randint(len(source_y))
                nn_field[y, x] = [source_y[idx], source_x[idx]]

        return nn_field

    def _propagate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nn_field: np.ndarray,
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
                        current_patch, source_patch, current_valid, source_valid
                    )

                # Try propagating from neighbors
                for dy, dx in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and all(nn_field[ny, nx] >= 0):
                        # Offset the neighbor's NN in opposite direction
                        prop_y = nn_field[ny, nx][0] - dy
                        prop_x = nn_field[ny, nx][1] - dx

                        if 0 <= prop_y < h and 0 <= prop_x < w and mask[prop_y, prop_x] == 0:
                            source_patch, source_valid = self._get_patch(
                                image, mask, prop_y, prop_x
                            )
                            dist = self._patch_distance(
                                current_patch, source_patch, current_valid, source_valid
                            )

                            if dist < best_dist:
                                best_dist = dist
                                best_nn = [prop_y, prop_x]

                nn_field[y, x] = best_nn

    def _random_search(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nn_field: np.ndarray,
    ) -> None:
        """Perform random search around current best match."""
        h, w = image.shape
        search_radius = max(h, w)

        # Get valid source regions once
        source_y, source_x = np.where(mask == 0)
        if len(source_y) == 0:
            return

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

                radius = search_radius
                while radius >= 1:
                    # Try random offsets within current search radius
                    for _ in range(3):  # Try a few random positions at each radius
                        offset_y = np.random.randint(-radius, radius + 1)
                        offset_x = np.random.randint(-radius, radius + 1)

                        ny = best_nn[0] + offset_y
                        nx = best_nn[1] + offset_x

                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                            source_patch, source_valid = self._get_patch(image, mask, ny, nx)
                            dist = self._patch_distance(
                                current_patch, source_patch, current_valid, source_valid
                            )

                            if dist < best_dist:
                                best_dist = dist
                                best_nn = [ny, nx]

                    radius = int(radius * self.search_ratio)

                nn_field[y, x] = best_nn

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Inpaint using PatchMatch algorithm."""
        if len(image.shape) != 2:
            raise ValueError("Only grayscale images are supported")

        h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image dimensions ({h}, {w}) must be larger " f"than patch size {self.patch_size}"
            )

        logger.info(f"Starting PatchMatch inpainting with {self.num_iterations} iterations")

        # Initialize with current image
        result = image.copy()
        current_mask = mask.copy()

        # Main PatchMatch iterations
        for iter_idx in tqdm(range(self.num_iterations), desc="PatchMatch"):
            # Initialize NN field
            nn_field = self._initialize_nn_field(result, current_mask)

            # PatchMatch iterations
            for _ in range(2):  # Inner iterations for refinement
                # Forward propagation
                self._propagate(result, current_mask, nn_field, reverse=False)

                # Random search
                self._random_search(result, current_mask, nn_field)

                # Backward propagation
                self._propagate(result, current_mask, nn_field, reverse=True)

            # Update image using current NN field
            new_result = result.copy()
            for y, x in zip(*np.where(current_mask > 0)):
                nn_y, nn_x = nn_field[y, x]
                new_result[y, x] = result[nn_y, nn_x]

            # Blend results
            result = new_result

            # Update mask
            if iter_idx < self.num_iterations - 1:
                # Gradually reduce the mask
                current_mask = gaussian_filter(current_mask, sigma=0.5) > 0.5

        logger.info("PatchMatch inpainting completed")
        return result
