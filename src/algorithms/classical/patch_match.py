import numpy as np
from loguru import logger
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

    def _initialize_nn_field(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Initialize the nearest-neighbor field randomly.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint

        Returns:
            Random initial nearest-neighbor field of shape (H, W, 2)
            containing (y, x) coordinates for each pixel's nearest neighbor
        """
        height, width = image.shape[:2]
        nn_field = np.zeros((height, width, 2), dtype=np.int32)

        # Get coordinates of pixels to be filled
        y_coords, x_coords = np.where(mask > 0)

        # For each pixel to be filled, assign random nearest neighbor
        # from non-masked regions
        valid_y, valid_x = np.where(mask == 0)
        if len(valid_y) == 0:
            raise ValueError("No valid pixels found in source region")

        for y, x in zip(y_coords, x_coords):
            # Choose random valid pixel
            idx = np.random.randint(len(valid_y))
            nn_field[y, x] = [valid_y[idx], valid_x[idx]]

        return nn_field

    def _get_patch(
        self,
        image: np.ndarray,
        center_y: int,
        center_x: int,
    ) -> np.ndarray:
        """Extract patch centered at given coordinates.

        Args:
            image: Input image
            center_y: Y-coordinate of patch center
            center_x: X-coordinate of patch center

        Returns:
            Patch of size (patch_size, patch_size)
        """
        y1 = max(0, center_y - self.half_patch)
        y2 = min(image.shape[0], center_y + self.half_patch + 1)
        x1 = max(0, center_x - self.half_patch)
        x2 = min(image.shape[1], center_x + self.half_patch + 1)

        patch = image[y1:y2, x1:x2]

        # Pad if necessary
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            pad_y1 = self.half_patch - (center_y - y1)
            pad_y2 = self.half_patch - (y2 - center_y - 1)
            pad_x1 = self.half_patch - (center_x - x1)
            pad_x2 = self.half_patch - (x2 - center_x - 1)
            patch = np.pad(
                patch,
                ((max(0, pad_y1), max(0, pad_y2)), (max(0, pad_x1), max(0, pad_x2))),
                mode="edge",
            )

        return patch

    def _patch_distance(
        self,
        patch1: np.ndarray,
        patch2: np.ndarray,
        mask: np.ndarray = None,
    ) -> float:
        """Compute distance between two patches.

        Args:
            patch1: First patch
            patch2: Second patch
            mask: Optional mask indicating valid pixels

        Returns:
            Sum of squared differences between valid pixels
        """
        if mask is None:
            return np.mean((patch1 - patch2) ** 2)
        else:
            valid_pixels = mask == 0
            if not np.any(valid_pixels):
                return float("inf")
            return np.sum((patch1 - patch2) ** 2 * valid_pixels) / np.sum(valid_pixels)

    def _propagate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nn_field: np.ndarray,
        reverse: bool = False,
    ) -> None:
        """Propagate good matches to neighboring pixels.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint
            nn_field: Current nearest-neighbor field
            reverse: Whether to propagate in reverse direction
        """
        height, width = image.shape[:2]
        y_range = range(height - 1, -1, -1) if reverse else range(height)
        x_range = range(width - 1, -1, -1) if reverse else range(width)

        for y in y_range:
            for x in x_range:
                if mask[y, x] == 0:  # Skip source pixels
                    continue

                # Get current best match
                current_nn = nn_field[y, x]
                current_patch = self._get_patch(image, y, x)
                best_dist = self._patch_distance(
                    current_patch,
                    self._get_patch(image, current_nn[0], current_nn[1]),
                    self._get_patch(mask, y, x),
                )

                # Try propagating from left/right neighbor
                if 0 <= x + (-1 if reverse else 1) < width:
                    neighbor_nn = nn_field[y, x + (-1 if reverse else 1)]
                    neighbor_patch = self._get_patch(image, neighbor_nn[0], neighbor_nn[1])
                    dist = self._patch_distance(
                        current_patch, neighbor_patch, self._get_patch(mask, y, x)
                    )

                    if dist < best_dist:
                        nn_field[y, x] = neighbor_nn
                        best_dist = dist

                # Try propagating from up/down neighbor
                if 0 <= y + (-1 if reverse else 1) < height:
                    neighbor_nn = nn_field[y + (-1 if reverse else 1), x]
                    neighbor_patch = self._get_patch(image, neighbor_nn[0], neighbor_nn[1])
                    dist = self._patch_distance(
                        current_patch, neighbor_patch, self._get_patch(mask, y, x)
                    )

                    if dist < best_dist:
                        nn_field[y, x] = neighbor_nn
                        best_dist = dist

    def _random_search(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nn_field: np.ndarray,
    ) -> None:
        """Perform random search to improve matches.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint
            nn_field: Current nearest-neighbor field
        """
        height, width = image.shape[:2]
        max_search_radius = max(height, width)

        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:  # Skip source pixels
                    continue

                current_nn = nn_field[y, x]
                current_patch = self._get_patch(image, y, x)
                best_dist = self._patch_distance(
                    current_patch,
                    self._get_patch(image, current_nn[0], current_nn[1]),
                    self._get_patch(mask, y, x),
                )

                # Try random candidates
                search_radius = max_search_radius
                while search_radius >= 1:
                    # Generate random offset within search radius
                    rand_y = current_nn[0] + np.random.randint(-search_radius, search_radius + 1)
                    rand_x = current_nn[1] + np.random.randint(-search_radius, search_radius + 1)

                    # Clamp to image boundaries
                    rand_y = np.clip(rand_y, 0, height - 1)
                    rand_x = np.clip(rand_x, 0, width - 1)

                    # Skip if in masked region
                    if mask[rand_y, rand_x] > 0:
                        continue

                    # Compute distance
                    rand_patch = self._get_patch(image, rand_y, rand_x)
                    dist = self._patch_distance(
                        current_patch, rand_patch, self._get_patch(mask, y, x)
                    )

                    if dist < best_dist:
                        nn_field[y, x] = [rand_y, rand_x]
                        best_dist = dist

                    search_radius *= self.search_ratio

    def _reconstruct_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nn_field: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct the image using the nearest-neighbor field.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint
            nn_field: Nearest-neighbor field

        Returns:
            Reconstructed image
        """
        result = image.copy()
        height, width = image.shape[:2]

        # Create weight accumulator for blending
        weights = np.zeros_like(image, dtype=np.float32)
        accumulated = np.zeros_like(image, dtype=np.float32)

        # Create Gaussian weights for patches
        y, x = np.mgrid[
            -self.half_patch : self.half_patch + 1, -self.half_patch : self.half_patch + 1
        ]
        patch_weights = np.exp(-(x**2 + y**2) / (2 * self.alpha**2))

        # Reconstruct each masked pixel
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue

                # Get corresponding patch from source
                src_y, src_x = nn_field[y, x]
                patch = self._get_patch(image, src_y, src_x)

                # Add weighted contribution
                y1 = max(0, y - self.half_patch)
                y2 = min(height, y + self.half_patch + 1)
                x1 = max(0, x - self.half_patch)
                x2 = min(width, x + self.half_patch + 1)

                patch_y1 = self.half_patch - (y - y1)
                patch_y2 = self.half_patch + (y2 - y)
                patch_x1 = self.half_patch - (x - x1)
                patch_x2 = self.half_patch + (x2 - x)

                weight_region = patch_weights[patch_y1:patch_y2, patch_x1:patch_x2]
                patch_region = patch[patch_y1:patch_y2, patch_x1:patch_x2]

                accumulated[y1:y2, x1:x2] += weight_region * patch_region
                weights[y1:y2, x1:x2] += weight_region

        # Normalize by accumulated weights
        mask_region = mask > 0
        result[mask_region] = accumulated[mask_region] / (weights[mask_region] + 1e-10)

        return result

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Inpaint the masked region using PatchMatch.

        Args:
            image: Input image
            mask: Binary mask where 1 indicates pixels to inpaint

        Returns:
            Inpainted image
        """
        if len(image.shape) != 2:
            raise ValueError("Only grayscale images are supported")

        height, width = image.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image dimensions ({height}, {width}) must be larger "
                f"than patch size {self.patch_size}"
            )

        logger.info(f"Starting PatchMatch inpainting with {self.num_iterations} iterations")

        # Initialize nearest-neighbor field
        nn_field = self._initialize_nn_field(image, mask)

        # Iteratively improve the nearest-neighbor field
        for iter_idx in tqdm(range(self.num_iterations), desc="PatchMatch"):
            # Propagate in forward direction
            self._propagate(image, mask, nn_field, reverse=False)

            # Propagate in reverse direction
            self._propagate(image, mask, nn_field, reverse=True)

            # Random search
            self._random_search(image, mask, nn_field)

        # Reconstruct the final image
        result = self._reconstruct_image(image, mask, nn_field)

        logger.info("PatchMatch inpainting completed")
        return result
