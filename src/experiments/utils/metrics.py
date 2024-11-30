from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class InpaintingMetrics:
    """Comprehensive metrics for inpainting quality assessment."""

    # Traditional image quality metrics
    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index
    mae: float  # Mean Absolute Error

    # Distribution metrics
    emd: float  # Earth Mover's Distance

    # Edge and structure metrics
    gradient_err: float  # Gradient magnitude error

    # Inpainting specific metrics
    boundary_err: float  # Boundary consistency
    texture_err: float  # Texture consistency

    # Execution time
    execution_time: float

    @classmethod
    def compute(
        cls,
        original: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        execution_time: float,
        window_size: int = 7,
    ) -> "InpaintingMetrics":
        """Compute all metrics comparing original and inpainted result."""
        # Ensure correct format
        if original.dtype != np.uint8:
            original = (original * 255).astype(np.uint8)
        if result.dtype != np.uint8:
            result = (result * 255).astype(np.uint8)
        mask_bool = mask.astype(bool)

        # Basic image quality metrics
        psnr = peak_signal_noise_ratio(original, result, data_range=255)
        ssim = structural_similarity(original, result, data_range=255)
        mae = np.mean(np.abs(original.astype(float) - result.astype(float)))

        # Distribution similarity (EMD)
        emd = wasserstein_distance(original[mask_bool].flatten(), result[mask_bool].flatten())

        # Edge and structure metrics
        gradient_err = cls._compute_gradient_error(original, result, mask_bool)

        # Inpainting specific metrics
        boundary_err = cls._compute_boundary_error(original, result, mask, window_size)
        texture_err = cls._compute_texture_error(original, result, mask_bool, window_size)

        return cls(
            psnr=psnr,
            ssim=ssim,
            mae=mae,
            emd=emd,
            gradient_err=gradient_err,
            boundary_err=boundary_err,
            texture_err=texture_err,
            execution_time=execution_time,
        )

    @staticmethod
    def _compute_gradient_error(
        original: np.ndarray, result: np.ndarray, mask: np.ndarray
    ) -> float:
        """Compute average error in gradient magnitudes."""
        # Compute gradients
        grad_orig_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
        grad_orig_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
        grad_res_x = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
        grad_res_y = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude differences
        mag_orig = np.sqrt(grad_orig_x**2 + grad_orig_y**2)
        mag_res = np.sqrt(grad_res_x**2 + grad_res_y**2)

        # Compute error in masked region
        error = np.mean(np.abs(mag_orig[mask] - mag_res[mask]))
        return error

    @staticmethod
    def _compute_total_variation(result: np.ndarray, mask: np.ndarray) -> float:
        """Compute total variation in the inpainted region."""
        # Get masked region coordinates
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return 0.0

        # Compute min/max bounds of masked region
        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = cols.min(), cols.max() + 1

        # Extract region of interest
        region = result[y1:y2, x1:x2].astype(float)
        region_mask = mask[y1:y2, x1:x2]

        # Compute variations
        dx = np.abs(np.diff(region, axis=0))
        dy = np.abs(np.diff(region, axis=1))

        # Only consider variations where both pixels are in mask
        dx_mask = region_mask[1:, :] & region_mask[:-1, :]
        dy_mask = region_mask[:, 1:] & region_mask[:, :-1]

        if not np.any(dx_mask) and not np.any(dy_mask):
            return 0.0

        tv = 0.0
        if np.any(dx_mask):
            tv += np.mean(dx[dx_mask])
        if np.any(dy_mask):
            tv += np.mean(dy[dy_mask])

        return tv

    @staticmethod
    def _compute_boundary_error(
        original: np.ndarray, result: np.ndarray, mask: np.ndarray, window_size: int
    ) -> float:
        """Compute error along mask boundaries."""
        # Get mask boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundary = cv2.dilate(mask, kernel) - mask

        # Extract patches along boundary
        half_window = window_size // 2
        boundary_y, boundary_x = np.where(boundary > 0)

        if len(boundary_y) == 0:
            return 0.0

        errors = []
        for y, x in zip(boundary_y, boundary_x):
            # Extract patches
            y1, y2 = max(0, y - half_window), min(mask.shape[0], y + half_window + 1)
            x1, x2 = max(0, x - half_window), min(mask.shape[1], x + half_window + 1)

            patch_orig = original[y1:y2, x1:x2]
            patch_res = result[y1:y2, x1:x2]

            # Compute local error
            err = np.mean(np.abs(patch_orig - patch_res))
            errors.append(err)

        return np.mean(errors)

    @staticmethod
    def _compute_texture_error(
        original: np.ndarray, result: np.ndarray, mask: np.ndarray, window_size: int
    ) -> float:
        """Compute texture consistency error using local standard deviations."""

        def get_local_std(img: np.ndarray, kernel_size: int) -> np.ndarray:
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            mean = cv2.filter2D(img.astype(float), -1, kernel)
            mean_sq = cv2.filter2D(img.astype(float) ** 2, -1, kernel)
            return np.sqrt(mean_sq - mean**2)

        # Compute local standard deviations
        std_orig = get_local_std(original, window_size)
        std_result = get_local_std(result, window_size)

        # Compare only in masked region
        error = np.mean(np.abs(std_orig[mask] - std_result[mask]))
        return error

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "PSNR": self.psnr,
            "SSIM": self.ssim,
            "MAE": self.mae,
            "EMD": self.emd,
            "Gradient Error": self.gradient_err,
            "Boundary Error": self.boundary_err,
            "Texture Error": self.texture_err,
            "Time (s)": self.execution_time,
        }

    def get_summary(self) -> str:
        """Get formatted string summary of metrics."""
        return "\n".join(f"{k}: {v:.4f}" for k, v in self.to_dict().items())


if __name__ == "__main__":
    # Test metrics computation
    from pathlib import Path

    from src.experiments.utils.datasets import InpaintingDataset

    # Load a test case
    dataset = InpaintingDataset(Path("data/datasets"))
    test_cases = dataset.generate_synthetic_dataset(size=128)

    # Get a test case
    test_case = test_cases["shapes"]
    original = test_case["image"]
    mask = test_case["masks"]["center"]

    # Create a simple "inpainting" result (just for testing)
    result = original.copy()
    result[mask > 0] = np.mean(original[mask == 0])  # Simple mean filling

    # Compute metrics
    metrics = InpaintingMetrics.compute(
        original=original, result=result, mask=mask, execution_time=0.1
    )

    # Print results
    print("Test Metrics:")
    print(metrics.get_summary())

    # Visual test
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(result, cmap="gray")
    axes[2].set_title("Result\n" + f"PSNR: {metrics.psnr:.2f}, SSIM: {metrics.ssim:.2f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("metrics_test.png")
    plt.close()
