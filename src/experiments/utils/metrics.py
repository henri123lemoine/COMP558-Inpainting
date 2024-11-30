from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class InpaintingMetrics:
    """Metrics for inpainting quality assessment."""

    # Traditional image quality metrics (from libraries)
    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index
    mae: float  # Mean Absolute Error

    # Simple distribution metric
    emd: float  # Earth Mover's Distance

    # Simple edge metric
    edge_error: float  # Basic Sobel edge difference

    # Time
    execution_time: float

    @classmethod
    def compute(
        cls,
        original: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        execution_time: float,
    ) -> "InpaintingMetrics":
        """Compute metrics comparing original and inpainted result."""
        # Ensure correct format
        if original.dtype != np.uint8:
            original = (original * 255).astype(np.uint8)
        if result.dtype != np.uint8:
            result = (result * 255).astype(np.uint8)
        mask_bool = mask.astype(bool)

        # Core metrics using library implementations
        psnr = peak_signal_noise_ratio(original, result, data_range=255)
        ssim = structural_similarity(original[mask_bool], result[mask_bool], data_range=255)
        mae = np.mean(np.abs(original.astype(float) - result.astype(float)))

        # Simple distribution comparison in masked region
        emd = wasserstein_distance(original[mask_bool].flatten(), result[mask_bool].flatten())

        # Simple edge comparison
        edge_error = cls._compute_edge_error(original, result, mask_bool)

        return cls(
            psnr=psnr,
            ssim=ssim,
            mae=mae,
            emd=emd,
            edge_error=edge_error,
            execution_time=execution_time,
        )

    @staticmethod
    def _compute_edge_error(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> float:
        """Basic edge difference using Sobel."""
        # Get edges
        edges_orig = cv2.Sobel(original, cv2.CV_64F, 1, 1)
        edges_result = cv2.Sobel(result, cv2.CV_64F, 1, 1)

        # Compare only in masked region
        return np.mean(np.abs(edges_orig[mask] - edges_result[mask]))

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "PSNR": self.psnr,
            "SSIM": self.ssim,
            "MAE": self.mae,
            "EMD": self.emd,
            "Edge Error": self.edge_error,
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
