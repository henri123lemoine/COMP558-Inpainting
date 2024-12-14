from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from scipy.stats import wasserstein_distance
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class InpaintingMetrics:
    """Metrics for inpainting quality assessment."""

    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index
    mae: float  # Mean Absolute Error

    # distribution metric
    emd: float  # Earth Mover's Distance

    # edge metric
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
        mask_bool = mask.astype(bool)

        psnr = peak_signal_noise_ratio(original, result, data_range=255)
        ssim = structural_similarity(original[mask_bool], result[mask_bool], data_range=255)
        mae = np.mean(np.abs(original.astype(float) - result.astype(float)))
        emd = wasserstein_distance(original[mask_bool].flatten(), result[mask_bool].flatten())
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
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt

    from src.datasets.images import InpaintingDataset

    dataset = InpaintingDataset(Path("data/datasets"))
    test_cases = dataset.generate_synthetic_dataset(size=128)

    # Randomly select one test case
    test_case_name = random.choice(list(test_cases.keys()))
    test_case = test_cases[test_case_name]
    original = test_case.original
    mask = test_case.mask

    result = original.copy()
    result[mask > 0] = np.mean(original[mask == 0])

    metrics = InpaintingMetrics.compute(
        original=original, result=result, mask=mask, execution_time=0.1
    )

    logger.debug("Test Metrics:")
    logger.debug(
        metrics.get_summary()
    )  # Interesintg observation: RESULTS ARE PRETTY BAD! This makes sense. Inpainting is not trivial!

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
