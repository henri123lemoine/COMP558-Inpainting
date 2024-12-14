from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.gridspec import GridSpec


def plot_inpainting_result(
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    save_path: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 4),
    metrics: Any | None = None,
    show_histogram: bool = False,
) -> None:
    """Plot original, mask, and inpainted result side by side with optional histogram."""
    if show_histogram:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
        gs = GridSpec(2, 3, height_ratios=[4, 1])
    else:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3)

    if title:
        fig.suptitle(title, fontsize=12, y=0.95)

    # Plot original
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    # Plot mask
    ax_mask = fig.add_subplot(gs[0, 1])
    ax_mask.imshow(mask, cmap="gray")
    ax_mask.set_title("Mask")
    ax_mask.axis("off")

    # Plot result
    ax_result = fig.add_subplot(gs[0, 2])
    ax_result.imshow(result, cmap="gray" if len(result.shape) == 2 else None)
    ax_result.set_title("Result")
    ax_result.axis("off")

    if metrics is not None:
        key_metrics = {
            "PSNR": metrics.psnr,
            "SSIM": metrics.ssim,
            "Edge": metrics.edge_error,
            "Time": metrics.execution_time,
        }
        metrics_text = " | ".join(f"{k}: {v:.2f}" for k, v in key_metrics.items())
        fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=10, family="monospace")

    # Add histogram comparison if requested
    if show_histogram:
        ax_hist = fig.add_subplot(gs[1, :])
        _plot_histograms(original, result, mask, ax_hist)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved visualization to {save_path}")

    plt.close()


def plot_multiple_results(
    results_dict: dict[str, dict[str, np.ndarray]],
    save_path: str | None = None,
    figsize: tuple[int, int] | None = None,
    metrics_dict: dict[str, Any] | None = None,
    show_histogram: bool = False,
) -> None:
    """Plot results from multiple algorithms in a clean, organized layout."""
    n_algorithms = len(results_dict)

    # Get first result to determine if images are grayscale
    first_result = next(iter(results_dict.values()))
    is_grayscale = len(first_result["original"].shape) == 2

    # Calculate figure size if not provided
    if not figsize:
        base_height = 3
        total_height = base_height * (n_algorithms + 1)  # +1 for original and mask
        figsize = (12, total_height)

    fig = plt.figure(figsize=figsize)

    # Create grid: top row for original + mask, then one row per algorithm
    n_rows = n_algorithms + 1
    if show_histogram:
        gs = GridSpec(n_rows * 2, 2, height_ratios=[1, 0.3] * n_rows)
    else:
        gs = GridSpec(n_rows, 2)

    # Plot original and mask in first row
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(first_result["original"], cmap="gray" if is_grayscale else None)
    ax_orig.set_title("Original Image")
    ax_orig.axis("off")

    ax_mask = fig.add_subplot(gs[0, 1])
    ax_mask.imshow(first_result["mask"], cmap="gray")
    ax_mask.set_title("Mask")
    ax_mask.axis("off")

    # Plot each algorithm's result
    for idx, (name, images) in enumerate(results_dict.items(), 1):
        row = idx * (2 if show_histogram else 1)

        # Result
        ax_result = fig.add_subplot(gs[row, :])
        ax_result.imshow(images["result"], cmap="gray" if is_grayscale else None)

        # Add metrics if provided
        if metrics_dict and name in metrics_dict:
            metrics = metrics_dict[name]
            metrics_text = f"PSNR: {metrics.psnr:.2f} | SSIM: {metrics.ssim:.2f} | "
            metrics_text += f"Edge: {metrics.edge_error:.2f} | Time: {metrics.execution_time:.2f}s"
            ax_result.set_title(f"{name}\n{metrics_text}", pad=10)
        else:
            ax_result.set_title(name, pad=10)

        ax_result.axis("off")

        # Add histogram if requested
        if show_histogram:
            ax_hist = fig.add_subplot(gs[row + 1, :])
            _plot_histograms(images["original"], images["result"], images["mask"], ax_hist)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved comparison visualization to {save_path}")

    plt.close()


def _plot_histograms(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Plot histogram comparison of original and inpainted regions."""
    # Convert to appropriate format
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8)
    if result.dtype != np.uint8:
        result = (result * 255).astype(np.uint8)

    mask_bool = mask.astype(bool)

    # Plot histograms
    ax.hist(
        original[mask_bool].ravel(),
        bins=50,
        alpha=0.5,
        label="Original (inpainted region)",
        density=True,
    )
    ax.hist(
        result[mask_bool].ravel(),
        bins=50,
        alpha=0.5,
        label="Inpainted result",
        density=True,
    )
    ax.hist(
        original[~mask_bool].ravel(),
        bins=50,
        alpha=0.5,
        label="Original (context region)",
        density=True,
    )

    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize="small")


def create_error_heatmap(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Create a heatmap visualization of inpainting errors."""
    # Compute absolute error
    error = np.abs(original.astype(float) - result.astype(float))
    error_masked = np.ma.masked_where(~mask.astype(bool), error)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Plot original
    ax1.imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    ax1.set_title("Original")
    ax1.axis("off")

    # Plot result
    ax2.imshow(result, cmap="gray" if len(result.shape) == 2 else None)
    ax2.set_title("Inpainted Result")
    ax2.axis("off")

    # Plot error heatmap
    im = ax3.imshow(error_masked, cmap="hot")
    ax3.set_title("Error Heatmap")
    ax3.axis("off")
    plt.colorbar(im, ax=ax3, label="Absolute Error")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved error heatmap to {save_path}")

    plt.close()
