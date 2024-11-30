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
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (15, 5),
    metrics: Any | None = None,
) -> None:
    """Plot original, mask, and inpainted result side by side.

    Args:
        original: Original image
        mask: Inpainting mask
        result: Inpainted result
        save_path: Path to save the figure
        title: Title for the figure
        figsize: Figure size in inches
        metrics: Optional metrics to display
    """
    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[4, 1])

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
    ax_result.set_title("Inpainted Result")
    ax_result.axis("off")

    # Add metrics if provided
    if metrics is not None:
        metrics_text = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.to_dict().items())
        fig.text(0.1, 0.15, metrics_text, fontsize=10, family="monospace")

    # Add histogram comparison
    ax_hist = fig.add_subplot(gs[1, :])
    _plot_histograms(original, result, mask, ax_hist)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved visualization to {save_path}")

    plt.close()


def plot_multiple_results(
    results_dict: dict[str, dict[str, np.ndarray]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] | None = None,
    metrics_dict: dict[str, Any] | None = None,
) -> None:
    """Plot results from multiple algorithms side by side.

    Args:
        results_dict: Dictionary of results in the format:
            {
                'algorithm_name': {
                    'original': original_image,
                    'mask': mask_image,
                    'result': result_image
                }
            }
        save_path: Path to save the figure
        figsize: Figure size in inches
        metrics_dict: Optional dictionary of metrics for each algorithm
    """
    n_algorithms = len(results_dict)
    if not figsize:
        figsize = (15, 5 * n_algorithms)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_algorithms * 2, 3, height_ratios=[4, 1] * n_algorithms)

    for idx, (name, images) in enumerate(results_dict.items()):
        row = idx * 2

        # Plot original
        ax_orig = fig.add_subplot(gs[row, 0])
        ax_orig.imshow(
            images["original"], cmap="gray" if len(images["original"].shape) == 2 else None
        )
        ax_orig.set_title("Original" if idx == 0 else "")
        ax_orig.axis("off")

        # Plot mask
        ax_mask = fig.add_subplot(gs[row, 1])
        ax_mask.imshow(images["mask"], cmap="gray")
        ax_mask.set_title("Mask" if idx == 0 else "")
        ax_mask.axis("off")

        # Plot result
        ax_result = fig.add_subplot(gs[row, 2])
        ax_result.imshow(
            images["result"], cmap="gray" if len(images["result"].shape) == 2 else None
        )
        ax_result.set_title("Result" if idx == 0 else "")
        ax_result.axis("off")

        # Add algorithm name
        ax_orig.set_ylabel(name, size="large", rotation=0, labelpad=50)

        # Add metrics if provided
        if metrics_dict and name in metrics_dict:
            metrics_text = "\n".join(
                f"{k}: {v:.4f}" for k, v in metrics_dict[name].to_dict().items()
            )
            fig.text(
                0.7, 1 - (idx + 0.5) / n_algorithms, metrics_text, fontsize=8, family="monospace"
            )

        # Add histogram comparison
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
    """Plot histogram comparison of original and inpainted regions.

    Args:
        original: Original image
        result: Inpainted result
        mask: Inpainting mask
        ax: Matplotlib axes to plot on
    """
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


def plot_convergence(
    metrics_history: list[dict[str, float]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot convergence of metrics during inpainting.

    Args:
        metrics_history: List of metric dictionaries for each iteration
        save_path: Path to save the figure
        figsize: Figure size in inches
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    metrics = list(metrics_history[0].keys())
    for idx, metric in enumerate(metrics):
        values = [m[metric] for m in metrics_history]
        axes[idx].plot(values, marker="o")
        axes[idx].set_xlabel("Iteration")
        axes[idx].set_ylabel(metric)
        axes[idx].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved convergence plot to {save_path}")

    plt.close()


def create_error_heatmap(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Create a heatmap visualization of inpainting errors.

    Args:
        original: Original image
        result: Inpainted result
        mask: Inpainting mask
        save_path: Path to save the figure
        figsize: Figure size in inches
    """
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
