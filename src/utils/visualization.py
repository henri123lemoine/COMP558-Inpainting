from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def plot_inpainting_result(
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Plot original, mask, and inpainted result side by side.

    Args:
        original: Original image
        mask: Inpainting mask
        result: Inpainted result
        save_path: Path to save the figure (optional)
        title: Title for the figure (optional)
        figsize: Figure size in inches
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if title:
        fig.suptitle(title)

    # Plot original
    axes[0].imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    # Plot result
    axes[2].imshow(result, cmap="gray" if len(result.shape) == 2 else None)
    axes[2].set_title("Inpainted Result")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved visualization to {save_path}")

    plt.close()


def plot_multiple_results(
    results_dict: dict[str, dict[str, np.ndarray]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] | None = None,
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
        save_path: Path to save the figure (optional)
        figsize: Figure size in inches (optional)
    """
    n_algorithms = len(results_dict)
    if not figsize:
        figsize = (5 * (n_algorithms + 1), 5 * n_algorithms)

    fig, axes = plt.subplots(n_algorithms, 3, figsize=figsize)

    # If there's only one algorithm, wrap axes in a list
    if n_algorithms == 1:
        axes = [axes]

    for idx, (name, images) in enumerate(results_dict.items()):
        # Plot original
        axes[idx][0].imshow(
            images["original"], cmap="gray" if len(images["original"].shape) == 2 else None
        )
        axes[idx][0].set_title("Original" if idx == 0 else "")
        axes[idx][0].axis("off")

        # Plot mask
        axes[idx][1].imshow(images["mask"], cmap="gray")
        axes[idx][1].set_title("Mask" if idx == 0 else "")
        axes[idx][1].axis("off")

        # Plot result
        axes[idx][2].imshow(
            images["result"], cmap="gray" if len(images["result"].shape) == 2 else None
        )
        axes[idx][2].set_title("Result" if idx == 0 else "")
        axes[idx][2].axis("off")

        # Add algorithm name
        axes[idx][0].set_ylabel(name, size="large", rotation=0, labelpad=50)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved comparison visualization to {save_path}")

    plt.close()
