from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.gridspec import GridSpec


def plot_inpainting_result(
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 4),
    metrics: Any | None = None,
) -> None:
    """Plot masked input and inpainted result side by side."""
    fig = plt.figure(figsize=figsize)
    is_grayscale = len(original.shape) == 2
    masked = original.copy()
    if masked.dtype != np.uint8:
        masked = (masked * 255).astype(np.uint8)

    if is_grayscale:
        masked[mask.astype(bool)] = 255
    else:
        mask_3d = np.repeat(mask[..., np.newaxis], 3, axis=2)
        masked[mask_3d.astype(bool)] = 255

    gs = GridSpec(1, 2)

    ax_masked = fig.add_subplot(gs[0, 0])
    ax_masked.imshow(masked, cmap="gray" if is_grayscale else None)
    ax_masked.set_title("Input with Mask")
    ax_masked.axis("off")

    ax_result = fig.add_subplot(gs[0, 1])
    ax_result.imshow(result, cmap="gray" if is_grayscale else None)
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
        fig.text(
            0.98,
            0.5,
            metrics_text,
            ha="right",
            va="center",
            fontsize=10,
            family="monospace",
            rotation=270,
        )

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
    """Plot results from multiple algorithms in a compact layout."""
    n_algorithms = len(results_dict)
    first_result = next(iter(results_dict.values()))
    is_grayscale = len(first_result["original"].shape) == 2

    if not figsize:
        figsize = (12, 2 + 2 * n_algorithms)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_algorithms + 1, 1, hspace=0)

    masked = first_result["original"].copy()
    if masked.dtype != np.uint8:
        masked = (masked * 255).astype(np.uint8)

    if is_grayscale:
        masked[first_result["mask"].astype(bool)] = 255
    else:
        mask_3d = np.repeat(first_result["mask"][..., np.newaxis], 3, axis=2)
        masked[mask_3d.astype(bool)] = 255

    ax_input = fig.add_subplot(gs[0])
    ax_input.imshow(masked, cmap="gray" if is_grayscale else None)
    ax_input.set_title("Input with Mask")
    ax_input.axis("off")

    for idx, (name, images) in enumerate(results_dict.items(), 1):
        ax_result = fig.add_subplot(gs[idx])
        ax_result.imshow(images["result"], cmap="gray" if is_grayscale else None)
        ax_result.axis("off")

        if metrics_dict and name in metrics_dict:
            metrics = metrics_dict[name]
            text_lines = [
                name,
                f"PSNR: {metrics.psnr:.2f}",
                f"SSIM: {metrics.ssim:.2f}",
                f"Edge: {metrics.edge_error:.2f}",
                f"Time: {metrics.execution_time:.2f}s",
            ]

            paragraph_offset = 0.05
            pos = ax_result.get_position()
            y_start = pos.y0 + pos.height - paragraph_offset * idx
            for i, line in enumerate(text_lines):
                line_height = 0.015
                y_pos = y_start - i * line_height

                fig.text(
                    pos.x1,
                    y_pos,
                    line,
                    ha="left",
                    va="center",
                    fontsize=9,
                    family="monospace",
                    transform=fig.transFigure,
                )

    plt.tight_layout(rect=[0, 0, 0.98, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved comparison visualization to {save_path}")

    plt.close()


def plot_metrics_by_category(
    results: pd.DataFrame,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 300,
) -> None:
    """Plot metrics broken down by category."""
    metrics = ["PSNR", "SSIM", "Edge Error", "Time (s)"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, metric in zip(axes.flat, metrics):
        sns.boxplot(data=results, x="Category", y=metric, hue="Algorithm", ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
        if metric not in ["Time (s)", "Edge Error"]:
            ax.text(
                0.98,
                0.98,
                "↑ higher is better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontstyle="italic",
            )
        else:
            ax.text(
                0.98,
                0.98,
                "↓ lower is better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontstyle="italic",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        logger.debug(f"Saved metrics by category plot to {save_path}")

    plt.close()


def plot_metrics_distribution(
    results: pd.DataFrame,
    algorithms: list[Any],
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 300,
) -> None:
    """Plot distribution of metrics across all cases."""
    metrics = ["PSNR", "SSIM", "Edge Error", "Time (s)"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, metric in zip(axes.flat, metrics):
        for algorithm in algorithms:
            data = results[results["Algorithm"] == algorithm.name][metric]
            sns.kdeplot(data=data, label=algorithm.name, ax=ax)

        ax.set_title(f"{metric} Distribution")
        ax.legend(title="Algorithm")

        if metric not in ["Time (s)", "Edge Error"]:
            ax.text(
                0.98,
                0.98,
                "↑ higher is better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontstyle="italic",
            )
        else:
            ax.text(
                0.98,
                0.98,
                "↓ lower is better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontstyle="italic",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        logger.debug(f"Saved metrics distribution plot to {save_path}")

    plt.close()


def generate_benchmark_report(
    results: pd.DataFrame,
    algorithms: list[Any],
    output_dir: Path,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 300,
) -> None:
    """Generate comprehensive benchmark report with tables and figures."""
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    latex_dir = output_dir / "latex"

    for directory in [metrics_dir, figures_dir, latex_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    results.to_csv(metrics_dir / "full_results.csv", index=False)

    summary = (
        results.groupby(["Algorithm", "Category"])
        .agg(
            {
                "PSNR": ["mean", "std"],
                "SSIM": ["mean", "std"],
                "Edge Error": ["mean", "std"],
                "Time (s)": ["mean", "std"],
            }
        )
        .round(4)
    )

    with open(latex_dir / "summary_table.tex", "w") as f:
        f.write(
            summary.to_latex(
                caption="Summary of Inpainting Results", label="tab:inpainting_summary"
            )
        )

    plot_metrics_by_category(
        results, save_path=figures_dir / "metrics_by_category.pdf", figsize=figsize, dpi=dpi
    )

    plot_metrics_distribution(
        results,
        algorithms,
        save_path=figures_dir / "metrics_distribution.pdf",
        figsize=figsize,
        dpi=dpi,
    )


def create_error_heatmap(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Create a heatmap visualization of inpainting errors."""
    is_grayscale = len(original.shape) == 2
    error = np.abs(original.astype(float) - result.astype(float))
    if not is_grayscale:
        error = np.mean(error, axis=2)

    error_masked = np.ma.masked_where(~mask.astype(bool), error)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Display original with appropriate colormap
    ax1.imshow(original, cmap="gray" if is_grayscale else None)
    ax1.set_title("Original")
    ax1.axis("off")

    # Display result with appropriate colormap
    ax2.imshow(result, cmap="gray" if is_grayscale else None)
    ax2.set_title("Inpainted Result")
    ax2.axis("off")

    # Error heatmap is always displayed using a heatmap colormap
    im = ax3.imshow(error_masked, cmap="hot")
    ax3.set_title("Error Heatmap")
    ax3.axis("off")
    plt.colorbar(im, ax=ax3, label="Mean Absolute Error")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.debug(f"Saved error heatmap to {save_path}")

    plt.close()
