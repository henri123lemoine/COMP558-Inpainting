import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask
from src.datasets.images import InpaintingDataset
from src.experiments.utils.metrics import InpaintingMetrics
from src.experiments.utils.visualization import (
    create_error_heatmap,
    plot_inpainting_result,
    plot_multiple_results,
)
from src.settings import DATA_PATH


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    # Dataset parameters
    synthetic_size: int = 128  # Size of synthetic test images
    n_real_images: int = 20  # Number of real test images to use
    real_size: int = 128  # Size to resize real images to

    # Output parameters
    save_individual: bool = True  # Save individual results
    save_comparisons: bool = True  # Save side-by-side comparisons
    save_heatmaps: bool = True  # Save error heatmaps

    # Visualization parameters
    figsize: tuple[int, int] = (15, 10)
    dpi: int = 300


class InpaintingBenchmark:
    """Comprehensive benchmark for inpainting algorithms."""

    def __init__(
        self,
        output_dir: Path | None = None,
        config: BenchmarkConfig | None = None,
    ):
        self.output_dir = output_dir or DATA_PATH / "benchmark_results"
        self.config = config or BenchmarkConfig()

        # Setup directories
        self.dataset_dir = self.output_dir / "datasets"
        self.results_dir = self.output_dir / "results"
        self.metrics_dir = self.output_dir / "metrics"
        self.figures_dir = self.output_dir / "figures"
        self.latex_dir = self.output_dir / "latex"

        for directory in [
            self.dataset_dir,
            self.results_dir,
            self.metrics_dir,
            self.figures_dir,
            self.latex_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.dataset = InpaintingDataset(self.dataset_dir)

        sns.set_theme()
        sns.set_context("paper")

    def run(self, algorithms: list[InpaintingAlgorithm]) -> pd.DataFrame:
        """Run complete benchmark suite."""
        results = []

        # Get all test cases
        synthetic_samples = self.dataset.generate_synthetic_dataset(
            size=self.config.synthetic_size, mask_types=["center", "random", "brush", "text"]
        )
        real_samples = self.dataset.load_real_dataset(
            n_images=self.config.n_real_images,
            size=self.config.real_size,
            mask_types=["center", "random", "brush", "text"],
        )

        all_samples = {**synthetic_samples, **real_samples}
        logger.info(f"Running benchmark on {len(all_samples)} test cases")

        # Note: case_name now includes the mask type (e.g. "lines_center", "real_0_brush")
        for case_name, sample in tqdm(all_samples.items(), desc="Test cases"):
            base_name, mask_type = case_name.rsplit("_", 1)

            algorithm_results = {}
            for algorithm in algorithms:
                logger.debug(f"Running {algorithm.name} on {base_name} with {mask_type} mask")

                try:
                    start_time = time.time()
                    # Note: Images are already properly scaled in InpaintSample
                    result = algorithm.inpaint(
                        # sample.original.astype(np.float32) / 255.0,
                        sample.masked / 255.0,
                        sample.mask.astype(np.float32) / 255.0,
                    )
                    result = (result * 255).astype(np.uint8)
                    exec_time = time.time() - start_time

                    metrics = InpaintingMetrics.compute(
                        original=sample.original,
                        result=result,
                        mask=sample.mask,
                        execution_time=exec_time,
                    )

                    algorithm_results[algorithm.name] = {"result": result, "metrics": metrics}

                    results.append(
                        {
                            "Algorithm": algorithm.name,
                            "Case": base_name,
                            "Mask": mask_type,
                            "Category": str(sample.category.name),
                            **metrics.to_dict(),
                        }
                    )

                    # Save individual results if requested
                    if self.config.save_individual:
                        self._save_individual_result(
                            algorithm.name,
                            base_name,
                            mask_type,
                            sample.original,
                            sample.mask,
                            result,
                            metrics,
                        )

                except Exception as e:
                    logger.error(f"Error processing {case_name} with {algorithm.name}: {str(e)}")
                    continue

            # Generate comparisons if we have results
            if algorithm_results and self.config.save_comparisons:
                self._save_comparison(
                    base_name, mask_type, sample.original, sample.mask, algorithm_results
                )

        df = pd.DataFrame(results)
        self._generate_report(df, algorithms)

        return df

    def quick_test(
        self,
        algorithm: InpaintingAlgorithm,
        test_case: str = "lines",
        mask_type: str = "center",
        size: int = 128,
    ) -> None:
        """Quickly test a single algorithm on a single case."""
        # Generate or load the test case
        if test_case.startswith("real"):
            samples = self.dataset.load_real_dataset(n_images=1, size=size, mask_types=[mask_type])
        else:
            samples = self.dataset.generate_synthetic_dataset(
                size=size, force_regenerate=True, mask_types=[mask_type]
            )

        # Get the specific test case
        case_name = f"{test_case}_{mask_type}"
        sample = samples.get(case_name)

        if sample is None:
            raise ValueError(f"Test case {case_name} not found")

        # Run the algorithm
        try:
            logger.info(f"Running {algorithm.name} on {case_name}")

            # Time the execution
            start_time = time.time()
            result = algorithm.inpaint(
                sample.masked / 255.0, sample.mask.astype(np.float32) / 255.0
            )
            exec_time = time.time() - start_time

            # Handle any NaN values in output
            result = np.nan_to_num(result, nan=0.0)  # Convert NaNs to 0
            result = np.clip(result, 0, 1)  # Ensure values in [0,1]
            result = (result * 255).astype(np.uint8)

            # Compute metrics with execution time
            metrics = InpaintingMetrics.compute(
                original=sample.original, result=result, mask=sample.mask, execution_time=exec_time
            )

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # Original
            axes[0, 0].imshow(sample.original, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            # Masked
            masked_viz = sample.masked.copy()
            masked_viz[np.isnan(masked_viz)] = 1.0  # White for visualization
            axes[0, 1].imshow(masked_viz, cmap="gray")
            axes[0, 1].set_title("Masked Input")
            axes[0, 1].axis("off")

            # Result
            axes[1, 0].imshow(result, cmap="gray")
            axes[1, 0].set_title(
                f"Result\nPSNR: {metrics.psnr:.2f}, SSIM: {metrics.ssim:.3f}\nTime: {exec_time:.2f}s"
            )
            axes[1, 0].axis("off")

            # Difference
            diff = np.abs(sample.original.astype(float) - result.astype(float))
            diff = diff / diff.max() if diff.max() > 0 else diff  # Normalize for visualization
            axes[1, 1].imshow(diff, cmap="hot")
            axes[1, 1].set_title("Error Map (brighter = larger error)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.show()

            logger.info(f"PSNR: {metrics.psnr:.2f}")
            logger.info(f"SSIM: {metrics.ssim:.3f}")
            logger.info(f"EMD: {metrics.emd:.3f}")
            logger.info(f"Time: {exec_time:.2f}s")

        except Exception as e:
            logger.error(f"Error running {algorithm.name}: {str(e)}")
            raise

    def _save_individual_result(
        self,
        algorithm_name: str,
        case_name: str,
        mask_name: str,
        image: Image,
        mask: Mask,
        result: Image,
        metrics: InpaintingMetrics,
    ) -> None:
        """Save individual result with visualizations."""
        base_path = self.results_dir / algorithm_name / case_name / mask_name
        base_path.mkdir(parents=True, exist_ok=True)

        # Save result visualization
        plot_inpainting_result(
            original=image,
            mask=mask,
            result=result,
            save_path=base_path / "result.png",
            title=f"{algorithm_name} on {case_name} ({mask_name})",
            metrics=metrics,
            figsize=self.config.figsize,
        )

        # Save error heatmap
        if self.config.save_heatmaps:
            create_error_heatmap(
                original=image,
                result=result,
                mask=mask,
                save_path=base_path / "error_heatmap.png",
                figsize=self.config.figsize,
            )

    def _save_comparison(
        self,
        case_name: str,
        mask_name: str,
        image: Image,
        mask: Mask,
        algorithm_results: dict[str, dict[str, Any]],
    ) -> None:
        """Save side-by-side comparison of all algorithms."""
        results_dict = {
            alg_name: {"original": image, "mask": mask, "result": res["result"]}
            for alg_name, res in algorithm_results.items()
        }

        metrics_dict = {alg_name: res["metrics"] for alg_name, res in algorithm_results.items()}

        comparison_path = self.figures_dir / "comparisons" / f"{case_name}_{mask_name}.png"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)

        plot_multiple_results(
            results_dict,
            save_path=comparison_path,
            metrics_dict=metrics_dict,
            figsize=self.config.figsize,
        )

    def _generate_report(
        self, results: pd.DataFrame, algorithms: list[InpaintingAlgorithm]
    ) -> None:
        """Generate comprehensive report with tables and figures."""
        # Save full results
        results.to_csv(self.metrics_dir / "full_results.csv", index=False)

        # Generate summary statistics
        summary = (
            results.groupby(["Algorithm", "Category"])
            .agg(
                {
                    "PSNR": ["mean", "std"],
                    "SSIM": ["mean", "std"],
                    "EMD": ["mean", "std"],
                    "Time (s)": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Save LaTeX tables
        with open(self.latex_dir / "summary_table.tex", "w") as f:
            f.write(
                summary.to_latex(
                    caption="Summary of Inpainting Results", label="tab:inpainting_summary"
                )
            )

        # Generate figures
        self._plot_metrics_by_category(results)
        self._plot_metrics_distribution(results, algorithms)

    def _plot_metrics_by_category(self, results: pd.DataFrame) -> None:
        """Plot metrics broken down by category."""
        metrics = ["PSNR", "SSIM", "EMD", "Time (s)"]
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)

        for ax, metric in zip(axes.flat, metrics):
            sns.boxplot(data=results, x="Category", y=metric, hue="Algorithm", ax=ax)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "metrics_by_category.pdf", bbox_inches="tight", dpi=self.config.dpi
        )
        plt.close()

    def _plot_metrics_distribution(
        self, results: pd.DataFrame, algorithms: list[InpaintingAlgorithm]
    ) -> None:
        """Plot distribution of metrics across all cases."""
        metrics = ["PSNR", "SSIM", "EMD", "Time (s)"]
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)

        for ax, metric in zip(axes.flat, metrics):
            for algorithm in algorithms:
                sns.kdeplot(
                    data=results[results["Algorithm"] == algorithm.name][metric],
                    label=algorithm.name,
                    ax=ax,
                )
            ax.set_title(f"{metric} Distribution")
            ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "metrics_distribution.pdf", bbox_inches="tight", dpi=self.config.dpi
        )
        plt.close()


if __name__ == "__main__":
    from src.algorithms import (
        EfrosLeungInpainting,
        LamaInpainting,
        LCMInpainting,
        NavierStokesInpainting,
        PatchMatchInpainting,
    )

    image_size = 16
    config = BenchmarkConfig(
        synthetic_size=image_size,
        n_real_images=2,
        real_size=image_size,
        save_individual=True,
        save_comparisons=True,
        save_heatmaps=True,
    )
    benchmark = InpaintingBenchmark(config=config)
    algorithms = [
        # LamaInpainting(),
        # LCMInpainting(),
        EfrosLeungInpainting(window_size=3),
        NavierStokesInpainting(max_iter=100),
        PatchMatchInpainting(patch_size=7, num_iterations=50),
    ]
    results_df = benchmark.run(algorithms)
    logger.info("Benchmark completed! Results saved to data/benchmark_results/")

    # benchmark = InpaintingBenchmark()
    # benchmark.quick_test(NavierStokesInpainting(), test_case="lines", mask_type="center")
    # benchmark.quick_test(NavierStokesInpainting(), test_case="real_0", mask_type="brush")
