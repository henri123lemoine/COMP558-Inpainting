# ruff: noqa: E402
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask
from src.experiments.utils.datasets import InpaintingDataset
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
        algorithms: list[InpaintingAlgorithm],
        output_dir: Path | None = None,
        config: BenchmarkConfig | None = None,
    ):
        self.algorithms = algorithms
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

        # Initialize dataset
        self.dataset = InpaintingDataset(self.dataset_dir)

        # Setup style for all plots
        sns.set_theme()
        sns.set_context("paper")

        logger.info(
            f"Initialized benchmark with {len(algorithms)} algorithms: "
            f"{', '.join(alg.name for alg in algorithms)}"
        )

    def run(self) -> pd.DataFrame:
        """Run complete benchmark suite."""
        results = []

        # Generate test cases
        synthetic_cases = self.dataset.generate_synthetic_dataset(size=self.config.synthetic_size)
        real_cases = self.dataset.load_real_dataset(
            n_images=self.config.n_real_images, size=self.config.real_size
        )

        # Combine all test cases
        all_cases = {**synthetic_cases, **real_cases}
        logger.info(f"Running benchmark on {len(all_cases)} test cases")

        # Process each test case
        for case_name, case_data in tqdm(all_cases.items(), desc="Processing test cases"):
            category = case_data["category"]
            image = case_data["image"]

            for mask_name, mask in case_data["masks"].items():
                # Run each algorithm
                algorithm_results = {}

                for algorithm in self.algorithms:
                    logger.debug(f"Running {algorithm.name} on {case_name} with {mask_name} mask")

                    # Run algorithm and measure time
                    try:
                        start_time = time.time()
                        result = algorithm.inpaint(image, mask)
                        exec_time = time.time() - start_time

                        # Compute metrics
                        metrics = InpaintingMetrics.compute(
                            original=image, result=result, mask=mask, execution_time=exec_time
                        )

                        # Store results
                        algorithm_results[algorithm.name] = {"result": result, "metrics": metrics}

                        # Add to results list
                        results.append(
                            {
                                "Algorithm": algorithm.name,
                                "Case": case_name,
                                "Mask": mask_name,
                                "Category": category,
                                **metrics.to_dict(),
                            }
                        )

                        # Save individual results if requested
                        if self.config.save_individual:
                            self._save_individual_result(
                                algorithm.name, case_name, mask_name, image, mask, result, metrics
                            )

                    except Exception as e:
                        logger.error(
                            f"Error processing {case_name} with {algorithm.name}: {str(e)}"
                        )
                        continue

                # Generate comparisons if we have results
                if algorithm_results and self.config.save_comparisons:
                    self._save_comparison(case_name, mask_name, image, mask, algorithm_results)

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        self._generate_report(df)

        return df

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

    def _generate_report(self, results: pd.DataFrame) -> None:
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
        self._plot_metrics_distribution(results)

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

    def _plot_metrics_distribution(self, results: pd.DataFrame) -> None:
        """Plot distribution of metrics across all cases."""
        metrics = ["PSNR", "SSIM", "EMD", "Time (s)"]
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)

        for ax, metric in zip(axes.flat, metrics):
            for algorithm in self.algorithms:
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
    from src.algorithms.classical import (
        EfrosLeungInpainting,
        NavierStokesInpainting,
        PatchMatchInpainting,
    )

    algorithms = [
        NavierStokesInpainting(),
        EfrosLeungInpainting(window_size=11, error_threshold=0.1),
        PatchMatchInpainting(patch_size=7, num_iterations=10),
    ]

    config = BenchmarkConfig(synthetic_size=128, n_real_images=10, save_heatmaps=True)

    benchmark = InpaintingBenchmark(algorithms=algorithms, config=config)
    results_df = benchmark.run()
    logger.info("Benchmark completed! Results saved to data/benchmark_results/")
