import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask
from src.datasets.images import InpaintingDataset, InpaintSample
from src.datasets.utils import ImageCategory
from src.experiments.utils.metrics import InpaintingMetrics
from src.experiments.utils.visualization import (
    create_error_heatmap,
    generate_benchmark_report,
    plot_inpainting_result,
    plot_multiple_results,
)
from src.settings import DATA_PATH
from src.utils.logging import worker_init

FIGSIZE: tuple[int, int] = (15, 10)
DPI: int = 300


class InpaintingBenchmark:
    """Comprehensive benchmark for inpainting algorithms."""

    def __init__(
        self,
        output_dir: Path | None = None,
    ):
        self.output_dir = output_dir or (DATA_PATH / "benchmark_results")

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

    def run(
        self,
        algorithms: list[InpaintingAlgorithm],
        samples: dict[str, InpaintSample],
        log_level: str = "INFO",
    ) -> pd.DataFrame:
        """Run complete benchmark suite."""
        results = []
        worker_initializer = partial(worker_init, level=log_level)
        with ProcessPoolExecutor(initializer=worker_initializer) as executor:
            futures = [
                executor.submit(self._process_test_case, case_name, sample, algorithms, [])
                for case_name, sample in samples.items()
            ]
            for future in tqdm(futures, desc="Processing cases"):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing a test case: {e}")

        df = pd.DataFrame(results)
        generate_benchmark_report(df, algorithms, self.metrics_dir, FIGSIZE, DPI)
        return df

    def _process_test_case(
        self,
        case_name: str,
        sample: InpaintSample,
        algorithms: list[InpaintingAlgorithm],
        results: list[dict[str, Any]],
    ) -> None:
        """Process a single test case with all algorithms."""
        base_name = case_name
        mask_type = (
            "custom" if sample.category == ImageCategory.CUSTOM else case_name.rsplit("_", 1)[1]
        )

        algorithm_results = {}
        for algorithm in algorithms:
            logger.debug(f"Running {algorithm.name} on {base_name}")

            try:
                start_time = time.time()
                original, _, mask, result = algorithm.inpaint(sample.original, sample.mask)
                exec_time = time.time() - start_time

                metrics = InpaintingMetrics.compute(
                    original=original,
                    result=result,
                    mask=mask,
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

                self._save_individual_result(
                    algorithm_name=algorithm.name,
                    case_name=base_name,
                    mask_name=mask_type,
                    image=sample.original,
                    mask=sample.mask,
                    result=result,
                    metrics=metrics,
                )

            except Exception as e:
                logger.error(f"Error processing {case_name} with {algorithm.name}: {str(e)}")
                continue

        if algorithm_results:
            self._save_comparison(
                base_name, mask_type, sample.original, sample.mask, algorithm_results
            )

        return results

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

        plot_inpainting_result(
            original=image,
            mask=mask,
            result=result,
            save_path=base_path / "result.png",
            title=f"{algorithm_name} on {case_name} ({mask_name})",
            metrics=metrics,
            figsize=FIGSIZE,
        )

        create_error_heatmap(
            original=image,
            result=result,
            mask=mask,
            save_path=base_path / "error_heatmap.png",
            figsize=FIGSIZE,
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
            figsize=FIGSIZE,
        )


if __name__ == "__main__":
    import pandas as pd

    from src.algorithms import EfrosLeungInpainting, NavierStokesInpainting, PatchMatchInpainting
    from src.utils.logging import setup_logger

    setup_logger(level="INFO")
    logger.info("Starting experiments")

    REAL_IMAGE_SIZE = 64
    SYNTHETIC_SIZE = 32
    CUSTOM_IMAGES_SIZE = 64  # 256
    N_REAL_IMAGES = 6
    benchmark = InpaintingBenchmark()

    algorithms = [
        EfrosLeungInpainting(window_size=9),
        NavierStokesInpainting(max_iterations=300),
        PatchMatchInpainting(patch_size=9, num_iterations=3),
    ]

    synthetic_samples = benchmark.dataset.generate_synthetic_dataset(
        size=SYNTHETIC_SIZE,
        mask_types=["brush", "center", "random", "text"],
    )

    real_samples = benchmark.dataset.load_real_dataset(
        n_images=N_REAL_IMAGES,
        size=REAL_IMAGE_SIZE,
        mask_types=["brush", "text"],
    )

    custom_samples = benchmark.dataset.load_custom_dataset(target_size=CUSTOM_IMAGES_SIZE)

    all_samples = {**custom_samples, **real_samples, **synthetic_samples}

    logger.info(f"Running benchmark on {len(all_samples)} test cases")
    results_df = benchmark.run(algorithms, samples=all_samples)
    logger.info("\nFinal results:")
    logger.info(results_df)
    logger.info("Benchmark completed! Results saved to data/benchmark_results/")
