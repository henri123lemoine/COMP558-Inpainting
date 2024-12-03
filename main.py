from loguru import logger

from src.algorithms import (
    EfrosLeungInpainting,
    LamaInpainting,
    LCMInpainting,
    NavierStokesInpainting,
    PatchMatchInpainting,
)
from src.experiments.benchmark import BenchmarkConfig, InpaintingBenchmark
from src.utils.logging import setup_logger

if __name__ == "__main__":
    setup_logger()
    logger.info("All experiments completed")

    image_size = 64
    config = BenchmarkConfig(
        synthetic_size=image_size,
        n_real_images=2,
        real_size=image_size,
        save_individual=True,
        save_comparisons=True,
        save_heatmaps=True,
        custom_size=(image_size, image_size),
    )
    benchmark = InpaintingBenchmark(config=config)

    algorithms = [
        # LamaInpainting(),
        # LCMInpainting(),
        EfrosLeungInpainting(),
        NavierStokesInpainting(),
        PatchMatchInpainting(),
    ]

    synthetic_samples = benchmark.dataset.generate_synthetic_dataset(
        size=benchmark.config.synthetic_size
    )
    real_samples = benchmark.dataset.load_real_dataset(
        n_images=benchmark.config.n_real_images,
        size=benchmark.config.real_size,
    )
    custom_samples = benchmark.dataset.load_custom_dataset(target_size=benchmark.config.custom_size)
    all_samples = {**synthetic_samples, **real_samples, **custom_samples}

    logger.info(f"Running benchmark on {len(all_samples)} test cases")
    results_df = benchmark.run(algorithms, samples=all_samples)
    print("\nFinal results:")
    print(results_df)
    logger.info("Benchmark completed! Results saved to data/benchmark_results/")
