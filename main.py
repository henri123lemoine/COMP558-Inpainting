from loguru import logger

from src.algorithms.classical.patch_match import PatchMatchInpainting
from src.algorithms.classical.texture_synthesis import EfrosLeungInpainting
from src.experiments.comparison import ComparisonExperiment
from src.utils.logging import setup_logger


def main():
    """Run inpainting experiments."""
    setup_logger()
    logger.info("Starting inpainting experiments")

    algorithms = [
        EfrosLeungInpainting(window_size=11, error_threshold=0.1, sigma=1.0, n_candidates=10),
        PatchMatchInpainting(patch_size=7, num_iterations=5, search_ratio=0.5, alpha=0.1),
    ]
    experiment = ComparisonExperiment(name="classical_methods_comparison", algorithms=algorithms)
    experiment.run()


if __name__ == "__main__":
    main()
