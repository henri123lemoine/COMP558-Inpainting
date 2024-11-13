import argparse

from loguru import logger

from src.algorithms.classical.patch_match import PatchMatchInpainting
from src.algorithms.classical.texture_synthesis import EfrosLeungInpainting
from src.experiments.comparison import ComparisonExperiment
from src.utils.logging import setup_logger


def run_experiments(args: argparse.Namespace) -> None:
    setup_logger()
    logger.info("Starting inpainting experiments")

    algorithms = {
        "efros_leung": EfrosLeungInpainting(
            window_size=11, error_threshold=0.1, sigma=1.0, n_candidates=10
        ),
        "patch_match": PatchMatchInpainting(
            patch_size=7, num_iterations=5, search_ratio=0.5, alpha=0.1
        ),
    }

    if args.experiment == "classical_comparison" or args.experiment == "all":
        logger.info("Running classical methods comparison")
        experiment = ComparisonExperiment(
            name="classical_methods_comparison", algorithms=list(algorithms.values())
        )
        experiment.run()

    logger.info("All experiments completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inpainting experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["classical_comparison", "all"],
        default="all",
        help="Experiment to run",
    )
    args = parser.parse_args()

    run_experiments(args)
