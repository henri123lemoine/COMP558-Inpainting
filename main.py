import argparse

from loguru import logger

from src.algorithms.classical.navier_stokes import NavierStokesInpainting
from src.algorithms.classical.patch_match import PatchMatchInpainting
from src.algorithms.classical.texture_synthesis import EfrosLeungInpainting
from src.experiments.comparison import ComparisonExperiment
from src.utils.logging import setup_logger


def get_algorithm(name: str, quick: bool = False):
    """Get algorithm instance by name."""
    if name == "efros_leung":
        return EfrosLeungInpainting(
            window_size=7 if quick else 11,
            error_threshold=0.15 if quick else 0.1,
            sigma=1.0,
            n_candidates=5 if quick else 10,
        )
    elif name == "patch_match":
        return PatchMatchInpainting(
            patch_size=5 if quick else 7,
            num_iterations=3 if quick else 10,
            search_ratio=0.5,
            alpha=0.1,
        )
    elif name == "navier_stokes":
        return NavierStokesInpainting(
            dt=0.02 if quick else 0.005,
            num_iterations=100 if quick else 3000,
            relaxation=1.0,
            anisotropic_lambda=2.0,
            sigma=1.0,
            poisson_iterations=20 if quick else 1000,
            diffusion_iterations=3 if quick else 20,
            diffusion_frequency=10,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def run_experiments(args: argparse.Namespace) -> None:
    setup_logger()
    logger.info("Starting inpainting experiments")

    if args.experiment == "single":
        if not args.algorithm:
            parser.error("--algorithm is required for single experiment")

        algorithm = get_algorithm(args.algorithm, args.quick)
        experiment = ComparisonExperiment(
            name=f"single_{args.algorithm}",
            algorithms=[algorithm],
        )
    elif args.experiment == "classical_comparison" or args.experiment == "all":
        algorithms = [
            get_algorithm("efros_leung", args.quick),
            get_algorithm("patch_match", args.quick),
            get_algorithm("navier_stokes", args.quick),
        ]
        experiment = ComparisonExperiment(
            name="classical_methods_comparison",
            algorithms=algorithms,
        )

    experiment.run()
    logger.info("All experiments completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inpainting experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["classical_comparison", "single", "all"],
        default="all",
        help="Experiment to run",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["efros_leung", "patch_match", "navier_stokes"],
        help="Algorithm to use for single experiment",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run experiments in quick mode with reduced quality",
    )
    args = parser.parse_args()

    run_experiments(args)
