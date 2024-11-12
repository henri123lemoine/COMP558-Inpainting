from loguru import logger

from src.algorithms.classical.texture_synthesis import EfrosLeungInpainting
from src.experiments.texture_synthesis import TextureSynthesisExperiment
from src.utils.logging import setup_logger


def main():
    """Run inpainting experiments."""
    setup_logger()
    logger.info("Starting inpainting experiments")

    # Run texture synthesis experiment
    logger.info("Running texture synthesis experiment")
    algorithm = EfrosLeungInpainting(
        window_size=11, error_threshold=0.1, sigma=1.0, n_candidates=10
    )
    experiment = TextureSynthesisExperiment(name="texture_synthesis_test", algorithm=algorithm)
    experiment.run()


if __name__ == "__main__":
    main()
