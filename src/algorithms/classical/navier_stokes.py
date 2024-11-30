from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class NavierStokesParams:
    max_iter: int = 100  # Maximum iterations
    dt: float = 0.05  # Time step
    nu: float = 0.1  # Diffusion coefficient
    K: float = 2.0  # Perona-Malik parameter


class NavierStokesInpainting(InpaintingAlgorithm):
    """Navier-Stokes based inpainting algorithm.

    Based on:
    Bertalmío, M., Bertozzi, A. L., & Sapiro, G. (2001).
    Navier-stokes, fluid dynamics, and image and video inpainting.
    In Proceedings of the 2001 IEEE Computer Society Conference on
    Computer Vision and Pattern Recognition.
    `https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf`
    """

    def __init__(self):
        super().__init__(name="Navier-Stokes")
        self.params = NavierStokesParams()

    def inpaint(self, image: Image, mask: Mask) -> Image:
        """Perform inpainting using the Navier-Stokes algorithm."""

        if image.ndim != 2:
            raise ValueError("Navier-Stokes inpainting requires a grayscale image.")
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape.")

        image = image.astype(np.float64)
        if image.max() > 1.0:
            image /= 255.0

        height, width = image.shape
        smoothness = self.compute_laplacian(image)
        v_x, v_y = self.compute_gradients(image)

        for iteration in range(self.params.max_iter):
            smoothness_x, smoothness_y = self.compute_gradients(smoothness)
            grad_smoothness_mag = np.sqrt(smoothness_x**2 + smoothness_y**2)
            g = perona_malik(grad_smoothness_mag, self.params.K)
            diffusion = self.compute_laplacian(g * smoothness)  # check

            smoothness_new = smoothness + self.params.dt * (
                -v_x * smoothness_x - v_y * smoothness_y + self.params.nu * diffusion
            )
            smoothness_new[mask == 0] = self.compute_laplacian(image)[mask == 0]

            b = smoothness_new.flatten()
            data, row, col = [], [], []

            for i in range(height):
                for j in range(width):
                    index = i * width + j
                    if mask[i, j] > 0:
                        indices, values = [], []
                        indices.append(index)
                        values.append(4)
                        if i > 0:
                            indices.append((i - 1) * width + j)
                            values.append(-1)
                        if i < height - 1:
                            indices.append((i + 1) * width + j)
                            values.append(-1)
                        if j > 0:
                            indices.append(i * width + (j - 1))
                            values.append(-1)
                        if j < width - 1:
                            indices.append(i * width + (j + 1))
                            values.append(-1)

                        row.extend([index] * len(indices))
                        col.extend(indices)
                        data.extend(values)
                    else:
                        row.append(index)
                        col.append(index)
                        data.append(1)
                        b[index] = image[i, j]

            A = coo_matrix((data, (row, col)), shape=(height * width, height * width))
            I_flat = spsolve(A.tocsr(), b)
            image = I_flat.reshape(height, width)
            image = np.clip(image, 0, 1)

            smoothness = self.compute_laplacian(image)
            v_x, v_y = self.compute_gradients(image)

            # Convergence check
            if iteration % 10 == 0 or iteration == self.params.max_iter - 1:
                change = np.linalg.norm(smoothness_new - smoothness)
                logger.info(f"Iteration {iteration + 1}/{self.params.max_iter}, Change: {change}")
                if change < 1e-6:
                    logger.info(f"Converged after {iteration + 1} iterations.")
                    break

        return image

    def compute_gradients(self, image: Image) -> tuple[Image, Image]:
        """Compute gradients Ix and Iy using finite differences."""

        Ix = np.zeros_like(image)
        Iy = np.zeros_like(image)
        Ix[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
        Iy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
        return Ix, Iy

    def compute_laplacian(self, image: Image) -> Image:
        """Compute the Laplacian using finite differences."""
        L = np.zeros_like(image)
        L[1:-1, 1:-1] = (
            image[1:-1, :-2]
            + image[1:-1, 2:]
            + image[:-2, 1:-1]
            + image[2:, 1:-1]
            - 4 * image[1:-1, 1:-1]
        )
        return L


def perona_malik(g, K=2):
    return 1 / (1 + (g / K) ** 2)
