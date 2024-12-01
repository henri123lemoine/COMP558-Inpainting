from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class NavierStokesParams:
    """Parameters for Navier-Stokes inpainting algorithm."""

    max_iter: int = 300
    dt: float = 0.1
    nu: float = 0.2
    K: float = 1.5
    tol: float = 1e-6

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{field_name} must be positive")


class NavierStokesInpainting(InpaintingAlgorithm):
    """Navier-Stokes based inpainting algorithm.

    Based on:
    Bertalmío, M., Bertozzi, A. L., & Sapiro, G. (2001).
    Navier-stokes, fluid dynamics, and image and video inpainting.
    In Proceedings of the 2001 IEEE Computer Society Conference on
    Computer Vision and Pattern Recognition.
    `https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf`
    """

    def __init__(
        self,
        max_iter: int = 2000,
        dt: float = 0.01,
        nu: float = 0.05,
        K: float = 0.7,
        tol: float = 1e-8,
    ):
        super().__init__(name="Navier-Stokes")
        self.params = NavierStokesParams(max_iter=max_iter, dt=dt, nu=nu, K=K, tol=tol)

    def _inpaint(self, image: Image, mask: Mask) -> Image:
        """Perform inpainting using the Navier-Stokes algorithm."""
        height, width = image.shape
        smoothness = self.compute_laplacian(image)
        v_x, v_y = self.compute_gradients(image)

        for _ in tqdm(range(self.params.max_iter), desc="Navier-Stokes"):
            smoothness_x, smoothness_y = self.compute_gradients(smoothness)
            grad_smoothness_mag = np.sqrt(smoothness_x**2 + smoothness_y**2)
            g = self.perona_malik(grad_smoothness_mag, self.params.K)
            diffusion = self.compute_laplacian(g * smoothness)

            smoothness_new = smoothness + self.params.dt * (
                -v_x * smoothness_x - v_y * smoothness_y + self.params.nu * diffusion
            )
            smoothness_new[~mask] = self.compute_laplacian(image)[~mask]

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
            change = np.linalg.norm(smoothness_new - smoothness)
            if change < self.params.tol:
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

    def perona_malik(self, g, K=2):
        return 1 / (1 + (g / K) ** 2)


if __name__ == "__main__":
    inpainter = NavierStokesInpainting()
    inpainter.run_example()
