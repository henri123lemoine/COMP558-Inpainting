from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class NavierStokesParams:
    max_iter: int
    dt: float
    nu: float
    K: float


class NavierStokesInpainting(InpaintingAlgorithm):
    """Navier-Stokes based inpainting algorithm.

    Based on:
    Bertalmío, M., Bertozzi, A. L., & Sapiro, G. (2001).
    Navier-stokes, fluid dynamics, and image and video inpainting.
    In Proceedings of the 2001 IEEE Computer Society Conference on
    Computer Vision and Pattern Recognition.
    `https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf`
    """

    def __init__(self, max_iter: int = 500, dt: float = 0.02, nu: float = 0.15, K: float = 2.0):
        super().__init__(name="Navier-Stokes")
        self.params = NavierStokesParams(
            max_iter=max_iter,
            dt=dt,
            nu=nu,
            K=K,
        )

    def _inpaint(self, image: Image, mask: Mask) -> Image:
        """Perform inpainting using the Navier-Stokes algorithm."""
        if image.ndim != 2:
            raise ValueError("Navier-Stokes inpainting requires a grayscale image.")

        height, width = image.shape
        working_image = np.copy(image)
        nan_mask = np.isnan(working_image)

        for i in range(height):
            for j in range(width):
                if nan_mask[i, j]:
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if (
                                0 <= i + di < height
                                and 0 <= j + dj < width
                                and not nan_mask[i + di, j + dj]
                            ):
                                neighbors.append(working_image[i + di, j + dj])
                    working_image[i, j] = np.mean(neighbors) if neighbors else 0.5

        # Now proceed with Navier-Stokes
        smoothness = self.compute_laplacian(working_image, mask)
        v_x, v_y = self.compute_gradients(working_image, mask)

        for iteration in tqdm(range(self.params.max_iter), "Navier-Stokes"):
            smoothness_x, smoothness_y = self.compute_gradients(smoothness, mask)
            grad_smoothness_mag = np.sqrt(smoothness_x**2 + smoothness_y**2)
            g = self.perona_malik(grad_smoothness_mag, self.params.K)
            diffusion = self.compute_laplacian(g * smoothness, mask)

            smoothness_new = smoothness + self.params.dt * (
                -v_x * smoothness_x - v_y * smoothness_y + self.params.nu * diffusion
            )

            # Only update smoothness in the masked region
            smoothness_new[~nan_mask] = self.compute_laplacian(working_image, mask)[~nan_mask]

            b = smoothness_new.flatten()
            data, row, col = [], [], []

            for i in range(height):
                for j in range(width):
                    index = i * width + j
                    if not nan_mask[i, j]:
                        row.append(index)
                        col.append(index)
                        data.append(1)
                        b[index] = working_image[i, j]
                    else:
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

            A = coo_matrix((data, (row, col)), shape=(height * width, height * width))
            result_flat = spsolve(A.tocsr(), b)
            working_image = result_flat.reshape(height, width)
            working_image = np.clip(working_image, 0, 1)

            smoothness = self.compute_laplacian(working_image, mask)
            v_x, v_y = self.compute_gradients(working_image, mask)

            # Convergence check
            if iteration % 10 == 0 or iteration == self.params.max_iter - 1:
                change = np.linalg.norm(smoothness_new - smoothness)
                if change < 1e-6:
                    break

        return working_image

    def compute_gradients(self, image: Image, mask: Mask) -> tuple[Image, Image]:
        """Compute gradients Ix and Iy using finite differences."""
        # check if mask is valid for pixels in x and y direction
        valid_x = mask[:, :-2] * mask[:, 2:]
        valid_y = mask[:-2, :] * mask[2:, :]

        Ix = np.zeros_like(image)
        Iy = np.zeros_like(image)
        Ix[:, 1:-1] = np.where(valid_x > 0, (image[:, 2:] - image[:, :-2]) / 2, 0)
        Iy[1:-1, :] = np.where(valid_y > 0, (image[2:, :] - image[:-2, :]) / 2, 0)
        return Ix, Iy

    def compute_laplacian(self, image: Image, mask: Mask) -> Image:
        """Compute the Laplacian using finite differences."""

        valid = (
            mask[1:-1, :-2] * mask[1:-1, 2:] * mask[:-2, 1:-1] * mask[2:, 1:-1] * mask[1:-1, 1:-1]
        )

        L = np.zeros_like(image)
        L[1:-1, 1:-1] = np.where(
            valid > 0,
            image[1:-1, :-2]
            + image[1:-1, 2:]
            + image[:-2, 1:-1]
            + image[2:, 1:-1]
            - 4 * image[1:-1, 1:-1],
            0,
        )
        return L

    def perona_malik(self, g, K=2):
        return 1 / (1 + (g / K) ** 2)


if __name__ == "__main__":
    inpainter = NavierStokesInpainting()
    inpainter.run_example(scale_factor=1)
