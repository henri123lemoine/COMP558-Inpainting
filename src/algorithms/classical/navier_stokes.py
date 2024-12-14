from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, InpaintingParams, Mask


@dataclass(frozen=True)
class NavierStokesParams(InpaintingParams):
    max_iterations: int = 1000
    dt: float = 0.02
    nu: float = 0.15
    K: float = 2.0


class NavierStokesInpainting(InpaintingAlgorithm):
    """Navier-Stokes based inpainting algorithm.

    Based on:
    Bertalmío, M., Bertozzi, A. L., & Sapiro, G. (2001).
    Navier-stokes, fluid dynamics, and image and video inpainting.
    In Proceedings of the 2001 IEEE Computer Society Conference on
    Computer Vision and Pattern Recognition.
    `https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf`
    """

    params_class = NavierStokesParams

    def __init__(self, **kwargs):
        super().__init__(name="Navier-Stokes", **kwargs)

    def _inpaint(self, image: Image, mask: Mask) -> Image:
        """Perform inpainting using the Navier-Stokes algorithm."""
        height, width, channels = image.shape
        working_image = np.copy(image)
        nan_mask = np.isnan(working_image)

        # Initial fill with local mean while preserving color relationships
        for i in range(height):
            for j in range(width):
                if np.any(nan_mask[i, j]):
                    neighbors = []
                    weights = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if (
                                0 <= i + di < height
                                and 0 <= j + dj < width
                                and not np.any(nan_mask[i + di, j + dj])
                            ):
                                neighbors.append(working_image[i + di, j + dj])
                                dist = np.sqrt(di**2 + dj**2)
                                weights.append(1.0 / (dist + 1e-6))

                    if neighbors:
                        weights = np.array(weights)
                        weights = weights / np.sum(weights)
                        working_image[i, j] = np.average(neighbors, axis=0, weights=weights)
                    else:
                        working_image[i, j] = np.mean(
                            working_image[~np.any(nan_mask, axis=-1)], axis=0
                        )

        # Compute coupled smoothness across channels
        smoothness = np.zeros_like(working_image)
        for c in range(channels):
            smoothness[..., c] = self.compute_laplacian(working_image[..., c], mask)

        # Compute coupled gradients that preserve color relationships
        v_x = np.zeros_like(working_image)
        v_y = np.zeros_like(working_image)
        for c in range(channels):
            v_x[..., c], v_y[..., c] = self.compute_gradients(working_image[..., c], mask)

        # Scale gradients by local color variance to preserve edges
        edge_strength = np.sum(v_x**2 + v_y**2, axis=-1, keepdims=True)
        edge_strength = self.perona_malik(np.sqrt(edge_strength), self.params.K)

        for iteration in tqdm(range(self.params.max_iterations), "Navier-Stokes"):
            # Compute coupled evolution
            smoothness_x = np.zeros_like(working_image)
            smoothness_y = np.zeros_like(working_image)
            for c in range(channels):
                smoothness_x[..., c], smoothness_y[..., c] = self.compute_gradients(
                    smoothness[..., c], mask
                )

            grad_smoothness_mag = np.sqrt(
                np.sum(smoothness_x**2 + smoothness_y**2, axis=-1, keepdims=True)
            )
            g = self.perona_malik(grad_smoothness_mag, self.params.K)

            diffusion = np.zeros_like(working_image)
            for c in range(channels):
                diffusion[..., c] = self.compute_laplacian(g[..., 0] * smoothness[..., c], mask)

            # Update all channels together to preserve color relationships
            smoothness_new = smoothness + self.params.dt * (
                -v_x * smoothness_x * edge_strength
                - v_y * smoothness_y * edge_strength
                + self.params.nu * diffusion
            )

            # Only update smoothness in the masked region
            for c in range(channels):
                smoothness_new[..., c][~nan_mask[..., c]] = self.compute_laplacian(
                    working_image[..., c], mask
                )[~nan_mask[..., c]]

            # Solve the coupled system
            for c in range(channels):
                b = smoothness_new[..., c].flatten()
                data, row, col = [], [], []

                for i in range(height):
                    for j in range(width):
                        index = i * width + j
                        if not nan_mask[i, j, c]:
                            row.append(index)
                            col.append(index)
                            data.append(1)
                            b[index] = working_image[i, j, c]
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
                working_image[..., c] = result_flat.reshape(height, width)

            working_image = np.clip(working_image, 0, 1)

            # Update coupled terms
            for c in range(channels):
                smoothness[..., c] = self.compute_laplacian(working_image[..., c], mask)
                v_x[..., c], v_y[..., c] = self.compute_gradients(working_image[..., c], mask)

            # Convergence check across all channels
            if iteration % 10 == 0 or iteration == self.params.max_iterations - 1:
                change = np.linalg.norm(smoothness_new - smoothness)
                if change < 1e-6:
                    break

        return working_image.squeeze()

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
    inpainter.run()
