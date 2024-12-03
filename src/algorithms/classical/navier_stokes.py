import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from src.algorithms.base import InpaintingAlgorithm, Image, Mask
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class NavierStokesParams:
    max_iter: int = 300
    dt: float = 0.05
    nu: float = 0.1
    K: float = 2.0


class NavierStokesInpainting(InpaintingAlgorithm):
    """Navier-Stokes based inpainting algorithm."""

    def __init__(self, max_iter: int = 600, dt: float = 0.025, nu: float = 0.1, K: float = 2.0):
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
        smoothness = self.compute_laplacian(image, mask)
        v_x, v_y = self.compute_gradients(image, mask)

        for iteration in tqdm(range(self.params.max_iter), "Navier-Stokes"):
            smoothness_x, smoothness_y = self.compute_gradients(smoothness, mask)
            grad_smoothness_mag = np.sqrt(smoothness_x**2 + smoothness_y**2)
            g = self.perona_malik(grad_smoothness_mag, self.params.K)
            diffusion = self.compute_laplacian(g * smoothness, mask)

            smoothness_new = smoothness + self.params.dt * (
                -v_x * smoothness_x - v_y * smoothness_y + self.params.nu * diffusion
            )
            smoothness_new[mask == 0] = self.compute_laplacian(image, mask)[mask == 0]

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

            smoothness = self.compute_laplacian(image, mask)
            v_x, v_y = self.compute_gradients(image, mask)

            if iteration % 10 == 0 or iteration == self.params.max_iter - 1:
                change = np.linalg.norm(smoothness_new - smoothness)
                if change < 1e-6:
                    break

        return image

    def compute_gradients(self, image: Image, mask: Mask) -> tuple[Image, Image]:
        valid_x = mask[:, :-2] * mask[:, 2:]
        valid_y = mask[:-2, :] * mask[2:, :]

        Ix = np.zeros_like(image)
        Iy = np.zeros_like(image)
        Ix[:, 1:-1] = np.where(valid_x > 0, (image[:, 2:] - image[:, :-2]) / 2, 0)
        Iy[1:-1, :] = np.where(valid_y > 0, (image[2:, :] - image[:-2, :]) / 2, 0)
        return Ix, Iy

    def compute_laplacian(self, image: Image, mask: Mask) -> Image:
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
    parser = argparse.ArgumentParser(description="Run Navier-Stokes Inpainting")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to the inpainting mask")
    args = parser.parse_args()

    # Load the image and mask
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Error: Could not read the image file at {args.image}")
    if mask is None:
        raise FileNotFoundError(f"Error: Could not read the mask file at {args.mask}")

    # Resize the mask to match the image dimensions
    if mask.shape != image.shape:
        print(f"Resizing mask from {mask.shape} to {image.shape}...")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Normalize the image and mask
    image = image / 255.0 if image.max() > 1 else image
    mask = (mask > 0).astype(np.uint8)

    # Run the inpainting algorithm
    inpainter = NavierStokesInpainting(max_iter=50)
    inpainted_image = inpainter._inpaint(image, mask)

    # Save the inpainted image
    output_path = "inpainted_image.png"
    cv2.imwrite(output_path, (inpainted_image * 255).astype(np.uint8))
    print(f"Inpainted image saved to {output_path}")

    # Prepare images for display
    image_display = image if image.max() <= 1 else image / 255.0
    inpainted_image_display = inpainted_image if inpainted_image.max() <= 1 else inpainted_image / 255.0

    # Create a side-by-side comparison
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image_display, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    # Inpainted Image
    plt.subplot(1, 3, 3)
    plt.imshow(inpainted_image_display, cmap='gray')
    plt.title('Inpainted Image')
    plt.axis('off')

    plt.tight_layout()

    # Save the comparison figure
    plt.savefig('comparison.png')

    # Display the figure
    plt.show()

