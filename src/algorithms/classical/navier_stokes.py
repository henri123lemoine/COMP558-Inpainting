from dataclasses import dataclass
from typing import Final, Optional

import cv2
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class NavierStokesParams:
    """Parameters for Navier-Stokes inpainting algorithm."""
    
    dt: float  # Time step
    num_iterations: int  # Number of iterations
    relaxation: float  # Relaxation parameter for Poisson equation
    anisotropic_lambda: float  # Weight of anisotropic diffusion
    sigma: float  # Gaussian smoothing parameter
    poisson_iterations: int  # Number of iterations for Poisson solver
    diffusion_iterations: int  # Number of anisotropic diffusion iterations per step
    diffusion_frequency: int  # How often to apply anisotropic diffusion

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.num_iterations < 1:
            raise ValueError("Number of iterations must be positive")
        if not 0 < self.relaxation < 2:
            raise ValueError("Relaxation parameter must be between 0 and 2")
        if self.anisotropic_lambda <= 0:
            raise ValueError("Anisotropic lambda must be positive")
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        if self.poisson_iterations < 1:
            raise ValueError("Poisson iterations must be positive")
        if self.diffusion_iterations < 0:
            raise ValueError("Diffusion iterations must be non-negative")
        if self.diffusion_frequency < 1:
            raise ValueError("Diffusion frequency must be positive")


class NavierStokesInpainting(InpaintingAlgorithm):
    """Navier-Stokes based inpainting algorithm.
    
    Based on:
    Bertalmío, M., Bertozzi, A. L., & Sapiro, G. (2001).
    Navier-stokes, fluid dynamics, and image and video inpainting.
    In Proceedings of the 2001 IEEE Computer Society Conference on
    Computer Vision and Pattern Recognition.
    `https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf`
    """

    # Constants for numerical stability and convergence
    MAX_ERROR: Final[float] = 1e-6  # Maximum error for convergence
    MIN_GRADIENT: Final[float] = 1e-6  # Minimum gradient magnitude
    MAX_POISSON_ERROR: Final[float] = 1e-4  # Maximum error for Poisson solver

    def __init__(
        self,
        dt: float = 0.01,
        num_iterations: int = 300,
        relaxation: float = 1.0,
        anisotropic_lambda: float = 2.0,
        sigma: float = 1.0,
        poisson_iterations: int = 50,
        diffusion_iterations: int = 5,
        diffusion_frequency: int = 10,
    ) -> None:
        """Initialize Navier-Stokes inpainting algorithm."""
        super().__init__("NavierStokes")
        
        self.params = NavierStokesParams(
            dt=dt,
            num_iterations=num_iterations,
            relaxation=relaxation,
            anisotropic_lambda=anisotropic_lambda,
            sigma=sigma,
            poisson_iterations=poisson_iterations,
            diffusion_iterations=diffusion_iterations,
            diffusion_frequency=diffusion_frequency,
        )

        logger.debug(
            f"Initialized Navier-Stokes inpainting with: dt={dt}, "
            f"num_iterations={num_iterations}, relaxation={relaxation}, "
            f"anisotropic_lambda={anisotropic_lambda}"
        )

    def _compute_vorticity(self, image: Image, mask: Optional[Mask] = None) -> Image:
        """Compute vorticity (Laplacian of image intensity)."""
        # Ensure image is in correct format
        if image.dtype != np.float64:
            image = image.astype(np.float64)
            
        # Compute Laplacian using finite differences
        laplacian = np.zeros_like(image)
        laplacian[1:-1, 1:-1] = (
            image[:-2, 1:-1] +  # up
            image[2:, 1:-1] +   # down
            image[1:-1, :-2] +  # left
            image[1:-1, 2:] -   # right
            4 * image[1:-1, 1:-1]  # center
        )
        
        if mask is not None:
            # Only compute vorticity outside the mask
            laplacian[mask > 0] = 0
            
        return laplacian

    def _solve_poisson(
        self,
        vorticity: Image,
        boundary_image: Image,
        mask: Mask,
    ) -> Image:
        """Solve Poisson equation ΔI = w with Dirichlet boundary conditions."""
        image = boundary_image.copy()
        error = float('inf')
        
        # Jacobi iteration method
        for iter_idx in range(self.params.poisson_iterations):
            prev_image = image.copy()
            
            # Update interior points using 5-point stencil
            image[1:-1, 1:-1] = (
                prev_image[:-2, 1:-1] +  # up
                prev_image[2:, 1:-1] +   # down
                prev_image[1:-1, :-2] +  # left
                prev_image[1:-1, 2:] -   # right
                vorticity[1:-1, 1:-1]    # source term
            ) / 4.0
            
            # Enforce boundary conditions
            image[~mask] = boundary_image[~mask]
            
            # Check convergence
            error = np.max(np.abs(image - prev_image))
            if error < self.MAX_POISSON_ERROR:
                logger.debug(f"Poisson solver converged after {iter_idx + 1} iterations")
                break
        
        # if error >= self.MAX_POISSON_ERROR:
        #     logger.warning(
        #         f"Poisson solver did not converge after {self.params.poisson_iterations} "
        #         f"iterations. Final error: {error:.2e}"
        #     )
        return image

    def _anisotropic_diffusion(
        self,
        image: Image,
        mask: Mask,
        num_iterations: int,
    ) -> Image:
        """Apply anisotropic diffusion to sharpen edges."""
        result = image.copy()
        
        for _ in range(num_iterations):
            # Create padded arrays for computing gradients
            padded = np.pad(result, pad_width=1, mode='edge')
            
            # Compute gradients using central differences
            dx = np.zeros_like(result)
            dy = np.zeros_like(result)
            
            # Forward differences
            dx_forward = padded[1:-1, 2:] - padded[1:-1, 1:-1]  # right - center
            dy_forward = padded[2:, 1:-1] - padded[1:-1, 1:-1]  # down - center
            
            # Backward differences
            dx_backward = padded[1:-1, 1:-1] - padded[1:-1, :-2]  # center - left
            dy_backward = padded[1:-1, 1:-1] - padded[:-2, 1:-1]  # center - up
            
            # Compute diffusion coefficients
            def g(grad):
                return 1.0 / (1.0 + (np.abs(grad) / self.params.anisotropic_lambda) ** 2)
            
            # Compute fluxes
            flux_x = g(dx_forward) * dx_forward - g(dx_backward) * dx_backward
            flux_y = g(dy_forward) * dy_forward - g(dy_backward) * dy_backward
            
            # Update only masked region
            update = self.params.dt * (flux_x + flux_y)
            result[mask > 0] += update[mask > 0]
            
            # Ensure values stay in valid range [0, 1]
            np.clip(result, 0, 1, out=result)
        
        return result

    def _transport_vorticity(
        self,
        vorticity: Image,
        velocity: Image,
        mask: Mask,
    ) -> Image:
        """Transport vorticity using upwind scheme."""
        result = vorticity.copy()
        
        # Compute velocity components using central differences
        u = np.zeros_like(velocity)
        v = np.zeros_like(velocity)
        
        u[1:-1, 1:-1] = -(velocity[1:-1, 2:] - velocity[1:-1, :-2]) / 2
        v[1:-1, 1:-1] = (velocity[2:, 1:-1] - velocity[:-2, 1:-1]) / 2
        
        # Masked region only
        mask_interior = mask[1:-1, 1:-1]
        if not np.any(mask_interior):
            return result
            
        # Upwind differencing
        dx = np.zeros_like(result[1:-1, 1:-1])
        dy = np.zeros_like(result[1:-1, 1:-1])
        
        # x-direction
        pos_u = u[1:-1, 1:-1] > 0
        dx[pos_u] = (vorticity[1:-1, 1:-1] - vorticity[1:-1, :-2])[pos_u]
        dx[~pos_u] = (vorticity[1:-1, 2:] - vorticity[1:-1, 1:-1])[~pos_u]
        
        # y-direction
        pos_v = v[1:-1, 1:-1] > 0
        dy[pos_v] = (vorticity[1:-1, 1:-1] - vorticity[:-2, 1:-1])[pos_v]
        dy[~pos_v] = (vorticity[2:, 1:-1] - vorticity[1:-1, 1:-1])[~pos_v]
        
        # Update vorticity
        update = -self.params.dt * (u[1:-1, 1:-1] * dx + v[1:-1, 1:-1] * dy)
        result[1:-1, 1:-1][mask_interior] += update[mask_interior]
        
        return result

    def inpaint(
        self,
        image: Image,
        mask: Mask,
        **kwargs,
    ) -> Image:
        """Inpaint using Navier-Stokes fluid dynamics."""
        # Validate inputs
        if len(image.shape) != 2:
            raise ValueError("Only grayscale images are supported")
            
        h, w = image.shape
        if h < 3 or w < 3:
            raise ValueError(f"Image dimensions ({h}, {w}) must be at least 3x3")
            
        logger.info(
            f"Starting Navier-Stokes inpainting: {h}x{w} image, "
            f"{np.sum(mask > 0)} pixels to inpaint"
        )

        # Initialize result
        result = image.copy()
        if result.dtype != np.float64:
            result = result.astype(np.float64)
            
        current_mask = mask.astype(bool)
        
        # Main iteration loop
        for iter_idx in tqdm(range(self.params.num_iterations), desc="Navier-Stokes"):
            # Compute vorticity
            vorticity = self._compute_vorticity(result, current_mask)
            
            # Transport vorticity
            vorticity = self._transport_vorticity(vorticity, result, current_mask)
            
            # Solve Poisson equation to recover image
            result = self._solve_poisson(vorticity, result, current_mask)
            
            # Apply anisotropic diffusion periodically
            if (
                self.params.diffusion_iterations > 0 and
                iter_idx % self.params.diffusion_frequency == 0
            ):
                result = self._anisotropic_diffusion(
                    result,
                    current_mask,
                    self.params.diffusion_iterations,
                )
            
            # Update mask for next iteration (optional Gaussian smoothing)
            if iter_idx < self.params.num_iterations - 1:
                current_mask = gaussian_filter(
                    current_mask.astype(float),
                    sigma=self.params.sigma,
                ) > 0.5

        logger.info("Navier-Stokes inpainting completed")
        return result
