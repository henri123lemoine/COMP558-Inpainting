from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from loguru import logger

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class LCMInpaintingParams:
    """Parameters for LCM inpainting."""

    base_model: str = "runwayml/stable-diffusion-inpainting"
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    guidance_scale: float = 1.2  # Between 1.0 and 2.0 for LCM
    num_inference_steps: int = 4  # Between 4 and 8 for LCM
    prompt: str = ""
    negative_prompt: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    seed: int | None = None


class LCMInpainting(InpaintingAlgorithm):
    """Fast inpainting using Latent Consistency Model with LoRA adapters.

    Based on the LCM paper: https://arxiv.org/abs/2310.04378
    """

    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-inpainting",
        lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5",
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        prompt: str = "",
        negative_prompt: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ):
        """Initialize LCM inpainting model.

        Args:
            base_model: HuggingFace model ID for base inpainting model
            lcm_lora_id: HuggingFace model ID for LCM LoRA weights
            guidance_scale: Guidance scale for classifier-free guidance (1.0-2.0)
            num_inference_steps: Number of denoising steps (4-8)
            prompt: Text prompt to guide inpainting
            negative_prompt: Optional negative prompt
            device: Device to run on ('cuda' or 'cpu')
            dtype: Model dtype (float16 or float32)
            seed: Random seed for reproducibility
        """
        super().__init__(name="LCM")

        self.params = LCMInpaintingParams(
            base_model=base_model,
            lcm_lora_id=lcm_lora_id,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device if device else ("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=dtype
            if dtype
            else (torch.float16 if torch.cuda.is_available() else torch.float32),
            seed=seed,
        )

        logger.info(
            f"Initializing LCM Inpainting with device={self.params.device}, dtype={self.params.dtype}"
        )

        # Initialize pipeline
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.params.base_model,
                torch_dtype=self.params.dtype,
                variant="fp16" if self.params.dtype == torch.float16 else None,
            ).to(self.params.device)

            # Set LCM scheduler
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

            # Load LCM-LoRA weights
            self.pipe.load_lora_weights(self.params.lcm_lora_id)

            logger.info("Successfully initialized LCM inpainting pipeline")

        except Exception as e:
            logger.error(f"Failed to initialize LCM pipeline: {str(e)}")
            raise

    def inpaint(
        self,
        image: Image,
        mask: Mask,
        prompt: str | None = None,
        **kwargs,
    ) -> Image:
        original_height, original_width = image.shape[:2]

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)

        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        try:
            # Run inference
            output = self.pipe(
                prompt=prompt or self.params.prompt,
                negative_prompt=self.params.negative_prompt,
                image=image_uint8,
                mask_image=mask_uint8,
                num_inference_steps=self.params.num_inference_steps,
                guidance_scale=self.params.guidance_scale,
                generator=torch.manual_seed(self.params.seed) if self.params.seed else None,
            ).images[0]

            # Convert and resize
            result = np.array(output).astype(np.float32) / 255.0

            # Resize to original dimensions
            result = cv2.resize(result, (original_width, original_height))

            # Convert to grayscale
            if len(result.shape) == 3:
                result_gray = np.mean(result, axis=2)

            # Create the final blend
            final_result = image[..., 0].copy()  # Take first channel since they're identical
            final_result[mask > 0.5] = result_gray[mask > 0.5]

            return final_result

        except Exception as e:
            logger.error(f"Inpainting failed: {str(e)}")
            raise


if __name__ == "__main__":
    inpainter = LCMInpainting()

    image = cv2.imread("data/datasets/real/real_1227/image.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("data/datasets/real/real_1227/mask_brush.png", cv2.IMREAD_GRAYSCALE)
    print(image.shape, mask.shape)

    image = image.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    result = inpainter.inpaint(image, mask)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Mask")
    ax2.axis("off")

    ax3.imshow(result, cmap="gray")
    ax3.set_title("Result")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()
