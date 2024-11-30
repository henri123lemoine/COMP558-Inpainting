from dataclasses import dataclass

import cv2
import numpy as np
import PIL.Image
import torch
from diffusers import AutoPipelineForInpainting

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class DeepLearningParams:
    num_inference_steps: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_id: str = "runwayml/stable-diffusion-inpainting"


class StableDiffusionInpainting(InpaintingAlgorithm):
    """Deep Learning-based inpainting using pretrained models."""

    def __init__(
        self,
        num_inference_steps: int = 20,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name="DeepLearning")
        self.params = DeepLearningParams(
            num_inference_steps=num_inference_steps, model_id=model_id, device=device
        )

        # Initialize model
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            self.params.model_id,
            torch_dtype=torch.float16 if self.params.device == "cuda" else torch.float32,
        ).to(self.params.device)

    def inpaint(self, image: Image, mask: Mask, **kwargs) -> Image:
        """Perform inpainting using the deep learning model.

        Args:
            image: Input image normalized to [0, 1]
            mask: Binary mask where 1 indicates pixels to inpaint
            **kwargs: Additional parameters (unused)

        Returns:
            Inpainted image normalized to [0, 1]
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert numpy arrays to PIL images
        image_pil = PIL.Image.fromarray((image * 255).astype("uint8"))
        # For the mask, we need to invert it as the model expects 1 for areas to keep
        mask_pil = PIL.Image.fromarray(((1 - mask) * 255).astype("uint8"))

        # Resize to model's required size (512x512 is standard)
        image_pil = image_pil.resize((512, 512))
        mask_pil = mask_pil.resize((512, 512))

        # Run inference
        output = self.pipeline(
            image=image_pil,
            mask_image=mask_pil,
            prompt="",  # Empty prompt to minimize text conditioning
            num_inference_steps=self.params.num_inference_steps,
        ).images[0]

        # Convert back to numpy array and normalize
        result = np.array(output) / 255.0

        # Convert back to grayscale if input was grayscale
        if len(image.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Resize back to original size if needed
        if result.shape[:2] != image.shape[:2]:
            result = cv2.resize(result, (image.shape[1], image.shape[0]))

        return result


if __name__ == "__main__":
    image = cv2.imread("data/datasets/real/real_1227/image.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("data/datasets/real/real_1227/mask_brush.png", cv2.IMREAD_GRAYSCALE)

    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)

    inpainter = StableDiffusionInpainting()
    result = inpainter.inpaint(image, mask)

    # Convert back to uint8 for display
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.imshow("Result", result_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
