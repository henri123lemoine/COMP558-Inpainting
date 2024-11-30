from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class DeepLearningParams:
    model_path: str = "Carve/LaMa-ONNX"
    model_file: str = "lama_fp32.onnx"
    input_size: tuple[int, int] = (512, 512)


class DeepLearningInpainting(InpaintingAlgorithm):
    """Deep Learning-based inpainting using LaMa ONNX model."""

    def __init__(
        self,
        model_path: str = "Carve/LaMa-ONNX",
        model_file: str = "lama_fp32.onnx",
        input_size: tuple[int, int] = (512, 512),
    ):
        super().__init__(name="DeepLearning")
        self.params = DeepLearningParams(
            model_path=model_path, model_file=model_file, input_size=input_size
        )

        # Download and load the model
        model_path = hf_hub_download(
            repo_id=self.params.model_path, filename=self.params.model_file
        )
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],  # Use CPU for now
        )

    def preprocess(self, image: Image, mask: Mask) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess inputs for the model."""
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to model's input size
        image_resized = cv2.resize(image, self.params.input_size)
        mask_resized = cv2.resize(mask, self.params.input_size)

        # Add batch dimension and ensure correct format (NCHW)
        image_input = np.transpose(image_resized, (2, 0, 1))[None]
        mask_input = mask_resized[None, None]

        return image_input, mask_input

    def postprocess(self, output: np.ndarray, original_shape: tuple) -> Image:
        """Postprocess model output."""
        # Convert from NCHW to HWC
        result = output[0].transpose(1, 2, 0)

        # Resize back to original shape if needed
        if original_shape[:2] != result.shape[:2]:
            result = cv2.resize(result, (original_shape[1], original_shape[0]))

        # Convert back to grayscale if input was grayscale
        if len(original_shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        return result

    def inpaint(self, image: Image, mask: Mask, **kwargs) -> Image:
        """Perform inpainting using the LaMa model.

        Args:
            image: Input image normalized to [0, 1]
            mask: Binary mask where 1 indicates pixels to inpaint
            **kwargs: Additional parameters (unused)

        Returns:
            Inpainted image normalized to [0, 1]
        """
        # Preprocess inputs
        image_input, mask_input = self.preprocess(image, mask)

        # Get input name from the model
        input_name = self.session.get_inputs()[0].name
        mask_name = self.session.get_inputs()[1].name

        # Run inference
        outputs = self.session.run(
            None,  # Get all outputs
            {input_name: image_input.astype(np.float32), mask_name: mask_input.astype(np.float32)},
        )

        # Postprocess output
        result = self.postprocess(outputs[0], image.shape)
        return result


if __name__ == "__main__":
    # Load test images
    image = cv2.imread("data/datasets/real/real_1227/image.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("data/datasets/real/real_1227/mask_brush.png", cv2.IMREAD_GRAYSCALE)

    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)

    inpainter = DeepLearningInpainting()
    result = inpainter.inpaint(image, mask)

    # Display using matplotlib
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
