from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from loguru import logger

from src.algorithms.base import Image, InpaintingAlgorithm, Mask


@dataclass(frozen=True)
class LamaParams:
    model_path: str = "Carve/LaMa-ONNX"
    model_file: str = "lama_fp32.onnx"
    input_size: tuple[int, int] = (512, 512)


class LamaInpainting(InpaintingAlgorithm):
    """Deep Learning-based inpainting using LaMa ONNX model."""

    def __init__(
        self,
        model_path: str = "Carve/LaMa-ONNX",
        model_file: str = "lama_fp32.onnx",
        input_size: tuple[int, int] = (512, 512),
    ):
        super().__init__(name="Lama")
        self.params = LamaParams(
            model_path=model_path, model_file=model_file, input_size=input_size
        )

        model_path = hf_hub_download(
            repo_id=self.params.model_path, filename=self.params.model_file
        )
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    def preprocess(self, image: Image, mask: Mask) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess inputs for the model."""
        mask_bool = mask > 0.5
        noisy_image = image.copy()
        if len(image.shape) == 2:
            # Generate noise matching the statistics of non-masked regions
            valid_mean = np.mean(image[~mask_bool])
            valid_std = np.std(image[~mask_bool])
            noise = np.random.normal(valid_mean, valid_std, image.shape)
            noisy_image[mask_bool] = np.clip(noise[mask_bool], 0, 1)

        if len(noisy_image.shape) == 2:
            noisy_image = np.stack([noisy_image] * 3, axis=-1)

        # Normalize image to [-1, 1]
        if noisy_image.dtype == np.uint8:
            noisy_image = noisy_image.astype(np.float32) / 127.5 - 1.0
        else:  # Already float32 in [0,1]
            noisy_image = noisy_image * 2 - 1

        # Ensure mask is binary and inverted (0 for regions to inpaint)
        mask_inverted = 1.0 - (mask > 0.5).astype(np.float32)

        # Resize to model's input size
        image_resized = cv2.resize(noisy_image, self.params.input_size)
        mask_resized = cv2.resize(
            mask_inverted, self.params.input_size, interpolation=cv2.INTER_NEAREST
        )

        # ONNX model expects NCHW tensor format
        image_input = np.transpose(image_resized, (2, 0, 1))[None]
        mask_input = mask_resized[None, None]

        # Check final input shapes
        logger.info(f"Final input shapes - Image: {image_input.shape}, Mask: {mask_input.shape}")
        return image_input, mask_input

    def postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple,
        original_image: np.ndarray,
        mask: np.ndarray,
    ) -> Image:
        """Postprocess model output."""
        result = output[0].transpose(1, 2, 0) / 255.0  # Convert from [0, 255] to [0, 1]
        result = np.clip(result, 0, 1)
        if original_shape[:2] != result.shape[:2]:
            result = cv2.resize(result, (original_shape[1], original_shape[0]))
        if len(original_shape) == 2:
            result = np.mean(result, axis=2)
            mask_bool = mask > 0.5
            final_result = original_image.copy()
            final_result[mask_bool] = result[mask_bool]
            return final_result
        return result

    def inpaint(self, image: Image, mask: Mask, **kwargs) -> Image:
        """Perform inpainting using the LaMa model."""
        logger.info(
            f"Starting inpainting with image shape {image.shape} and mask shape {mask.shape}"
        )
        original_shape = image.shape
        original_image = image.copy()
        image_input, mask_input = self.preprocess(image, mask)
        input_name = self.session.get_inputs()[0].name
        mask_name = self.session.get_inputs()[1].name
        logger.info(
            f"Model inputs - {input_name}: {image_input.shape}, {mask_name}: {mask_input.shape}"
        )
        try:
            outputs = self.session.run(
                None,
                {
                    input_name: image_input.astype(np.float32),
                    mask_name: mask_input.astype(np.float32),
                },
            )
            logger.info(f"Model output shape: {outputs[0].shape}")
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            raise

        result = self.postprocess(outputs[0], original_shape, original_image, mask)
        if result.shape != original_shape:
            logger.error(f"Shape mismatch! Original: {original_shape}, Result: {result.shape}")
            raise ValueError("Output shape does not match input shape")
        return result


if __name__ == "__main__":
    inpainter = LamaInpainting()
    inpainter.run_example()
