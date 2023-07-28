from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from onnxruntime import InferenceSession
from PIL import Image

from .transform_dbnet import decode_dbnet


def prepare_inputs(image: Image.Image, image_size: Tuple[int, int]):
    """
    Prepare the input image for model inference.

    Args:
        image (Image.Image):
            The input image.
        image_size (Tuple[int, int]):
            The target size of the image in (width, height) format.

    Returns:
        image (np.ndarray):
            The processed input image as a numpy array,
            The image's shape is [C, H, W], dtype is float32 and values are scaled to 0, 1.
    """
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image = np.array(image, dtype="float32") / 255
    image = image.transpose((2, 0, 1))
    return image


@dataclass
class PredictorONNX:
    """
    ONNX model predictor for DBNet model.

    Args:
        onnx_path (str):
            The path to the ONNX model file.
        expand_rate (float, optional):
            The expansion rate used for expanding the bounding boxes. Default is 1.5.
        expand (bool, optional):
            A boolean value indicating whether to expand the bounding boxes. Default is True.
        max_distance (Optional[Union[int, float]], optional):
            The maximum distance for expanding the bounding boxes.
            If provided as a float, it will be scaled by the image size.
            If None, no maximum distance constraint is applied. Default is None.
        min_box_size (Optional[Union[int, float]], optional):
            The minimum size of the bounding boxes.
            If provided as a float, it will be scaled by the image size.
            If None, no minimum size constraint is applied. Default is None.
        threshold (float, optional):
            The threshold value for binarizing the probability maps. Default is 0.02.

    Methods:
        predict(image: Image, post_process: bool = True) -> Dict:
            Run inference on the input image and return the predicted results.

    Attributes:
        model (InferenceSession):
            The ONNX InferenceSession object.
        image_size (Tuple[int, int]):
            The target image size expected by the model.
            This is collected from onnx file input shape.
        input_name (str):
            The name of the input tensor expected by the model.
    """

    onnx_path: str
    expand_rate: float = 1.5
    expand: bool = True
    max_distance: Optional[Union[int, float]] = None
    min_box_size: Optional[Union[int, float]] = None
    threshold: float = 0.02

    def __post_init__(self):
        """
        Load ONNX model and collect additional information.
        """
        self.model = InferenceSession(self.onnx_path)
        model_input = self.model.get_inputs()[0]
        self.image_size = (model_input.shape[-1], model_input.shape[-2])
        self.input_name = model_input.name

    def predict(self, image: Image, post_process: bool = True):
        """
        Run inference on the input image and return the predicted results.

        Args:
            image (Image):
                The input image to perform inference on.
            post_process (bool, optional):
                A boolean value indicating whether to apply post-processing
                to the predicted results. If True, the bounding box polygons,
                classes, and scores will be computed using the `decode_dbnet` function.
                If False, only the probability maps will be returned. Default is True.

        Returns:
            Dict:
                A dictionary containing the predicted results.
                If `post_process` is True, the dictionary will include the following keys:

                boxes:
                    A list of bounding box in polygons format.
                    The polygons are rescaled to the image size.
                classes:
                    A list of class indices corresponding to each bounding box.
                scores:
                    A list of scores representing the confidence of each bounding box.
                    For now, score is not implemented, so the scores will be all 1.
                proba_maps:
                    The probability maps obtained from the model.

                If `post_process` is False, the dictionary will only include the "proba_maps" key.
        """
        # Preprocess
        image = image.convert("RGB")
        W, H = image.size
        image_np = prepare_inputs(image, self.image_size)
        image_np = image_np[np.newaxis, ...]

        # Forward
        (batch_outputs,) = self.model.run(None, {self.input_name: image_np})
        proba_maps = batch_outputs[0]

        # Post process
        if post_process:
            boxes, classes, scores = decode_dbnet(
                proba_maps,
                expand_rate=self.expand_rate,
                expand=self.expand,
                min_box_size=self.min_box_size,
                max_distance=self.max_distance,
                threshold=self.threshold,
            )
            boxes = [[(x * W, y * H) for (x, y) in polygon] for polygon in boxes]

        # Results
        results = {}
        if post_process:
            results["boxes"] = boxes
            results["classes"] = classes
            results["scores"] = scores
        results["proba_maps"] = proba_maps
        return results
