import json
import warnings
from typing import List, Tuple
from os import path

from PIL import Image
from pyrsistent import pvector, PVector

from . import box_utils as bu

warnings.simplefilter("once", UserWarning)


def parse_labelme(
    sample_path: str, class2str: List[str]
) -> [Image.Image, PVector, PVector]:
    """Load a labelme json.

    Args:
        sample_path (str):
            Path to sample labelme json.
        class2str (List[str]):
            List of class names, this is required to convert labelme
            class to number.

    Returns:
        image (Pil.Image.Image):
            The image
        boxes (List[List[Tuple[int, int]]]):
            List of boxes in polygon format
        classes (List[int]):
            List of classes according to the box.
    """
    # Read data
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)
    root_path = path.dirname(sample_path)
    shapes = data["shapes"]
    image_path = data["imagePath"]
    image_data = data["imageData"]

    # Load image
    image_path = path.join(root_path, image_path)
    image = Image.open(image_path)
    width, height = image.size

    # The target boxes and classes
    boxes = pvector()
    classes = pvector()
    for shape in shapes:
        # Target boxes
        if shape["shape_type"] == "rectangle":
            [x1, y1], [x2, y2] = shape["points"]
            box = (x1, y1, x2, y2)
            box = bu.xyxy2polygon(box)
            box = bu.scale_from(box, width, height)
        elif shape["shape_type"] == "polygon":
            box = shape["points"]
            box = bu.scale_from(box, width, height)
        else:
            continue

        # Target classes
        label = shape["label"]
        if label not in class2str:
            msg = f"Cannot find the label {label}, ignoring"
            warnings.warn(msg, UserWarning)
        else:
            class_idx = class2str.index(label)
            classes = classes.append(class_idx)
            boxes = boxes.append(box)

    # Correctness check
    assert len(classes) == len(boxes)

    return image, boxes, classes


def parse_labelme_list(
    index_path: str,
    class2str: List[str],
) -> List[Tuple[Image.Image, List, List]]:
    """Parse a list of Labelme JSON files and return a list of samples.

    Args:
        index_path (str):
            The path to the index file containing a list of Labelme JSON file paths.
        class2str (List[str]):
            A list mapping class indices to class names (strings).

    Returns:
        List[Tuple[Image.Image, List, List]]:
            A list of samples. Each sample is a tuple contains the image,
            object bounding boxes and the object classes.
    """
    root = path.dirname(index_path)
    with open(index_path) as f:
        sample_files = [line.strip() for line in f.readlines()]
        sample_files = [line for line in sample_files if len(line) > 0]

    samples = []
    for idx, sample_file in enumerate(sample_files):
        sample_file = path.join(root, sample_file)
        sample = parse_labelme(sample_file, class2str)
        samples.append(sample)
    return samples
