import json
from os import path

from PIL import Image
from pyrsistent import pvector, PVector

from . import box_utils as bu


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
        boxes = boxes.append(box)

        # Target classes
        class_idx = class2str.index(shape["label"])
        classes = classes.append(class_idx)

    # Correctness check
    assert len(classes) == len(boxes)

    return image, boxes, classes
