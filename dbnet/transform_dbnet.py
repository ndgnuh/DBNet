import warnings
from typing import Tuple, Union, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from . import box_utils as bu


def encode_dbnet(
    image: Image.Image,
    polygons: List[List[Tuple[float, float]]],
    classes: List[int],
    target_size: Tuple[int, int],
    num_classes: int,
    shrink_rate: float = 0.4,
    shrink: bool = True,
    max_distance: Union[int, float, None] = None,
    min_box_size: Union[int, float, None] = None,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """Encode the object detection labels to DBNet targets.

    Args:
        image (Image.Image):
            The input image.
        polygons (List[List[Tuple[float, float]]]):
            The target bounding boxes.
            Each box is a polygon represented as a list of (x, y) coordinates.
        classes (List[int]):
            List of object classes corresponding to the bounding boxes.
        target_size (Tuple[int, int]):
            A tuple of (image width, image height).
            This target size will be applied to the heatmaps,
            which includes the image, probability maps and threshold maps.
        num_classes (int):
            The number of classes in the dataset.
        shrink_rate (float, optional):
            The shrink ratio of the polygons when drawing the target heatmaps.
            Default to `0.4`.
        shrink (bool, optional):
            If `True`, the polygons are shrunken when drawing the probablity maps.
            If `False`, the original polygons are used.
            Default to `True`.
        max_distance (Union[int, float], optional):
            The maximum shrinking distance for the bounding boxes.
            None value means no maximum.
            Integer value means absolute maximum value.
            Float value means the maximum will be some percentage of the target area.
            Default to `None`.
        min_box_size (Union[int, float, None], optional):
            The minimum bounding box size.
            None value means no minimum size.
            Integer value means absolute minimum size.
            Float value means the minimum size will be some percentage of the target area.
            Default to `None`.

    Returns:
        image (Image.Image):
            The input image, resized to target size.
        proba_maps (np.ndarray):
            The target probability maps of type float32,
            shape [C, H, W] where C is the number of classes.
        threshold_maps (np.ndarray):
            The target threshold maps of type float32,
            shape [C, H, W] where C is the number of classes.
    """
    # Checking
    msg = "number of object classes is not the same as the number of bounding boxes"
    assert len(classes) == len(polygons), msg

    # Resize image to target size
    W, H = target_size
    S = W * H
    image = image.resize((W, H), Image.Resampling.LANCZOS)

    # Max distance barrier
    if isinstance(max_distance, float):
        max_distance = max_distance * S
    elif max_distance is None:
        max_distance = 99999

    # Min box size barrier
    if isinstance(min_box_size, float):
        min_box_size = min_box_size * S
    elif min_box_size is None:
        min_box_size = -1

    # Target maps
    proba_maps = [np.zeros((H, W), dtype="float32") for _ in range(num_classes)]
    threshold_maps = [np.zeros((H, W), dtype="float32") for _ in range(num_classes)]

    # Negative sample
    if len(classes) == 0:
        proba_maps = np.stack(proba_maps, axis=0)
        threshold_maps = np.stack(threshold_maps, axis=0)
        return image, proba_maps, threshold_maps

    # Draw target maps
    for polygon, class_idx in zip(polygons, classes):
        # Scale polygon to target size
        polygon = [(x * W, y * H) for (x, y) in polygon]

        # Polygon metrics
        area = bu.get_area(polygon)
        length = bu.get_length(polygon)
        shrink_dist = area * (1 + shrink_rate**2) / length
        shrink_dist = abs(min(max_distance, shrink_dist))

        # ignore small bounding boxes
        if abs(area) < min_box_size:
            continue

        # Shrink and expand polygon
        ccw = area > 0
        outer_polygon = bu.offset_polygon(polygon, shrink_dist, ccw)
        if shrink:
            inner_polygon = bu.offset_polygon(polygon, -shrink_dist, ccw)
        else:
            inner_polygon = polygon
        inner_polygon = [np.array(inner_polygon, int)]

        # Draw probability and threshold map
        cv2.fillPoly(proba_maps[class_idx], inner_polygon, 1)
        draw_threshold(threshold_maps[class_idx], polygon, outer_polygon, shrink_dist)

    # Clip the outputs between 0 and 1
    proba_maps = np.clip(proba_maps, 0, 1)
    threshold_maps = np.clip(threshold_maps, 0, 1)
    return image, proba_maps, threshold_maps


def draw_threshold(
    canvas: np.ndarray,
    polygon: List[Tuple[float, float]],
    outer_polygon: List[Tuple[float, float]],
    distance: float,
):
    """Draw a dbnet threshold map on the given canvas using the specified inner and outer polygons.

    The function calculates the distance between each pixel in the canvas
    (that are inside the polygon bounding region) and the line segments defined
    by the inner polygon. The distance is scaled and clipped to 0..1 value.

    Notes:
        This function mutates its inputs (`canvas`).

    Args:
        canvas (np.ndarray):
            The NumPy array representing the canvas on which the threshold will be drawn.
        polygon (List[Tuple[float, float]]):
            The inner (shrinked) polygon.
        outer_polygon (List[Tuple[float, float]]):
            The outer (expaned) polygon (bounding rectangle).
        distance (float):
            The distance used to scale the distance map.

    Returns:
        threshold (np.ndarray):
            The modified canvas with the threshold map (1 - distance map) drawn on it.

    Raises:
        ValueError:
            This is a known bug. Not yet reproducible.
    """
    H, W = canvas.shape

    # Polygon bounding rects
    xmin, ymin = np.min(outer_polygon, axis=0)
    xmax, ymax = np.max(outer_polygon, axis=0)

    # Squeeze to int and limit to the canvas
    xmin = max(0, int(round(xmin)))
    xmax = min(W, int(round(xmax)))
    ymin = max(0, int(round(ymin)))
    ymax = min(H, int(round(ymax)))

    # Draw threshold
    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    distances = []
    n = len(polygon)
    for i in range(n):
        xa, ya = polygon[i]
        xb, yb = polygon[(i + 1) % n]
        dist = point_segment_distance(xs, ys, xa, ya, xb, yb)
        dist = np.clip(dist / distance, 0, 1)
        distances.append(dist)

    # Paste on the canvas
    try:
        thresholds = 1 - np.min(distances, axis=0)
        thresholds = np.maximum(thresholds, canvas[ymin:ymax, xmin:xmax])
        canvas[ymin:ymax, xmin:xmax] = thresholds
    except ValueError as e:
        print("[TODO] Indice errors for what ever reason")
        print("xyxy:", xmin, xmax, ymin, ymax)
        print("distance shape:", thresholds.shape)
        print("region shape:", canvas[ymin:ymax, xmin:xmax].shape)
        raise e
    return canvas


def point_segment_distance(
    x: np.ndarray,
    y: np.ndarray,
    xa: float,
    ya: float,
    xb: float,
    yb: float,
):
    """Calculates the distance between a given set of points defined by (x, y)
    and a line segment defined by two points (xa, ya) and (xb, yb).

    Notes:
        This is the distance to *the segment*, not *the line*.
        Therefore, if the point is too far from the segment,
        the distance will be the minimum distance to two extreme points.

    Args:
        x (np.ndarray):
            NumPy array representing the x-coordinate of the point(s), shape [N].
        y (np.ndarray):
            NumPy array representing the y-coordinate of the point(s), shape [M].
        xa (float):
            x-coordinate of the first point defining the line segment.
        ya (float):
            y-coordinate of the first point defining the line segment.
        xb (float):
            x-coordinate of the second point defining the line segment.
        yb (float):
            y-coordinate of the second point defining the line segment.

    Returns:
        dists (np.ndarray):
            NumPy array containing the distance(s) between the point(s) and the line segment.
            Shape: [M, N] (this shape is convenience for pasting to an image region).
    """
    # Broadcasting shape so that the outputs is [M N]
    # x: N -> [1 N]
    # y: M -> [M 1]
    x = x[None, :]
    y = y[:, None]

    # M is the point where we want to compute distance from
    # AB is the segment
    MA2 = np.square(x - xa) + np.square(y - ya)
    MB2 = np.square(x - xb) + np.square(y - yb)
    AB2 = np.square(xa - xb) + np.square(ya - yb)

    # Cos of AMB = cos
    # c = sqrt(a^2 + b^2 - ab cos(α))
    cos = (MA2 + MB2 - AB2) / (2 * np.sqrt(MA2 * MB2) + 1e-6)

    # T is the extension of MB so that ATB is square
    # S_MAB = AT * MB / 2= AB * MH / 2
    # and AT = sin(alpha) * AM
    # therefore MA * MB * sin(π - AMB) = AB * MH
    # and MH = MA * MB * sin(π - AMB) / AB2
    sin2 = 1 - np.square(cos)
    sin2 = np.nan_to_num(sin2)
    dists = np.sqrt(MA2 * MB2 * sin2 / AB2)

    # Cos > 0 means that this AMB is acute
    # therefore the distance should be the height of the triangle from M
    cos_mask = cos >= 0
    dists[cos_mask] = np.sqrt(np.minimum(MA2, MB2)[cos_mask])
    return dists


def decode_dbnet(
    proba_maps: np.ndarray,
    expand_rate: float,
    expand: bool,
    max_distance: Optional[Union[int, float]],
    min_box_size: Optional[Union[int, float]],
    threshold: float = 0.02,
):
    """Decode the output probability maps of the DBNet model to bounding box polygons.

    Args:
        proba_maps (np.ndarray):
            The probability maps obtained from the DBNet model.
        expand_rate (float):
            The expansion rate used for expanding the bounding boxes.
        expand (bool):
            A boolean value indicating whether to expand the bounding boxes.
        max_distance (Optional[Union[int, float]]):
            The maximum distance for expanding the bounding boxes.
            If provided as a float, it will be scaled by the image size.
            If None, no maximum distance constraint is applied.
        min_box_size (Optional[Union[int, float]]):
            The minimum size of the bounding boxes.
            If provided as a float, it will be scaled by the image size.
            If None, no minimum size constraint is applied.
        threshold (float, optional):
            The threshold value for binarizing the probability maps. Default is 0.02.

    Returns:
        boxes (List[List[Tuple[float, float]]]):
            Object bounding box polygons.
            The polygon are scaled to relative value in 0..1 range.
        classes (List[int]):
            Object class indices.
        scores (List[float]):
            Bounding box scores. Not implemented and only returns 1 for now.
    """
    # Binarize
    bin_maps = (proba_maps > 0.02).astype("uint8")
    C, H, W = bin_maps.shape
    S = H * W

    # Guard for max distance ...
    if isinstance(max_distance, float):
        max_distance = max_distance * S
    elif max_distance is None:
        max_distance = 9999

    # ... and min box size
    if isinstance(min_box_size, float):
        min_box_size = min_box_size * S
    elif min_box_size is None:
        min_box_size = -1

    # Decode targets
    classes = []
    polygons = []
    scores = []

    # Foreach binary maps
    for c_idx in range(C):
        # Find countour
        proba_map = proba_maps[c_idx]
        proba_map = cv2.morphologyEx(proba_map, cv2.MORPH_OPEN, np.ones((5, 5)))
        bin_map = (proba_map > threshold).astype("uint8")
        cnts, _ = cv2.findContours(bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour
        for cnt in cnts:
            # Filter invalid contour
            if len(cnt) < 4:
                continue

            # Filter small boxes
            length = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt, oriented=True)
            if abs(area) < min_box_size or length < min_box_size:
                continue

            # Compute object score
            score = 1
            warnings.warn("[TODO] implement score computation", Warning)

            # Compute expand distance
            if expand:
                dist = abs(area * expand_rate / length)
                dist = min(max_distance, dist)
                try:
                    cnt = cnt[:, 0, :].tolist()
                    cnt = bu.offset_polygon(cnt, dist, area > 0)
                    cnt = np.array(cnt, "float32")[:, None, :]
                except AssertionError:
                    # Invalid polygon
                    continue

            # Simplify the countour
            cnt = cv2.approxPolyDP(cnt, closed=True, epsilon=length * 0.02)
            cnt = cnt[:, 0, :].tolist()

            # Filter invalid contour
            if len(cnt) < 4:
                continue

            # Descale polygon
            cnt = [(x / W, y / H) for (x, y) in cnt]

            # Append results
            polygons.append(cnt)
            classes.append(c_idx)
            scores.append(score)

    # Return
    return polygons, classes, scores
