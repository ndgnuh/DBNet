from typing import List, Tuple


def get_area(poly: List[Tuple[float, float]]) -> float:
    """Calculate area of a polygon.

    Args:
        poly (List[Tuple[float, float]]):
            List of tuple representing x, y points.
            Numpy arrays of shape [P, 2] would do too.
    Returns:
        area (float):
            The polygon area
    References:
        https://en.wikipedia.org/wiki/Polygon#Area
    """
    n = len(poly)
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area /= 2
    return area


def get_length(poly: List[Tuple[float, float]]) -> float:
    """Calculate the perimeter of a polygon.

    Args:
        poly (List[Tuple[float, float]]):
            List of tuple representing x, y points.
            Numpy arrays of shape [P, 2] would do too.

    Returns:
        length (float):
            The perimeter of the polygon

    References:
        https://en.wikipedia.org/wiki/Polygon#Area
    """
    n = len(poly)
    peri = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        peri += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return peri


def offset_polygon(poly: List[Tuple[float, float]], offset: float):
    """Offset polygon

    Args:
        poly (List[Tuple[float, float]]):
            The polygon to expand
        offset (float):
            The offset, positive offset means expanding

    Returns:
        new_poly (List[Tuple[float, float]]):
            The offset polygon
    """
    # Make offset positive when expand
    scale = 1000
    m = n = len(poly)
    offset = offset * scale
    offset_lines = [(0.0, 0.0, 0.0, 0.0)] * n
    new_poly = [(0.0, 0.0)] * n

    for i in range(n):
        # Line endpoints
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Skip collided points
        if x1 == x2 and y1 == y2:
            m = m - 1
            continue

        # Rescale for accuracy
        x1 = x1 * scale
        x2 = x2 * scale
        y1 = y1 * scale
        y2 = y2 * scale

        # Calculate the direction vector & normal vector
        vx, vy = x2 - x1, y2 - y1
        vx, vy = vy, -vx

        # normalize the normal vector
        length = (vx**2 + vy**2) ** 0.5
        vx, vy = vx / length, vy / length

        # Offset endpoints -> offset lines
        x1 = x1 + vx * offset
        y1 = y1 + vy * offset
        x2 = x2 + vx * offset
        y2 = y2 + vy * offset
        offset_lines[i] = (x1, y1, x2, y2)

    # Find intersections
    # New poly vertices are the intersection of the offset lines
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    for i in range(m):
        (x1, y1, x2, y2) = offset_lines[i]
        (x3, y3, x4, y4) = offset_lines[(i + 1) % m]
        deno = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if deno == 0:
            continue
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - x4 * y3)) / deno
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - x4 * y3)) / deno
        new_poly[i] = (x / scale, y / scale)

    assert len(new_poly) > 2
    return new_poly
