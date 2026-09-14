import cv2
import numpy as np


def order_points(points):
    points = np.array(points, dtype="float32")

    ordered = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates
    coordinate_sum = points.sum(axis=1)

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]

    # Difference of coordinates
    coordinate_difference = np.diff(points, axis=1)

    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]

    return ordered


def four_point_transform(image, points):
   ordered = order_points(points)

    top_left, top_right, bottom_right, bottom_left = ordered

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)

    max_height = int(max(height_right, height_left))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(
        ordered,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped
