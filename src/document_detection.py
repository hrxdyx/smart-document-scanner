import cv2
import numpy as np


def find_contours(edges):
  contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    return contours


def find_document_corners(
    contours,
    approximation_accuracy=0.02,
    minimum_area=10000
):
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        approximation = cv2.approxPolyDP(
            contour,
            approximation_accuracy * perimeter,
            True
        )

        if len(approximation) == 4:

            points = approximation.reshape(4, 2)

            return points

    return None


def draw_document_contour(image, corners):
  result = image.copy()

    if corners is None:
        return result

    points = corners.reshape((-1, 1, 2))

    cv2.polylines(
        result,
        [points],
        True,
        (0, 255, 0),
        3
    )

    for point in corners:

        x, y = point

        cv2.circle(
            result,
            (int(x), int(y)),
            8,
            (0, 0, 255),
            -1
        )

    return result
