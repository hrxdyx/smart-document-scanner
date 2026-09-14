import cv2


def detect_edges(blurred_image, low_threshold=50, high_threshold=150):
  edges = cv2.Canny(
        blurred_image,
        low_threshold,
        high_threshold
    )

    return edges
