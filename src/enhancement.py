import cv2


def enhance_color(image):
   enhanced = cv2.detailEnhance(
        image,
        sigma_s=10,
        sigma_r=0.15
    )

    return enhanced


def enhance_grayscale(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.equalizeHist(gray)

    return enhanced


def enhance_black_white(image):
   gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return binary


def enhance_document(image, mode="color"):
 if mode == "color":
        return enhance_color(image)

    elif mode == "gray":
        return enhance_grayscale(image)

    elif mode == "bw":
        return enhance_black_white(image)

    else:
        raise ValueError(
            "Invalid mode. Choose: color, gray, or bw."
        )
