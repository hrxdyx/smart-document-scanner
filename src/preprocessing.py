import cv2


def resize_image(image, width=1000):
    height, original_width = image.shape[:2]

    if original_width == width:
        return image

    ratio = width / float(original_width)
    new_height = int(height * ratio)

    resized = cv2.resize(image, (width, new_height))

    return resized


def convert_to_grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def preprocess_image(image, width=1000):
    resized = resize_image(image, width)

    gray = convert_to_grayscale(resized)

    blurred = apply_gaussian_blur(gray)

    return resized, gray, blurred
