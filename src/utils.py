import os
import cv2


def load_image(path):
  if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    image = cv2.imread(path)

    if image is None:
        raise ValueError(
            f"Unable to read the image: {path}"
        )

    return image


def save_image(path, image):
   directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        path,
        image
    )

    if not success:
        raise IOError(
            f"Unable to save image: {path}"
        )


def ensure_directory(path):
   os.makedirs(
        path,
        exist_ok=True
    )
