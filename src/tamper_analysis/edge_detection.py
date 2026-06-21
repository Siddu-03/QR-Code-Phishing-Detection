import cv2
import numpy as np


def detect_edges(image):
    """
    Detect edges in an image using Canny Edge Detection.

    Args:
        image (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Edge map
    """

    if image is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    return edges


if __name__ == "__main__":
    img = cv2.imread("data/test_qr/input.jpg")

    edges = detect_edges(img)

    cv2.imshow("Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()