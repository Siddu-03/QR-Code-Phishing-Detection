import cv2


def extract_contours(edge_image):
    """
    Extract contours from an edge image.

    Args:
        edge_image (numpy.ndarray)

    Returns:
        list: contours
    """

    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def get_contour_info(contours):
    """
    Return contour statistics.

    Args:
        contours (list)

    Returns:
        list
    """

    contour_data = []

    for contour in contours:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        x, y, w, h = cv2.boundingRect(contour)

        contour_data.append({
            "area": area,
            "perimeter": perimeter,
            "bbox": (x, y, w, h)
        })

    return contour_data


if __name__ == "__main__":

    image = cv2.imread("data/test_qr/input.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    contours = extract_contours(edges)

    info = get_contour_info(contours)

    print(f"Contours Found: {len(info)}")

    for item in info[:10]:
        print(item)