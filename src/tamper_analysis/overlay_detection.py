import cv2


def detect_overlay(image):
    """
    Detect suspicious overlay regions.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    overlays = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 500:

            x, y, w, h = cv2.boundingRect(contour)

            overlays.append({
                "type": "overlay",
                "bbox": (x, y, w, h),
                "area": area
            })

    return overlays


def detect_sticker(image):
    """
    Detect sticker-like regions.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    stickers = []

    for contour in contours:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4 and area > 1000:

            x, y, w, h = cv2.boundingRect(contour)

            stickers.append({
                "type": "sticker",
                "bbox": (x, y, w, h),
                "area": area
            })

    return stickers


if __name__ == "__main__":

    image = cv2.imread("data/test_qr/input.jpg")

    overlays = detect_overlay(image)

    stickers = detect_sticker(image)

    print("\nOverlay Candidates:")
    for overlay in overlays:
        print(overlay)

    print("\nSticker Candidates:")
    for sticker in stickers:
        print(sticker)