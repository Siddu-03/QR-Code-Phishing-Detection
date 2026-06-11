import cv2

image = cv2.imread("input.jpg")

if image is None:
    print("Error: input.jpg not found")
    exit()

x = 100
y = 100
w = 200
h = 200

overlay = image.copy()

cv2.rectangle(
    overlay,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    -1
)

alpha = 0.3

cv2.addWeighted(
    overlay,
    alpha,
    image,
    1 - alpha,
    0,
    image
)

cv2.rectangle(
    image,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    3
)

coordinate_text = f"x={x}, y={y}"

cv2.putText(
    image,
    coordinate_text,
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

cv2.imwrite("output.jpg", image)

print("================================")
print("QR Visualization Completed")
print("================================")
print(f"X Coordinate : {x}")
print(f"Y Coordinate : {y}")
print(f"Width        : {w}")
print(f"Height       : {h}")
print("Output saved as output.jpg")