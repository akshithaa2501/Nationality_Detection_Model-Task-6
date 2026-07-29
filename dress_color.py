import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model only once
model = YOLO("yolov8n.pt")


def color_name(r, g, b):

    if r > 210 and g > 210 and b > 210:
        return "White"

    elif r < 50 and g < 50 and b < 50:
        return "Black"

    elif r > 170 and g < 100 and b < 100:
        return "Red"

    elif g > 170 and r < 100 and b < 100:
        return "Green"

    elif b > 170 and r < 100 and g < 100:
        return "Blue"

    elif r > 170 and g > 170 and b < 100:
        return "Yellow"

    elif r > 170 and 80 < g < 170 and b < 90:
        return "Orange"

    elif r > 120 and b > 120:
        return "Purple"

    elif r > 120 and g > 70 and b < 80:
        return "Brown"

    elif abs(r-g) < 20 and abs(g-b) < 20:
        return "Gray"

    return "Mixed"


def detect_dress_color(img_path):

    img = cv2.imread(img_path)

    if img is None:
        return "Unknown"

    results = model.predict(
        img,
        conf=0.4,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:
        return "Unknown"

    person = None
    max_area = 0

    for box in boxes:

        cls = int(box.cls[0])

        # Person class
        if cls == 0:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2-x1)*(y2-y1)

            if area > max_area:

                max_area = area
                person = (x1, y1, x2, y2)

    if person is None:
        return "Unknown"

    x1, y1, x2, y2 = person

    height = y2-y1
    width = x2-x1

    # Torso region
    shirt = img[
        y1 + int(height*0.28): y1 + int(height*0.65),
        x1 + int(width*0.20): x1 + int(width*0.80)
    ]

    if shirt.size == 0:
        return "Unknown"

    shirt = cv2.cvtColor(shirt, cv2.COLOR_BGR2RGB)

    pixels = shirt.reshape((-1,3))
    pixels = np.float32(pixels)

    k = 4

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())

    dominant = centers[np.argmax(counts)]

    r, g, b = dominant.astype(int)

    return color_name(r, g, b)


if __name__ == "__main__":

    path = input("Enter image path : ")

    print("Dress Color :", detect_dress_color(path))