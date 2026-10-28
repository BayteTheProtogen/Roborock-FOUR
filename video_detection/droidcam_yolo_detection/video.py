import cv2
from ultralytics import YOLO

model = YOLO("besst.pt")
cap = cv2.VideoCapture("/dev/video4")

if not cap.isOpened():
    print("Could not open /dev/video4")
    exit()

triggered_classes = set()
counting = 0

# Clear file when program starts
open("reaction.txt", "w").close()

target_classes = {
    "construction-flat-sidewalk",
    "construction-flat-road",
    "marking-discrete-crosswalk-zebra",
}


def reaction(detected_class, side):
    global counting
    counting += 1

    with open("reaction.txt", "a") as file:
        file.write(
            f"{counting} {detected_class} detected on {side}\n"
        )


def detect_object(results, frame):
    image_width = frame.shape[1]
    detections = []

    # Process every detected object, including target_classes
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detected_class = results[0].names[class_id]

        x1, y1, x2, y2 = box.xyxy[0]
        center_x = (x1 + x2) / 2

        side = "left" if center_x < image_width / 2 else "right"

        detections.append((detected_class, side))

    return detections


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    # Detect all objects and their sides every frame
    detections = detect_object(results, frame)

    # Only react to classes NOT in target_classes
    current_unknown = {
        detected_class
        for detected_class, side in detections
        if detected_class not in target_classes
    }

    # React once per continuous appearance of each unknown class
    for detected_class, side in detections:
        if (
            detected_class not in target_classes
            and detected_class not in triggered_classes
        ):
            reaction(detected_class, side)

    # Remember which unknown classes are currently visible
    triggered_classes = current_unknown

    annotated = results[0].plot()
    cv2.imshow("YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()