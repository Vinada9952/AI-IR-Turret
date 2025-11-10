import cv2
from ultralytics import YOLO

# Open camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load YOLO model once
model = YOLO('./models/yolov8n.pt')

print( "setup done" )

def captureImage():
    """Capture a frame from the video feed"""
    global cap
    ret, frame = cap.read()
    if not ret:
        return False, None
    return True, frame

def segmentImage(frame, model, conf_thresh=0.5):
    """Return list of cropped person images from frame using provided model"""
    results = model(frame, classes=0)  # class 0 = person (COCO)
    person_images = []

    if len(results) == 0:
        return person_images

    # results[0].boxes.data columns: x1, y1, x2, y2, conf, class
    for det in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = map(float, det)
        if conf < conf_thresh:
            continue

        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2].copy()
        person_images.append(crop)

    return person_images

try:
    while True:
        ret, frame = captureImage()
        if not ret:
            print("Failed to grab frame")
            break

        persons = segmentImage(frame, model)

        # # Draw boxes on original frame and show crops
        # for i, (crop, bbox) in enumerate(persons):
        #     x1, y1, x2, y2, conf = bbox
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 6),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        #     cv2.imshow(f"Person {i}", crop)

        for i in range( len( persons ) ):
            cv2.imshow( f"Person {i}", persons[i] )

        # cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()