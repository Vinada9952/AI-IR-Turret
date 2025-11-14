import cv2
import time
from ultralytics import YOLO
import os
from pathlib import Path
import shutil

# Load YOLO model
model = YOLO('./model/yolov8n.pt')

def extract_faces( frame ):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    img = frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    if len(faces) == 0:
        return -1
    
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (200, 200))

    return face_resized

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

def ensure_directories():
    """Ensure all required directories exist"""
    dirs = ['./raw_data', './treated_data', './train_datas']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def segment_and_save_images():
    """Process raw images and save detected persons"""
    for img_file in os.listdir("./raw_data/"):
        frame = cv2.imread("./raw_data/" + img_file)
        if frame is None:
            print(f"Could not read {img_file}")
            continue

        persons = segmentImage(frame, model)
        for person in persons:
            timestamp = str(time.time()).replace('.', '_')
            face = extract_faces( person )
            try:
                if face != -1:
                    cv2.imshow( "full person", person )
                    cv2.imshow( "face", face )
                    cv2.waitKey( 1 )
                    cv2.imwrite(f"./treated_data/{timestamp}.jpg", face)
            except ValueError:
                cv2.imshow( "full person", person )
                cv2.imshow( "face", face )
                cv2.waitKey( 1 )
                cv2.imwrite(f"./treated_data/{timestamp}.jpg", face)
        # cv2.destroyAllWindows()

def classify_images():
    """Classify detected persons into categories"""
    # Get categories
    choices = os.listdir("./train_datas/")
    if not choices:
        print("No categories found in train_datas/")
        return

    # Show available categories
    print("\nAvailable categories:")
    for i, choice in enumerate(choices):
        print(f"{choice}: Press '{chr(i+65)}' ({chr(i+65).lower()})")
    print("Press 'q' to quit\n")

    # Process each image
    for img_file in os.listdir("./treated_data/"):
        img_path = f"./treated_data/{img_file}"
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Show image and wait for valid key
        while True:
            cv2.imshow("Classify Person", frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
                
            # Check if key corresponds to a valid category
            for i, _ in enumerate(choices):
                if key == ord(chr(i+65).lower()):
                    # Save to appropriate category
                    dest_path = f"./train_datas/{choices[i]}/{img_file}"
                    cv2.imwrite(dest_path, frame)
                    os.remove(img_path)  # Remove from treated_data
                    print(f"Saved to {choices[i]}")
                    cv2.destroyWindow("Classify Person")
                    return True
            
            print("Invalid key - press", end=" ")
            print(*[f"'{chr(i+65).lower()}' for {c}" for i, c in enumerate(choices)], 
                  "or 'q' to quit", sep=", ")

def main():
    ensure_directories()
    
    # Step 1: Process raw images
    print("Processing raw images...")
    segment_and_save_images()
    
    # Step 2: Classify detected persons
    print("\nStarting classification...")
    while True:
        if not os.listdir("./treated_data/"):
            print("No more images to classify!")
            break
            
        if not classify_images():
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()