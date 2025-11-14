import cv2
from ultralytics import YOLO
import mediapipe
import os
import numpy as np
from pathlib import Path
import shutil


# Initialize Mediapipe
mp = mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
print( "MediaPipe setup done" )

# Initialize YOLO
model = YOLO('./model/yolov8n.pt')
print( "YOLO setup done" )

cap = cv2.VideoCapture( 1 )
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print( "OpenCV setup done" )



def captureImage():
    ret, frame = cap.read()
    if ret:
        return frame
    print( "can't grab image" )
    exit()

def load_model():
    if not Path("model/lbph_model.yml").exists():
        print("Aucun modèle entraîné trouvé. Entraînez d'abord le modèle.")
        return None, None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(Path("model/lbph_model.yml")))
    except Exception as e:
        print("Erreur chargement modèle LBPH:", e)
        return None, None

    # charge label_map
    label_map = {}
    label_map_file = Path("model") / "labels.txt"
    if label_map_file.exists():
        with open(label_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, name = line.split(':', 1)
                label_map[int(idx)] = name
    return recognizer, label_map

recognizer, label_map = load_model()
print( "Recognizer setup done" )


def recognize(frame, confidence_threshold=80):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # recognizer, label_map = load_model()
    global recognizer, label_map
    if recognizer is None:
        return

    # face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # print("Lancement de la reconnaissance. Appuyez sur 'q' pour quitter.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    name = ""
    
    img_principale = None
    size_principale = 0

    for i, (x,y,w,h) in enumerate( faces ):
        if w*h > size_principale:
            size_principale = w*h
            try:
                img_principale = faces[i]
            except UnboundLocalError:
                return None, None, False
    try:
        if img_principale == None:
            return None, None, False
    except ValueError:
        pass
    
    face = img_principale
    ( x, y, w, h ) = face
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (200, 200))
    label, confidence = recognizer.predict(face_resized)
    # LBPH renvoie la "distance" -- plus petit vaut mieux. On utilise un seuil pour décider.
    name = label_map.get(label, "Inconnu") if label_map else str(label)
    text = f"{name} ({confidence:.1f})"
    if confidence > confidence_threshold:
        # trop éloigné -> inconnu
        text = f"Inconnu ({confidence:.1f})"
        name = "Inconnu"

    # affiche rectangle et nom
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame, name, True


def drawSkeleton(frame):
    """Process frame and draw skeleton landmarks"""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Process the frame
        results = pose.process(rgb)
        
        # Convert back to BGR
        rgb.flags.writeable = True
        output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        avg_torso = None
        
        if results.pose_landmarks:
            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            
            # Extract torso points
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            # right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate average torso position
            avg_torso = (
                # (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4,
                # (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            )
            
            # Draw torso center point
            h, w = output.shape[:2]
            center_point = (int(avg_torso[0] * w), int(avg_torso[1] * h))
            cv2.circle(output, center_point, 5, (255, 0, 0), -1)
            
        return output, avg_torso

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

print( "setup done" )

windows = [ "image" ]

try:
    while True:
        ret, frame, info = captureImage()
        # print(f"{ret=}, info={info}")
        if not ret:
            print("Failed to grab frame")
            break

        persons = segmentImage( frame, model )
        # overlay debug text
        # disp = frame.copy() if frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        # txt = f"{info.get('shape')} {info.get('dtype')} min={info.get('min')} max={info.get('max')} conv={info.get('converted')}"
        # cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # cv2.imshow( "image", disp )
        cv2.imshow( "image", frame )

        poses = []
        people = [ "image" ]
        printer = "\n\n"
        # cv2.destroyAllWindows()
        for i in range( len( persons ) ):
            output_frame, name, returned = recognize( persons[i] )
            if returned:
                if name not in windows:
                    windows.append( name )
                if name not in people:
                    people.append( name )
                
                persons[i], pos = drawSkeleton( output_frame )
                poses.append( pos )
                printer += f"{name} : {pos=}\n"
                cv2.imshow( f"{name}", persons[i] )
            # name = recognize( persons[i], "./model/recognition.pth" )
            # people.append( name )
            # persons[i], pos = drawSkeleton( persons[i] )
            # cv2.imshow( f"{name}", persons[i] )
            # poses.append( pos )
            # printer += f"{name} : {pos=}\n"
            # windows.append( name )
        
        tmp = windows
        for window in windows:
            if window not in people:
                try:
                    cv2.destroyWindow( window )
                    print( f"detroying window {window}" )
                    tmp.remove( window )
                except Exception:
                    pass
        windows = tmp
        
        
        print( printer )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()