import cv2
from ultralytics import YOLO
import mediapipe
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Initialize Mediapipe
mp = mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
print( "MediaPipe setup done" )

# Initialize YOLO
model = YOLO('./models/yolov8n.pt')
print( "YOLO setup done" )

# Initialize video capture (essaye DirectShow puis fallback)
def open_capture(port=0):
    cap_local = cv2.VideoCapture(port, cv2.CAP_DSHOW)
    if not cap_local.isOpened():
        cap_local = cv2.VideoCapture(port)
    return cap_local

cap = open_capture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print( "OpenCV setup done" )
# ...existing code...

def captureImage():
    """Capture a frame from the video feed and return debug info"""
    global cap
    ret, frame = cap.read()
    info = {}
    if not ret or frame is None:
        return ret, frame, info

    # Basic diagnostics
    info['shape'] = frame.shape
    info['dtype'] = str(frame.dtype)
    try:
        info['min'] = int(frame.min())
        info['max'] = int(frame.max())
    except Exception:
        info['min'] = None
        info['max'] = None
    info['converted'] = None

    # If single channel -> convert to BGR for display
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        info['converted'] = 'GRAY2BGR'
        return ret, frame, info

    # If 3 channels but all equal (gris déguisé), tenter quelques conversions YUV courantes
    b,g,r = frame[:,:,0].astype(int), frame[:,:,1].astype(int), frame[:,:,2].astype(int)
    channel_diff = (np.abs(b-g).mean() + np.abs(g-r).mean()) / 2
    if channel_diff < 1.0:
        # tentatives de conversion YUV -> BGR (peut échouer selon format)
        tried = []
        for conv in (cv2.COLOR_YUV2BGR_YUY2, cv2.COLOR_YUV2BGR_UYVY, cv2.COLOR_YUV2BGR_NV12, cv2.COLOR_YUV2BGR_I420):
            try:
                newf = cv2.cvtColor(frame, conv)
                # si la conversion donne des canaux différents, on l'utilise
                nb, ng, nr = newf[:,:,0], newf[:,:,1], newf[:,:,2]
                if (np.abs(nb-ng).mean() + np.abs(ng-nr).mean())/2 > 1.0:
                    frame = newf
                    info['converted'] = conv
                    break
                tried.append(conv)
            except Exception:
                tried.append(f"err_{conv}")
        if info['converted'] is None:
            info['converted'] = 'none_tried:' + ",".join(map(str,tried))

    return ret, frame, info

def recognize( frame, file_path ):
    transform = transforms.Compose( [
        transforms.Resize( ( 224, 224 ) ),
        transforms.ToTensor(),
        transforms.Normalize( 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ] )

    # Charger l'image et la transformer
    image = Image.fromarray( cv2.cvtColor( frame, cv2.COLOR_BGR2RGB ) )
    input_tensor = transform( image ).unsqueeze( 0 )  # Ajouter dimension batch

    # Charger le modèle et les classes
    checkpoint = torch.load( file_path, map_location='cuda' if torch.cuda.is_available() else 'cpu' )
    class_names = checkpoint['class_names']

    # Créer et préparer le modèle
    model = models.resnet18( pretrained=False )
    model.fc = torch.nn.Linear( model.fc.in_features, len( class_names ) )
    model.load_state_dict( checkpoint['model_state_dict'] )

    # Mettre le modèle en mode évaluation
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    model = model.to( device )
    model.eval()

    # Prédire avec torch.no_grad
    with torch.no_grad():
        input_tensor = input_tensor.to( device )
        output = model( input_tensor )
        _, predicted_idx = torch.max( output, 1 )
        predicted_label = class_names[predicted_idx.item()]

    return predicted_label

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
        people = []
        printer = "\n\n"
        for i in range( len( persons ) ):
            name = recognize( persons[i], "./models/recognition.pth" )
            persons[i], pos = drawSkeleton( persons[i] )
            cv2.imshow( f"{name}", persons[i] )
            poses.append( pos )
            printer += f"{name} : {pos=}\n"
        print( printer )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()