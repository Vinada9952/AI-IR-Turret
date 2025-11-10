import torch
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import pyautogui
import datetime
import numpy as np


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

while True:
    ret, frame, _ = captureImage()
    if ret:
        prediction = recognize( frame, "./models/recognition.pth" )
        cv2.imshow( "frame", frame )
        print( prediction )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break