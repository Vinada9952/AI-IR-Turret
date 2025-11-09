import cv2
import numpy as np
from ultralytics import YOLO

def capture_and_segment_persons():
    # Initialiser la caméra
    cap = cv2.VideoCapture(1)
    
    # Charger le modèle YOLO
    model = YOLO('yolov8n.pt')
    
    # Capturer une image
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture")
        return []
    
    # Détecter les personnes dans l'image
    results = model(frame, classes=0)  # class 0 est 'person' dans COCO
    
    # Liste pour stocker les images découpées
    person_images = []
    
    # Pour chaque détection
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = detection
        if conf > 0.5:  # Seuil de confiance
            # Convertir en entiers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Découper l'image
            person_img = frame[y1:y2, x1:x2]
            person_images.append(person_img)
    
    # Libérer la caméra
    cap.release()
    
    return person_images

# Utilisation
person_images = capture_and_segment_persons()

# Afficher ou sauvegarder les images
for i, img in enumerate(person_images):
    cv2.imshow(f'Person {i}', img)
    cv2.imwrite(f'person_{i}.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()