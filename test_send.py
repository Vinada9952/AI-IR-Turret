import cv2
import requests

SERVER_URL = "http://127.0.0.1:5000/upload"

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erreur: caméra introuvable")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur capture caméra")
        break

    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        response = requests.post(
            SERVER_URL,
            files={'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        )


    except Exception as e:
        print("Erreur:", e)

    print( response )

cap.release()
cv2.destroyAllWindows()
