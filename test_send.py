import cv2
import requests
import numpy as np

SERVER_URL = "http://127.0.0.1:5000/upload"

cap = cv2.VideoCapture(0)

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

        np_arr = np.frombuffer(response.content, np.uint8)
        server_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow("IMAGE RENVOYÉE PAR SERVEUR", server_img)

    except Exception as e:
        print("Erreur:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
