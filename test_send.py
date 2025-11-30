import cv2
import requests

SERVER_URL = "http://127.0.0.1:5000/upload"

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Erreur: caméra introuvable")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur capture caméra")
        break
    cv2.imshow( "Camera", frame )
    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        response = requests.post(
            SERVER_URL,
            files={'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        )
        print( response.json() )
        cv2.waitKey(1)


    except Exception as e:
        print("Erreur:", e)


cap.release()
cv2.destroyAllWindows()
