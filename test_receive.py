from flask import Flask, request, send_file
import numpy as np
import cv2
import io

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "Aucune image reçue", 400

    file = request.files['image']
    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    center_x = w // 2
    center_y = h // 2

    cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), -1)

    _, img_encoded = cv2.imencode('.jpg', img)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
