"""
Modèle de reconnaissance de visages avec LPBH
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO

cap = cv2.VideoCapture( 1 )
if not cap.isOpened():
    print("Impossible d'ouvrir la webcam.")
    exit()

# --- Config ---
DATASET_DIR = Path("train_datas")
MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "lbph_model.yml"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_WIDTH = 200
IMG_HEIGHT = 200
CAPTURE_COUNT = 50  # images per person

# --- Utilitaires ---

def ensure_dirs():
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def get_person_dirs():
    """Retourne la liste des noms de personnes (nom du dossier dans dataset)."""
    if not DATASET_DIR.exists():
        return []
    return [p.name for p in DATASET_DIR.iterdir() if p.is_dir()]

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

# --- Capture d'images pour une personne ---

def capture_person(name: str):
    """Capture des images via webcam et les enregistre dans dataset/<name>/"""
    person_dir = DATASET_DIR / name
    if person_dir.exists():
        print(f"Le dossier {person_dir} existe déjà.")
        resp = input("Supprimer et recommencer (o/N) ? ")
        if resp.lower() == 'o':
            shutil.rmtree(person_dir)
            person_dir.mkdir()
        else:
            print("Annulation.")
            return
    else:
        person_dir.mkdir(parents=True)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    print("Appuyez sur 'q' pour quitter, 's' pour sauvegarder manuellement une image, ou laissez le script capturer automatiquement.")
    saved = 0
    key = cv2.waitKey(1) & 0xFF
    while saved < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture webcam")
            break
        persons = segmentImage( frame, YOLO( "./model/yolov8n.pt" ) )
        for person in persons:
            gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

            for (x,y,w,h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                # Affiche bordure
                cv2.rectangle(person, (x,y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(person, f"Saved: {saved}/{CAPTURE_COUNT}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Capture - appuyez sur s pour sauvegarder manuellement", person)

            key = cv2.waitKey(1) & 0xFF
            # Sauvegarde auto si un visage détecté
            if len(faces) > 0:
                # prends le premier visage détecté
                (x,y,w,h) = faces[0]
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                filename = person_dir / f"{saved:03d}.png"
                cv2.imwrite(str(filename), face_resized)
                saved += 1

            if key == ord('s'):
                # sauvegarde manuelle (même si pas de visage détecté)
                if len(faces) > 0:
                    (x,y,w,h) = faces[0]
                    face = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                    filename = person_dir / f"{saved:03d}.png"
                    cv2.imwrite(str(filename), face_resized)
                    saved += 1
                else:
                    # sauvegarde l'image entière redimensionnée en niveaux de gris
                    face_resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
                    filename = person_dir / f"{saved:03d}.png"
                    cv2.imwrite(str(filename), face_resized)
                    saved += 1

            if key == ord('q'):
                break
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Capture terminée — {saved} images sauvegardées dans {person_dir}.")


# --- Entraînement ---

def prepare_training_data():
    """Lit dataset/ et renvoie (faces, labels, label_map)
    faces: liste d'images (np.array)
    labels: liste d'entiers
    label_map: dict int->nom
    """
    faces = []
    labels = []
    label_map = {}
    inv_label_map = {}
    persons = get_person_dirs()
    if not persons:
        print("Aucun dossier dans dataset/. Capturez d'abord des personnes.")
        return None, None, None

    for idx, person in enumerate(persons):
        label_map[idx] = person
        inv_label_map[person] = idx
        person_dir = DATASET_DIR / person
        for img_path in sorted(person_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            faces.append(img)
            labels.append(idx)

    return faces, labels, label_map


def train_model():
    faces, labels, label_map = prepare_training_data()
    if faces is None:
        return
    if len(faces) == 0:
        print("Aucune image pour l'entraînement.")
        return

    # Crée le recognizer LBPH
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("Impossible de créer LBPH recognizer. Assurez-vous d'avoir installé opencv-contrib-python.")
        print(e)
        return

    print("Entraînement en cours...")
    recognizer.train(faces, np.array(labels))
    recognizer.write(str(MODEL_FILE))

    # Sauvegarde la carte des labels
    label_map_file = MODEL_DIR / "labels.txt"
    with open(label_map_file, 'w', encoding='utf-8') as f:
        for idx in sorted(label_map.keys()):
            f.write(f"{idx}:{label_map[idx]}\n")

    print(f"Entraînement terminé. Modèle sauvegardé dans {MODEL_FILE}.")


# --- Chargement du modèle ---

def load_model():
    if not MODEL_FILE.exists():
        print("Aucun modèle entraîné trouvé. Entraînez d'abord le modèle.")
        return None, None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_FILE))
    except Exception as e:
        print("Erreur chargement modèle LBPH:", e)
        return None, None

    # charge label_map
    label_map = {}
    label_map_file = MODEL_DIR / "labels.txt"
    if label_map_file.exists():
        with open(label_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, name = line.split(':', 1)
                label_map[int(idx)] = name
    return recognizer, label_map


# --- Reconnaissance en temps réel ---

def recognize(frame, recognizer, label_map, confidence_threshold=80, face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)):
    # recognizer, label_map = load_model()
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
    face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
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


def segmentImage(frame, model=YOLO( "./model/yolov8n.pt" ), conf_thresh=0.5):
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

# --- Interface simple ---

def main_menu():
    ensure_dirs()
    while True:
        print("\n=== Menu Reconnaissance Faciale ===")
        print("1) Capturer une nouvelle personne")
        print("2) Entraîner le modèle")
        print("3) Lancer reconnaissance en temps réel")
        print("4) Lister personnes disponibles")
        print("5) Supprimer une personne du dataset")
        print("6) Quitter")
        choice = input("Choix: ")

        if choice == '1':
            name = input("Nom (utilisé comme label): ").strip()
            if name:
                capture_person(name)
        elif choice == '2':
            train_model()
        elif choice == '3':
            try:
                thr = input("Seuil confiance (par défaut 80, valeurs plus petites = plus strict): ")
                thr_val = float(thr) if thr.strip() else 80.0
            except Exception:
                thr_val = 80.0
            recognizer, label_map = load_model()
            windows = []
            while True:
                ret, frame = cap.read()
                ppl = []
                if not ret:
                    print("Erreur lecture webcam")
                    break
                persons = segmentImage( frame, YOLO( "./model/yolov8n.pt" ) )
                for person in persons:
                    output_frame, name, ret = recognize(person, recognizer, label_map, confidence_threshold=thr_val)
                    if ret:
                        if name not in windows:
                            windows.append( name )
                        if name not in ppl:
                            ppl.append( name )
                        cv2.imshow( name, output_frame )
                cv2.imshow( "Image principale", frame )
                tmp = windows
                for w in windows:
                    if w not in ppl:
                        cv2.destroyWindow( w )
                        tmp.remove( w )
                windows = tmp
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif choice == '4':
            persons = get_person_dirs()
            if not persons:
                print("Aucune personne dans dataset.")
            else:
                print("Personnes:")
                for p in persons:
                    print(" -", p)
        elif choice == '5':
            persons = get_person_dirs()
            if not persons:
                print("Aucune personne à supprimer.")
            else:
                print("Personnes:")
                for i,p in enumerate(persons):
                    print(i, p)
                sel = input("Index de la personne à supprimer (ou vide pour annuler): ")
                if sel.strip().isdigit():
                    i = int(sel.strip())
                    if 0 <= i < len(persons):
                        confirm = input(f"Supprimer {persons[i]} de dataset ? (o/N) ")
                        if confirm.lower() == 'o':
                            shutil.rmtree(DATASET_DIR / persons[i])
                            print("Supprimé.")
        elif choice == '6':
            print("Au revoir !")
            break
        else:
            print("Choix invalide.")


if __name__ == '__main__':
    main_menu()




# Programme de reconnaissance faciale simple (OpenCV + LBPH).

# Fonctionnalités :
# - Capturer des images d'une personne (via webcam) et les sauvegarder dans dataset/<nom_personne>/
# - Entraîner un modèle LBPH à partir du dataset
# - Reconnaître en temps réel les visages détectés par la webcam

# Dépendances :
# - Python 3.8+
# - opencv-contrib-python (inclut cv2.face)
# - numpy

# Installation :
#     pip install opencv-contrib-python numpy

# Usage :
#     python reconnaissance_faciale.py

# Le script propose un petit menu :
# 1) Capturer des images d'une nouvelle personne
# 2) Entraîner le modèle
# 3) Lancer la reconnaissance en temps réel
# 4) Quitter

# Notes :
# - Placez le script dans un dossier où il peut créer le dossier `dataset/` et `model/`.
# - Les images sont enregistrées en niveaux de gris, 200x200 px.
# - LBPH est robuste et léger ; pour de meilleures performances ou l'identification sur des grandes bases, regardez la librairie "face_recognition" (dlib) ou des modèles deep-learning.
# - Respectez la vie privée et la législation locale quand vous enregistrez/traitez des visages.