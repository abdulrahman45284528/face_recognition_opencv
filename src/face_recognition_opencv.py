# face_recognition_opencv.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import pickle


# Load Pre-trained FaceNet Model

facenet_model = load_model('facenet_keras.h5')


# Initialize KNN Classifier

classifier = KNeighborsClassifier(n_neighbors=3)


# Load Saved Embeddings (if available)

EMBEDDINGS_FILE = 'face_embeddings.pkl'

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_names = data['names']
    classifier.fit(known_face_encodings, known_face_names)
else:
    known_face_encodings = []
    known_face_names = []


# Preprocessing for FaceNet

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    face = (face - 127.5) / 127.5
    return np.expand_dims(face, axis=0)


# Get Face Embedding

def get_embedding(face):
    face = preprocess_face(face)
    embedding = facenet_model.predict(face)[0]
    return embedding


# Add New Face to Database

def add_new_face(name, face):
    embedding = get_embedding(face)
    known_face_encodings.append(embedding)
    known_face_names.append(name)
    classifier.fit(known_face_encodings, known_face_names)
    print(f"[INFO] New face added: {name}")

    # Save updated embeddings
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)
    print("[INFO] Embeddings saved successfully.")


# Recognize Face

def recognize_face(face):
    embedding = get_embedding(face)
    if len(known_face_encodings) > 0:
        label = classifier.predict([embedding])[0]
        confidence = classifier.predict_proba([embedding]).max()
        if confidence > 0.5:
            return label, confidence
    return None, 0.0


# Draw Result on Frame

def draw_label(frame, label, confidence, x, y, x2, y2):
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# Main Function for Webcam Detection

def main():
    print("[INFO] Starting Webcam...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using OpenCV's Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label, confidence = recognize_face(face)

            if label:
                draw_label(frame, label, confidence, x, y, x+w, y+h)
            else:
                draw_label(frame, "Unknown", 0, x, y, x+w, y+h)

        # Show the frame
        cv2.imshow("Face Recognition", frame)

        # Keybindings:
        # Press 'a' to add a new face
        # Press 'q' to quit
        key = cv2.waitKey(1)
        if key == ord('a'):
            name = input("[INFO] Enter the name for this face: ")
            if name:
                add_new_face(name, face)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
