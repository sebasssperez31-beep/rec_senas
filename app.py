from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import mediapipe as mp
import threading
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import time
from collections import deque   # 🟢 NUEVO

app = Flask(__name__)

# ==============================
# Configuración
# ==============================
stop_event = threading.Event()
training_event = threading.Event()
recognition_event = threading.Event()  # 🔥 Controla el reconocimiento
current_label = None
frames_captured = 0
max_images = 40   # límite de capturas

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

model_path = "model.pkl"
knn = None
last_prediction = None

# ====== Estadísticas de práctica ======
intentos = 0
correctos = 0
racha = 0

# ==============================
# MediaPipe Hands
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==============================
# Cámara y lock
# ==============================
camera = cv2.VideoCapture(0)
camera_lock = threading.Lock()

# 🟢 NUEVO: buffer para suavizar landmarks y predicciones
landmark_history = deque(maxlen=5)
prediction_history = deque(maxlen=5)


def smooth_landmarks(landmarks):
    """Promedia landmarks recientes para que no bailen."""
    landmark_history.append(landmarks)
    avg = np.mean(landmark_history, axis=0)
    return avg.tolist()


def stable_prediction(pred):
    """Devuelve predicción solo si es estable."""
    prediction_history.append(pred)
    if len(prediction_history) == prediction_history.maxlen:
        if all(p == prediction_history[0] for p in prediction_history):
            return pred
    return None


def process_frame():
    global frames_captured, knn, last_prediction
    while not stop_event.is_set():
        with camera_lock:
            ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if recognition_event.is_set():
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

        # 🟢 NUEVO: suavizado de landmarks
        if landmarks:
            landmarks = smooth_landmarks(landmarks)

        # Guardar imágenes
        if training_event.is_set() and current_label and landmarks:
            folder = os.path.join(dataset_path, current_label)
            if not os.path.exists(folder):
                os.makedirs(folder)
            frames_captured += 1
            np.save(os.path.join(folder, f"{frames_captured}.npy"), landmarks)

            if frames_captured >= max_images:
                training_event.clear()
                train_model()

        # Predicción estable
        if recognition_event.is_set() and knn and landmarks:
            try:
                pred = knn.predict([landmarks])[0]
                stable_pred = stable_prediction(pred)  # 🟢 NUEVO
                if stable_pred:
                    last_prediction = stable_pred
            except Exception:
                last_prediction = None

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')


def process_frame_force_draw(target_vocal=None):
    global frames_captured, knn, last_prediction, intentos, correctos, racha
    while not stop_event.is_set():
        with camera_lock:
            ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # 🟢 NUEVO: suavizado de landmarks
        if landmarks:
            landmarks = smooth_landmarks(landmarks)

        if training_event.is_set() and current_label and landmarks:
            folder = os.path.join(dataset_path, current_label)
            if not os.path.exists(folder):
                os.makedirs(folder)
            frames_captured += 1
            np.save(os.path.join(folder, f"{frames_captured}.npy"), landmarks)

            if frames_captured >= max_images:
                training_event.clear()
                train_model()

        if recognition_event.is_set() and knn and landmarks:
            try:
                pred = knn.predict([landmarks])[0]
                stable_pred = stable_prediction(pred)  # 🟢 NUEVO
                if stable_pred:
                    last_prediction = stable_pred
                    if target_vocal:
                        intentos += 1
                        if stable_pred == target_vocal:
                            correctos += 1
                            racha += 1
                        else:
                            racha = 0
            except Exception:
                last_prediction = None

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')

# ==============================
# (Tus rutas y funciones siguen iguales)
# ==============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/entrenamiento')
def entrenamiento():
    return render_template('entrenamiento.html')

@app.route('/train_all')
def train_all():
    return render_template('train_all.html')

@app.route('/video')
def video():
    return Response(process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_entrenar/<vocal>')
def video_entrenar(vocal):
    return Response(process_frame_force_draw(target_vocal=None),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_practica/<vocal>')
def video_practica(vocal):
    return Response(process_frame_force_draw(target_vocal=vocal.upper()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"label": last_prediction})

@app.route('/stop')
def stop():
    stop_event.set()
    return "Stopped"

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    recognition_event.set()
    return "Recognition started"

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    recognition_event.clear()
    return "Recognition stopped"

@app.route('/start_training', methods=['POST'])
def start_training():
    global current_label, frames_captured
    label = request.form.get("label")
    if not label:
        return "Missing label", 400
    current_label = label
    frames_captured = 0
    training_event.set()
    return "Training started"

@app.route('/stop_training', methods=['POST'])
def stop_training():
    training_event.clear()
    return "Training stopped"

@app.route('/practica/<vocal>')
def practica_vocal(vocal):
    vocal = vocal.upper()
    video_url = url_for('video_practica', vocal=vocal)
    return render_template('practica.html', vocal=vocal, video_url=video_url)

@app.route('/reiniciar_practica', methods=['POST'])
def reiniciar_practica():
    global intentos, correctos, racha
    intentos = 0
    correctos = 0
    racha = 0
    return jsonify({"status": "ok"})

@app.route('/status_practica')
def status_practica():
    return jsonify({
        "intentos": intentos,
        "correctos": correctos,
        "racha": racha,
        "precision": (correctos / intentos * 100) if intentos > 0 else 0
    })

@app.route('/entrenar/<vocal>')
def entrenar_vocal(vocal):
    video_url = url_for('video_entrenar', vocal=vocal)
    return render_template('entrenar.html', vocal=vocal, video_url=video_url)

def train_model():
    global knn
    X, y = [], []
    for label in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    landmarks = np.load(os.path.join(folder, file))
                    X.append(landmarks)
                    y.append(label)

    if len(X) > 0 and len(set(y)) > 1:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        joblib.dump(knn, model_path)
        print("✅ Modelo entrenado y guardado")
    else:
        print("⚠️ No se encontraron datos suficientes para entrenar")

if __name__ == "__main__":
    app.run(debug=True)
