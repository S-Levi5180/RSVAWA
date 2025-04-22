from flask import Flask, render_template, Response, request, url_for
import os
import pyttsx3
import uuid
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import threading
from model_def import CNNModel
import queue

engine = pyttsx3.init()
speech_queue = queue.Queue()

# App Configuration
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Labels & Model
CSV_PATH = "dataset1/processed_dataset1.csv"
df = pd.read_csv(CSV_PATH)
labels = sorted(df["label"].unique().tolist())
num_classes = len(labels)

# Map label name to encoded index
label_to_index = {label: idx for idx, label in enumerate(labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

model = CNNModel(num_classes)
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Webcam Setup
camera = cv2.VideoCapture(0)
spoken_labels = set()

# Text-to-Speech (Threaded)
def speech_loop():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

# Start the loop in a thread once
speech_thread = threading.Thread(target=speech_loop, daemon=True)
speech_thread.start()

def speak_text(text):
    speech_queue.put(text)

# Webcam Stream Generator
def generate_frames():
    """Yield frames from the webcam with prediction overlays and voice alerts."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = index_to_label[predicted.item()]

        # Draw label on frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Speak only once per label
        if label not in spoken_labels:
            speak_text(f"Caution: {label} ahead")
            spoken_labels.add(label)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle uploaded image and show prediction result."""
    file = request.files.get('file')
    if not file:
        return "❌ No file uploaded", 400

    try:
        # Save file
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert('RGB')
        input_tensor = transform(img).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = index_to_label[predicted.item()]

        # Speak prediction in background
        speak_text(f"Uploaded sign is: {label}")

        # Render result
        return render_template('result.html', label=label,
                               image_path=url_for('static', filename=f'uploads/{filename}'))

    except Exception as e:
        return f"❌ Error during prediction: {e}", 500

# Run the App
if __name__ == "__main__":
    app.run(debug=True)