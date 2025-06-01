from flask import Flask, request, jsonify ,render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('violence_detection_model.h5')

# Set the target image size (should match the input shape of your model)
IMG_SIZE = (128, 128)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image from the file
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    # Resize and preprocess the frame
    resized_frame = cv2.resize(frame, IMG_SIZE)
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Make prediction
    prediction = model.predict(input_frame)[0][0]
    label = "Violence" if prediction > 0.5 else "Non-Violence"
    confidence = float(prediction)

    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)