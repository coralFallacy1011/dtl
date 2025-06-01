from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import logging
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the model
model = None

def load_model_with_fallback():
    """Try to load the model with different methods"""
    global model
    
    # First, check if files exist and log directory contents
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Check for model files
    h5_exists = os.path.exists('violence_detection_model.h5')
    savedmodel_exists = os.path.exists('violence_detection_model')
    
    logger.info(f"violence_detection_model.h5 exists: {h5_exists}")
    logger.info(f"violence_detection_model directory exists: {savedmodel_exists}")
    
    if h5_exists:
        file_size = os.path.getsize('violence_detection_model.h5')
        logger.info(f"Model file size: {file_size} bytes")
    
    # Log TensorFlow version
    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    if not h5_exists and not savedmodel_exists:
        logger.error("No model files found!")
        return False
    
    if h5_exists:
        try:
            # Method 1: Try standard Keras load_model
            from tensorflow.keras.models import load_model
            logger.info("Attempting to load model with standard method...")
            model = load_model('violence_detection_model.h5')
            logger.info("Model loaded successfully with standard method")
            return True
        except Exception as e:
            logger.error(f"Standard model loading failed: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
        
        try:
            # Method 2: Try with compile=False
            from tensorflow.keras.models import load_model
            logger.info("Attempting to load model with compile=False...")
            model = load_model('violence_detection_model.h5', compile=False)
            logger.info("Model loaded successfully with compile=False")
            return True
        except Exception as e:
            logger.error(f"Model loading with compile=False failed: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
        
        try:
            # Method 3: Try with custom objects
            from tensorflow.keras.models import load_model
            logger.info("Attempting to load model with custom_objects={}...")
            model = load_model('violence_detection_model.h5', custom_objects={}, compile=False)
            logger.info("Model loaded successfully with custom_objects")
            return True
        except Exception as e:
            logger.error(f"Model loading with custom_objects failed: {str(e)}")
    
    if savedmodel_exists:
        try:
            # Method 4: Try loading SavedModel format
            import tensorflow as tf
            logger.info("Attempting to load SavedModel format...")
            model = tf.keras.models.load_model('violence_detection_model')
            logger.info("Model loaded from SavedModel format")
            return True
        except Exception as e:
            logger.error(f"SavedModel loading failed: {str(e)}")
    
    logger.error("All model loading methods failed")
    return False

# Set the target image size (should match the input shape of your model)
IMG_SIZE = (128, 128)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({'status': 'healthy', 'model': model_status})

@app.route('/debug')
def debug():
    """Debug endpoint to check system info"""
    import tensorflow as tf
    
    debug_info = {
        'working_directory': os.getcwd(),
        'directory_contents': os.listdir('.'),
        'tensorflow_version': tf.__version__,
        'model_file_exists': os.path.exists('violence_detection_model.h5'),
        'model_loaded': model is not None
    }
    
    if os.path.exists('violence_detection_model.h5'):
        debug_info['model_file_size'] = os.path.getsize('violence_detection_model.h5')
    
    return jsonify(debug_info)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image from the file
        in_memory_file = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Resize and preprocess the frame
        resized_frame = cv2.resize(frame, IMG_SIZE)
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction
        prediction = model.predict(input_frame)[0][0]
        label = "Violence" if prediction > 0.5 else "Non-Violence"
        confidence = float(prediction)

        return jsonify({'label': label, 'confidence': confidence})
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

if __name__ == '__main__':
    # Try to load the model at startup
    if not load_model_with_fallback():
        logger.error("Failed to load model. Server will start but predictions will fail.")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)