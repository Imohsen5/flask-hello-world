import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for the API

# Load the TensorFlow model
model = tf.keras.models.load_model('model/finetuned.h5')  # Adjust to your model location

# Labels for the hand signs (same as in the React code)
labels = [
    "bb", "taa", "aleff", "ra", "dal", "waw", "zay", "thal", "seen",
    "thaa", "yaa", "khaa", "dha", "jeem", "kaaf", "fa", "haa",
    "dhad", "ta", "ain", "ha", "saad", "sheen", "laam", "meem",
    "nun", "ghain", "gaaf"
]

# Function to process the image and predict the label
def process_image(image_data):
    # Decode the image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image (resize and normalize)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    label = labels[predicted_class]
    return label

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to receive image data and respond with translation
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.data

        # Process image and get prediction
        label = process_image(image_data)
        return jsonify({'translation': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
