from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '' or file is None:
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = Image.open(filepath)
    img = np.array(img)
    height, width, channels = img.shape
    scale_percent = 50  # Adjust to desired scale
    width_new = int(width * scale_percent / 100)
    height_new = int(height * scale_percent / 100)
    img = cv2.resize(img, (width_new, height_new))

    # Process the uploaded image using YOLO
    results = detector.detect(img)

    # Delete the uploaded image after processing (optional)
    # os.remove(filepath)

    # Return results as JSON
    return jsonify(results)

if __name__ == '__main__':
    from ultralytics import YOLOWorld  # Assuming you have a YOLO detection module

    # Initialize YOLO detector
    detector = YOLOWorld('yolov8s-world.pt')

    app.run(debug=True)