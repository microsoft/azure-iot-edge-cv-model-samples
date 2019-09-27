import json
import os
import io

# Imports for the REST API
from flask import Flask, request # For development
from waitress import serve # For production

# Imports for image processing
from PIL import Image

# Imports from predict_yolov3.py
import predict_yolov3
from predict_yolov3 import YOLOv3Predict

app = Flask(__name__)

@app.route('/')
def index():
    return 'Vision AI module listening'

# Prediction service /image route handles either
#     - octet-stream image file 
#     - a multipart/form-data with files in the imageData parameter
@app.route('/image', methods=['POST'])
def predict_image_handler():
    try:
        imageData = None
        if ('imageData' in request.files):
            imageData = request.files['imageData']
        else:
            imageData = io.BytesIO(request.get_data())
        img = Image.open(imageData)
        results = model.predict(img)
        return json.dumps(results)
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500

if __name__ == '__main__':
    # Load and intialize the model
    global model
    model = YOLOv3Predict('model.onnx', 'labels.txt')
    model.initialize()
    
    # Run the server
    print("Running the Vision AI module...")
    #app.run(host='0.0.0.0', port=8885, debug=True) # For development
    serve(app, host='0.0.0.0', port=8885) # For production
