from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model


app = Flask(__name__)

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def process_image(file):
    
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    class_name, confidence_score = process_image(file)

    return jsonify({"class_name": class_name, "confidence_score": confidence_score})

if __name__ == '__main__':
    app.run(debug=True)