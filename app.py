from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model and label encoder
model = joblib.load("voice_detector_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filepath = os.path.join("temp.wav")
    file.save(filepath)

    features = extract_features(filepath).reshape(1, -1)
    prediction = model.predict(features)
    confidence = np.max(model.predict_proba(features))

    result = {
        "prediction": label_encoder.inverse_transform(prediction)[0],
        "confidence": round(confidence * 100, 2)
    }
    return jsonify(result)  # ✅ Correct way to respond to a frontend API call

# Optional: Route to confirm server is running
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
