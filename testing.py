import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import wave

app = Flask(__name__)

# Hàm để trích xuất MFCC từ file WAV
def extract_features(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Tải mô hình đã lưu
saved_model = load_model("model.h5")

# Hàm dự đoán người nói
def predict_speaker_or_unknown(file_path, threshold=0.7):
    # Trích xuất đặc trưng từ file âm thanh
    mfcc_features = extract_features(file_path)
    
    # Định hình lại đầu vào cho mô hình
    mfcc_features = mfcc_features.reshape(1, -1)
    
    # Dự đoán xác suất đầu ra
    prediction = saved_model.predict(mfcc_features)
    prob = prediction[0][0]

    # In ra xác suất và quyết định
    print(f"Xác suất: {prob}")
    if prob < threshold:
        return "Người lạ"
    else:
        return "Người đã biết"

# Route để nhận file WAV và trả về kết quả dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error("Không tìm thấy file trong yêu cầu.")
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error("File không hợp lệ (filename rỗng).")
        return jsonify({'error': 'File không hợp lệ'}), 400

    # Lưu file và dự đoán
    file_path = os.path.join("data", "received.wav")
    file.save(file_path)

    result = predict_speaker_or_unknown(file_path)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
