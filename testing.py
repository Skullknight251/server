import os
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import pyaudio
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

# Route chính để ghi âm và nhận diện người nói
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ghi âm giọng nói
    file_path = "data/recorded.wav"
    # record_voice(file_path)
    
    # # Dự đoán giọng nói từ file đã ghi âm
    result = predict_speaker_or_unknown(file_path)
    return jsonify({'result': result})
    # return jsonify({'result': {"test": "asdg"}})

# Hàm để ghi âm giọng nói
def record_voice(file_path, seconds=5):
    FRAM_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAM_PER_BUFFER
    )
    
    print("Đang ghi âm...")
    frames = []
    
    for _ in range(0, int(RATE / FRAM_PER_BUFFER * seconds)):
        data = stream.read(FRAM_PER_BUFFER)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Lưu lại file âm thanh dưới định dạng WAV
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    
    print("Ghi âm hoàn tất")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
