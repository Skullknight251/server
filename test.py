import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Hàm để trích xuất MFCC từ file WAV
def extract_features(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Tải mô hình đã lưu
saved_model = load_model("model.h5")

# Hàm dự đoán người nói và phát hiện người lạ sử dụng mô hình đã lưu
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

# Ví dụ sử dụng
file_path = "E:/wave/data/nam3.wav"  # Đường dẫn tới file cần kiểm tra
predicted_speaker = predict_speaker_or_unknown("E:/Bản ghi Mới 9.wav")
print(f"Người nói được nhận diện là: {predicted_speaker}")