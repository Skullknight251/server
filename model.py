import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Hàm để trích xuất MFCC từ file WAV
def extract_features(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Đường dẫn tới folder chứa các file WAV
DATA_PATH_KNOWN = "E:/wave/data"       # Dữ liệu của người đã biết
DATA_PATH_UNKNOWN = "E:/voice"  # Dữ liệu giả cho người lạ

# Duyệt qua các file WAV và trích xuất đặc trưng
features = []
labels = []

# Thêm dữ liệu của người đã biết (gán nhãn 1)
for file_name in os.listdir(DATA_PATH_KNOWN):
    file_path = os.path.join(DATA_PATH_KNOWN, file_name)
    if file_path.endswith(".wav"):
        mfcc_features = extract_features(file_path)
        features.append(mfcc_features)
        labels.append(1)  # Nhãn "1" cho người đã biết

# Thêm dữ liệu của người lạ (gán nhãn 0)
for file_name in os.listdir(DATA_PATH_UNKNOWN):
    file_path = os.path.join(DATA_PATH_UNKNOWN, file_name)
    if file_path.endswith(".flac"):
        mfcc_features = extract_features(file_path)
        features.append(mfcc_features)
        labels.append(0)  # Nhãn "0" cho người lạ

# Chuyển đổi thành numpy array
X = np.array(features)
y = np.array(labels)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Lớp cuối cùng với sigmoid cho bài toán nhị phân
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình và lưu lại lịch sử quá trình huấn luyện
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")
model.save("model.h5")
