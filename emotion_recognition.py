import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Path to your RAVDESS folder
DATA_PATH = r"C:\Users\Hamza\Downloads\CodeAlpha_ML\RAVDESS"  # change path

# Function to extract MFCCs
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# Map emotion labels from file names
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

features, labels = [], []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion = emotion_map[file.split("-")[2]]
            file_path = os.path.join(root, file)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(emotion)

print("Features extracted:", len(features))


from sklearn.preprocessing import LabelEncoder

X = np.array(features)
y = LabelEncoder().fit_transform(labels)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = Sequential([
    LSTM(128, input_shape=(40, 1), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Reshape input for LSTM: (samples, timesteps, features)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


model.save("emotion_recognition_model.h5")
print("✅ Model saved successfully!")

