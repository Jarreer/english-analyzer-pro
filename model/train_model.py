import os
import glob
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils.extract_features import extract_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/ravdess")
DATA_PATH = os.path.abspath(DATA_PATH)

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

observed_emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def load_data():
    X, y = [], []

    for file in glob.glob(f"{DATA_PATH}/*/*.wav"):
        print("Checking:", file)
        file_name = os.path.basename(file)
        parts = file_name.split("-")

        if len(parts) < 3:
            print("❌ Unexpected filename:", file_name)
            continue

        emotion_code = parts[2]
        emotion_label = emotions.get(emotion_code)

        if emotion_label is None:
            print("❌ Unknown emotion code:", emotion_code)
            continue

        print("✅ Using file:", file_name, "| Emotion:", emotion_label)

        if emotion_label not in observed_emotions:
            continue

        try:
            features = extract_features(file)
            X.append(features)
            y.append(observed_emotions.index(emotion_label))
        except Exception as e:
            print(f"⚠️ Skipping file {file_name} due to error: {e}")
            continue

    print(f"📊 Final loaded samples: {len(X)}")
    return np.array(X), np.array(y)
 
X, y = load_data()
print(f"Loaded samples: {len(X)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model/emotion_model.pkl")