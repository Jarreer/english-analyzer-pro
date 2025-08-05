import numpy as np
import librosa

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # Downsample if audio is too long to prevent memory issues
        if len(y) > 80000:  # If longer than 5 seconds at 16kHz
            step = len(y) // 80000
            y = y[::step]

        # MFCC with reduced parameters
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512).T, axis=0)

        # Chroma with reduced parameters
        stft = np.abs(librosa.stft(y, hop_length=512))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=512).T, axis=0)

        # Mel spectrogram with reduced parameters
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512).T, axis=0)

        # Spectral contrast with reduced parameters
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, hop_length=512).T, axis=0)

        # Tonnetz with reduced parameters
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, hop_length=512).T, axis=0)

        # Zero-crossing rate with reduced parameters
        zcr = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=512).T, axis=0)

        # RMS Energy with reduced parameters
        rmse = np.mean(librosa.feature.rms(y=y, hop_length=512).T, axis=0)

        # Final feature vector - ensure exactly 180 features
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, zcr, rmse])
        
        # Pad or truncate to exactly 180 features
        if len(features) < 180:
            features = np.pad(features, (0, 180 - len(features)), 'constant')
        elif len(features) > 180:
            features = features[:180]
            
        return features

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return np.zeros(180)