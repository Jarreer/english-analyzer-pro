import joblib
import numpy as np
import librosa
from utils.extract_features import extract_features
from typing import Dict, Tuple

class EmotionAnalyzer:
    def __init__(self, model_path: str = "model/emotion_model.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load emotion model: {e}")
            self.model_loaded = False
    
    def analyze_speaking_confidence(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze speaking confidence using emotion detection with memory optimization"""
        if not self.model_loaded:
            return {
                'confidence_score': 50,
                'detected_emotion': 'neutral',
                'confidence_level': 'medium',
                'feedback': 'Emotion analysis not available',
                'emotion_probabilities': {'neutral': 1.0}
            }
        
        try:
            # Downsample audio to reduce memory usage
            if len(audio_data) > 80000:  # If longer than 5 seconds at 16kHz
                step = len(audio_data) // 80000
                audio_data = audio_data[::step]
            
            # Save temporary audio file for feature extraction
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, sample_rate)
                features = extract_features(f.name)
            
            # Ensure features are exactly 180 dimensions
            if len(features) != 180:
                if len(features) < 180:
                    features = np.pad(features, (0, 180 - len(features)), 'constant')
                else:
                    features = features[:180]
            
            # Predict emotion
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            emotion = self.emotions[prediction]
            confidence_score = self.calculate_confidence_score(emotion, probabilities)
            confidence_level = self.get_confidence_level(confidence_score)
            
            return {
                'confidence_score': confidence_score,
                'detected_emotion': emotion,
                'confidence_level': confidence_level,
                'emotion_probabilities': dict(zip(self.emotions, probabilities)),
                'feedback': self.generate_confidence_feedback(emotion, confidence_score)
            }
            
        except Exception as e:
            return {
                'confidence_score': 50,
                'detected_emotion': 'neutral',
                'confidence_level': 'medium',
                'feedback': f'Analysis error: {str(e)}',
                'emotion_probabilities': {'neutral': 1.0}
            }
    
    def calculate_confidence_score(self, emotion: str, probabilities: np.ndarray) -> float:
        """Calculate speaking confidence based on emotion and probability distribution"""
        # Base confidence scores for different emotions
        emotion_confidence = {
            'neutral': 70,
            'calm': 85,
            'happy': 90,
            'sad': 40,
            'angry': 60,
            'fearful': 30,
            'disgust': 50,
            'surprised': 75
        }
        
        base_score = emotion_confidence.get(emotion, 50)
        
        # Adjust based on prediction confidence
        max_prob = np.max(probabilities)
        confidence_multiplier = 0.5 + (max_prob * 0.5)  # 0.5 to 1.0
        
        return min(base_score * confidence_multiplier, 100)
    
    def get_confidence_level(self, score: float) -> str:
        """Convert confidence score to level"""
        if score >= 80:
            return 'high'
        elif score >= 60:
            return 'medium'
        else:
            return 'low'
    
    def generate_confidence_feedback(self, emotion: str, confidence_score: float) -> str:
        """Generate feedback based on detected emotion and confidence"""
        feedback_templates = {
            'neutral': {
                'high': "Excellent composure! You speak with natural confidence.",
                'medium': "Good composure. Try to add more enthusiasm to your delivery.",
                'low': "Work on projecting more confidence and energy in your speech."
            },
            'calm': {
                'high': "Perfect! You speak with calm confidence and clarity.",
                'medium': "Good calm delivery. Consider adding more vocal variety.",
                'low': "You sound calm but may need more vocal energy."
            },
            'happy': {
                'high': "Excellent! Your enthusiasm and confidence are engaging.",
                'medium': "Good enthusiasm. Balance it with clear articulation.",
                'low': "Your enthusiasm is good, but focus on clear pronunciation."
            },
            'sad': {
                'high': "You sound thoughtful. Try to add more energy for engagement.",
                'medium': "Consider adding more enthusiasm to your delivery.",
                'low': "Work on projecting more confidence and energy."
            },
            'angry': {
                'high': "You speak with intensity. Consider softening for clarity.",
                'medium': "Good energy, but try to maintain calmness for clarity.",
                'low': "Focus on speaking with confidence rather than intensity."
            },
            'fearful': {
                'high': "You sound cautious. Work on projecting more confidence.",
                'medium': "Try to speak with more confidence and less hesitation.",
                'low': "Practice speaking with more confidence and less nervousness."
            },
            'disgust': {
                'high': "You speak with conviction. Balance it with clarity.",
                'medium': "Good conviction, but ensure clear pronunciation.",
                'low': "Focus on clear pronunciation while maintaining conviction."
            },
            'surprised': {
                'high': "You speak with energy and engagement!",
                'medium': "Good energy. Balance it with clear articulation.",
                'low': "Your energy is good, but focus on clear pronunciation."
            }
        }
        
        level = self.get_confidence_level(confidence_score)
        return feedback_templates.get(emotion, {}).get(level, "Keep practicing to improve your speaking confidence.") 