import numpy as np
import librosa
import re
from phonemizer import phonemize
import pyphen
from typing import Dict, List, Tuple

class PronunciationAnalyzer:
    def __init__(self):
        self.dic = pyphen.Pyphen(lang='en_US')
        self.stress_patterns = {
            'noun': ['1', '0'],  # Primary stress on first syllable
            'verb': ['0', '1'],   # Primary stress on second syllable
            'adjective': ['1', '0', '0']  # Primary stress on first syllable
        }
    
    def extract_phonetic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Extract phonetic features from audio with memory optimization"""
        features = {}
        
        try:
            # Downsample audio to reduce memory usage
            if len(audio_data) > 80000:  # If longer than 5 seconds at 16kHz
                step = len(audio_data) // 80000
                audio_data = audio_data[::step]
            
            # Pitch analysis with memory optimization
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, hop_length=512)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Energy analysis
            rms = librosa.feature.rms(y=audio_data, hop_length=512)
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # Spectral features with reduced MFCC
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=8, hop_length=512)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            # Formant analysis (simplified)
            stft = np.abs(librosa.stft(audio_data, hop_length=512))
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=stft))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=stft))
            
        except Exception as e:
            # Fallback values if analysis fails
            features = {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_range': 0,
                'energy_mean': 0,
                'energy_std': 0,
                'mfcc_mean': [0] * 8,
                'mfcc_std': [0] * 8,
                'spectral_centroid': 0,
                'spectral_bandwidth': 0
            }
        
        return features
    
    def analyze_pronunciation(self, text: str, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze pronunciation with optimized speed"""
        try:
            # Extract basic audio features quickly
            audio_features = self.extract_phonetic_features(audio_data, sample_rate)
            
            # Simplified stress pattern analysis
            stress_analysis = self.analyze_stress_patterns(text)
            
            # Calculate pronunciation score
            pronunciation_score = self.calculate_pronunciation_score(audio_features, stress_analysis)
            
            # Detect accent indicators (simplified)
            accent_indicators = self.detect_accent_indicators(audio_features)
            
            return {
                'pronunciation_score': pronunciation_score,
                'audio_features': audio_features,
                'stress_analysis': stress_analysis,
                'accent_indicators': accent_indicators
            }
            
        except Exception as e:
            # Return default values if analysis fails
            return {
                'pronunciation_score': 50,
                'audio_features': {
                    'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
                    'energy_mean': 0, 'energy_std': 0,
                    'mfcc_mean': [0] * 8, 'mfcc_std': [0] * 8,
                    'spectral_centroid': 0, 'spectral_bandwidth': 0
                },
                'stress_analysis': {'stress_score': 0, 'stress_pattern': []},
                'accent_indicators': []
            }
    
    def analyze_stress_patterns(self, text: str) -> Dict:
        """Analyze stress patterns with optimized speed"""
        try:
            words = text.lower().split()
            stress_patterns = []
            
            for word in words[:10]:  # Limit to first 10 words for speed
                syllables = self.dic.inserted(word).split('-')
                if len(syllables) > 1:
                    # Simplified stress pattern
                    stress_patterns.append('1' + '0' * (len(syllables) - 1))
            
            stress_score = len(stress_patterns) / max(len(words), 1) * 100
            
            return {
                'stress_score': stress_score,
                'stress_pattern': stress_patterns[:5]  # Limit to 5 patterns
            }
            
        except Exception as e:
            return {
                'stress_score': 0,
                'stress_pattern': []
            }
    
    def calculate_pronunciation_score(self, audio_features: Dict, stress_analysis: Dict) -> float:
        """Calculate pronunciation score with optimized speed"""
        try:
            # Simplified scoring based on key features
            pitch_score = min(100, max(0, audio_features['pitch_std'] * 2))
            energy_score = min(100, max(0, (1 - audio_features['energy_std']) * 100))
            stress_score = stress_analysis.get('stress_score', 50)
            
            # Weighted average
            pronunciation_score = (pitch_score * 0.3 + energy_score * 0.3 + stress_score * 0.4)
            
            return min(100, max(0, pronunciation_score))
            
        except Exception as e:
            return 50.0  # Default score
    
    def detect_accent_indicators(self, audio_features: Dict) -> List[str]:
        """Detect accent indicators with optimized speed"""
        indicators = []
        
        try:
            # Simplified accent detection
            if audio_features['pitch_std'] < 50:
                indicators.append("Try varying your pitch more for natural speech")
            
            if audio_features['energy_std'] > 0.5:
                indicators.append("Work on consistent volume throughout speech")
            
            if audio_features['spectral_centroid'] < 1000:
                indicators.append("Practice clearer articulation")
                
        except Exception as e:
            indicators.append("Focus on clear pronunciation")
        
        return indicators[:3]  # Limit to 3 indicators 