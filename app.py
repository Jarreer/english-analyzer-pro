import streamlit as st 
import soundfile as sf

# Handle sounddevice import for cloud deployment
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except OSError:
    # PortAudio not available on Streamlit Cloud
    SOUNDDEVICE_AVAILABLE = False
    st.warning("🎙️ **Audio recording is not available on this platform.** Please use the file upload option below.")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    st.warning("🎙️ **Audio recording library not available.** Please use the file upload option below.")
import numpy as np
import speech_recognition as sr
import tempfile
import requests
import random
import matplotlib.pyplot as plt
import io
import time
import re
import sqlite3
from fpdf import FPDF
from gtts import gTTS
import os
import bcrypt
from datetime import datetime
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
# import seaborn as sns  # Commented out as it's not essential for core functionality
from utils.pronunciation_analyzer import PronunciationAnalyzer
from utils.emotion_integration import EmotionAnalyzer
from utils.content_manager import ContentManager

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

# Advanced Language Analysis Classes
class AdvancedLanguageAnalyzer:
    def __init__(self):
        # Idioms and phrasal verbs database
        self.idioms = {
            "break down", "come up with", "get along", "look up to", "put up with",
            "run out of", "take care of", "turn down", "work out", "give up",
            "find out", "make up", "set up", "bring up", "come across",
            "get over", "look after", "put off", "take off", "turn up"
        }
        
        # Academic vocabulary
        self.academic_words = {
            "furthermore", "moreover", "nevertheless", "consequently", "subsequently",
            "therefore", "thus", "hence", "accordingly", "conversely",
            "similarly", "likewise", "in contrast", "on the other hand", "however",
            "nevertheless", "nonetheless", "despite", "although", "whereas",
            "consequently", "as a result", "due to", "owing to", "in light of",
            "with regard to", "in terms of", "in relation to", "with respect to",
            "in accordance with", "in compliance with", "subsequently", "previously",
            "initially", "ultimately", "primarily", "secondarily", "fundamentally",
            "essentially", "basically", "specifically", "particularly", "especially"
        }
        
        # Discourse markers
        self.discourse_markers = {
            "addition": ["furthermore", "moreover", "in addition", "besides", "also"],
            "contrast": ["however", "nevertheless", "on the other hand", "in contrast", "whereas"],
            "cause_effect": ["therefore", "thus", "consequently", "as a result", "hence"],
            "sequence": ["firstly", "secondly", "finally", "next", "then"],
            "example": ["for example", "for instance", "such as", "namely", "specifically"],
            "conclusion": ["in conclusion", "to sum up", "overall", "in summary", "finally"]
        }
        
        # Formal vs informal indicators
        self.formal_indicators = {
            "formal": ["furthermore", "moreover", "consequently", "therefore", "thus", "hence"],
            "informal": ["gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "okay", "cool"]
        }
    
    def analyze_idiomatic_expressions(self, text):
        """Analyze use of idioms and phrasal verbs"""
        words = text.lower().split()
        found_idioms = []
        idiom_score = 0
        
        # Check for idioms and phrasal verbs
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.idioms:
                found_idioms.append(bigram)
                idiom_score += 10
        
        # Check for longer idioms
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if trigram in self.idioms:
                found_idioms.append(trigram)
                idiom_score += 15
        
        return {
            'idiom_score': min(idiom_score, 100),
            'found_idioms': found_idioms,
            'idiom_count': len(found_idioms)
        }
    
    def analyze_academic_vocabulary(self, text):
        """Analyze academic vs everyday vocabulary"""
        words = text.lower().split()
        academic_count = 0
        academic_words_found = []
        
        for word in words:
            if word in self.academic_words:
                academic_count += 1
                academic_words_found.append(word)
        
        academic_ratio = (academic_count / len(words)) * 100 if words else 0
        academic_score = min(academic_ratio * 2, 100)  # Scale to 100
        
        return {
            'academic_score': academic_score,
            'academic_words': academic_words_found,
            'academic_count': academic_count,
            'academic_ratio': academic_ratio
        }
    
    def analyze_complex_sentences(self, text):
        """Analyze sentence complexity"""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        complex_sentences = 0
        avg_sentence_length = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) > 15:  # Complex sentence threshold
                complex_sentences += 1
            avg_sentence_length += len(words)
        
        avg_sentence_length = avg_sentence_length / len(sentences) if sentences else 0
        complexity_ratio = (complex_sentences / len(sentences)) * 100 if sentences else 0
        complexity_score = min(complexity_ratio * 2, 100)
        
        return {
            'complexity_score': complexity_score,
            'complex_sentences': complex_sentences,
            'avg_sentence_length': avg_sentence_length,
            'complexity_ratio': complexity_ratio
        }
    
    def analyze_discourse_markers(self, text):
        """Analyze use of discourse markers"""
        text_lower = text.lower()
        marker_count = 0
        markers_found = {}
        
        for category, markers in self.discourse_markers.items():
            category_count = 0
            for marker in markers:
                if marker in text_lower:
                    category_count += 1
                    marker_count += 1
            if category_count > 0:
                markers_found[category] = category_count
        
        discourse_score = min(marker_count * 10, 100)
        
        return {
            'discourse_score': discourse_score,
            'marker_count': marker_count,
            'markers_found': markers_found
        }
    
    def analyze_register(self, text):
        """Analyze formal vs informal register"""
        text_lower = text.lower()
        formal_count = 0
        informal_count = 0
        
        for formal_word in self.formal_indicators["formal"]:
            if formal_word in text_lower:
                formal_count += 1
        
        for informal_word in self.formal_indicators["informal"]:
            if informal_word in text_lower:
                informal_count += 1
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            register_score = 50  # Neutral
        else:
            register_score = (formal_count / total_indicators) * 100
        
        register_type = "formal" if register_score > 70 else "informal" if register_score < 30 else "neutral"
        
        return {
            'register_score': register_score,
            'register_type': register_type,
            'formal_count': formal_count,
            'informal_count': informal_count
        }

class IELTSBandCalculator:
    def __init__(self):
        self.band_descriptors = {
            "9": "Expert user - Has full operational command of the language",
            "8": "Very good user - Has fully operational command with occasional inaccuracies",
            "7": "Good user - Has operational command with occasional inaccuracies",
            "6": "Competent user - Has generally effective command despite inaccuracies",
            "5": "Modest user - Has partial command, coping with overall meaning",
            "4": "Limited user - Basic competence is limited to familiar situations",
            "3": "Extremely limited user - Conveys and understands only general meaning",
            "2": "Intermittent user - No real communication possible",
            "1": "Non-user - Essentially has no ability to use the language"
        }
    
    def calculate_ielts_band(self, grammar_score, vocab_score, pronunciation_score, 
                           fluency_score, coherence_score, task_achievement_score):
        """Calculate IELTS band score based on multiple criteria"""
        
        # Weighted average of all scores
        weighted_score = (
            grammar_score * 0.25 +
            vocab_score * 0.20 +
            pronunciation_score * 0.20 +
            fluency_score * 0.15 +
            coherence_score * 0.10 +
            task_achievement_score * 0.10
        )
        
        # Convert to IELTS band
        if weighted_score >= 90:
            band = 9
        elif weighted_score >= 80:
            band = 8
        elif weighted_score >= 70:
            band = 7
        elif weighted_score >= 60:
            band = 6
        elif weighted_score >= 50:
            band = 5
        elif weighted_score >= 40:
            band = 4
        elif weighted_score >= 30:
            band = 3
        elif weighted_score >= 20:
            band = 2
        else:
            band = 1
        
        return {
            'band': band,
            'band_descriptor': self.band_descriptors[str(band)],
            'weighted_score': weighted_score,
            'detailed_scores': {
                'grammar': grammar_score,
                'vocabulary': vocab_score,
                'pronunciation': pronunciation_score,
                'fluency': fluency_score,
                'coherence': coherence_score,
                'task_achievement': task_achievement_score
            }
        }
    
    def get_band_feedback(self, band):
        """Get specific feedback for IELTS band"""
        feedback = {
            "9": "Exceptional performance! You demonstrate native-like proficiency.",
            "8": "Excellent performance with minor areas for refinement.",
            "7": "Strong performance with some room for improvement in specific areas.",
            "6": "Good foundation with clear areas for targeted improvement.",
            "5": "Moderate proficiency - focus on core language skills.",
            "4": "Basic proficiency - need systematic improvement in multiple areas.",
            "3": "Limited proficiency - requires intensive language development.",
            "2": "Very limited proficiency - needs fundamental language training.",
            "1": "Minimal proficiency - requires basic language instruction."
        }
        return feedback.get(str(band), "Continue practicing to improve your skills.")

# Initialize advanced analyzers
advanced_analyzer = AdvancedLanguageAnalyzer()
ielts_calculator = IELTSBandCalculator()

# Add caching for better performance
@st.cache_data
def get_advanced_analyzer():
    return AdvancedLanguageAnalyzer()

@st.cache_data
def get_ielts_calculator():
    return IELTSBandCalculator()

# --- AUTH SETUP ---
def create_user_tables():
    with sqlite3.connect("user_results.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                name TEXT,
                password_hash TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_results (
                username TEXT,
                date TEXT,
                grammar REAL,
                vocab REAL,
                speed REAL,
                cefr TEXT
            )
        """)

create_user_tables()

def register_user(username, name, password):
    with sqlite3.connect("user_results.db") as conn:
        if conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone():
            return False
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        conn.execute("INSERT INTO users VALUES (?, ?, ?)", (username, name, hashed))
        return True

def verify_user(username, password):
    with sqlite3.connect("user_results.db") as conn:
        user = conn.execute("SELECT name, password_hash FROM users WHERE username = ?", (username,)).fetchone()
        if user and bcrypt.checkpw(password.encode(), user[1].encode()):
            return user[0]
    return None

# --- LOGIN & SIGNUP UI ---
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "username": None, "name": None}

def login_ui():
    st.subheader("🔐 Login")
    user = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login", key="login_button"):
        name = verify_user(user, pwd)
        if name:
            st.session_state.auth = {"logged_in": True, "username": user, "name": name}
            st.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

def signup_ui():
    st.subheader("🆕 Sign Up")
    user = st.text_input("New Username", key="signup_user")
    name = st.text_input("Your Name", key="signup_name")
    pwd = st.text_input("New Password", type="password", key="signup_pwd")
    if st.button("Create Account", key="signup_button"):
        if register_user(user, name, pwd):
            st.success("✅ Account created. Please login.")
        else:
            st.warning("⚠️ Username already exists.")

# --- Login Route ---
if not st.session_state.auth["logged_in"]:
    st.title("🎤 English Analyzer Pro")
    col1, col2 = st.columns(2)
    with col1:
        login_ui()
    with col2:
        signup_ui()
    st.stop()

# --- LOGGED IN ---
username = st.session_state.auth["username"]
name = st.session_state.auth["name"]

# --- DB Setup ---
def create_connection():
    return sqlite3.connect("user_results.db")

def create_table():
    with create_connection() as conn:
        # First create the basic table structure
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_results (
                username TEXT,
                date TEXT,
                grammar REAL,
                vocab REAL,
                speed REAL,
                cefr TEXT
            )
        """)
        
        # Check if new columns exist, if not add them
        try:
            conn.execute("SELECT pronunciation FROM user_results LIMIT 1")
        except sqlite3.OperationalError:
            # Add new columns to existing table
            conn.execute("ALTER TABLE user_results ADD COLUMN pronunciation REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN confidence REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN topic TEXT DEFAULT 'general'")
            conn.execute("ALTER TABLE user_results ADD COLUMN difficulty TEXT DEFAULT 'intermediate'")
            print("✅ Database schema updated successfully")
        
        # Add advanced analysis columns
        try:
            conn.execute("SELECT idiom_score FROM user_results LIMIT 1")
        except sqlite3.OperationalError:
            # Add advanced analysis columns
            conn.execute("ALTER TABLE user_results ADD COLUMN idiom_score REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN academic_score REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN complexity_score REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN discourse_score REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN register_score REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN ielts_band REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN coherence_score REAL DEFAULT 0")
            conn.execute("ALTER TABLE user_results ADD COLUMN task_achievement_score REAL DEFAULT 0")
            print("✅ Advanced analysis columns added successfully")
create_table()

def save_result_to_db(username, grammar, vocab, speed, cefr, pronunciation=0, confidence=0, topic="general", difficulty="intermediate",
                     idiom_score=0, academic_score=0, complexity_score=0, discourse_score=0, register_score=0, 
                     ielts_band=0, coherence_score=0, task_achievement_score=0):
    with create_connection() as conn:
        conn.execute("INSERT INTO user_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            username,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            grammar,
            vocab,
            speed,
            cefr,
            pronunciation,
            confidence,
            topic,
            difficulty,
            idiom_score,
            academic_score,
            complexity_score,
            discourse_score,
            register_score,
            ielts_band,
            coherence_score,
            task_achievement_score
        ))

def get_user_history(username):
    with create_connection() as conn:
        try:
            # Try to get all columns including advanced analysis ones
            return conn.execute("SELECT date, grammar, vocab, speed, pronunciation, confidence, topic, difficulty, idiom_score, academic_score, complexity_score, discourse_score, register_score, ielts_band, coherence_score, task_achievement_score FROM user_results WHERE username = ?", (username,)).fetchall()
        except sqlite3.OperationalError:
            try:
                # Fallback to intermediate schema
                return conn.execute("SELECT date, grammar, vocab, speed, pronunciation, confidence, topic, difficulty FROM user_results WHERE username = ?", (username,)).fetchall()
            except sqlite3.OperationalError:
                # Fallback to old schema if new columns don't exist
                return conn.execute("SELECT date, grammar, vocab, speed FROM user_results WHERE username = ?", (username,)).fetchall()

def get_leaderboard():
    with create_connection() as conn:
        try:
            # Try new schema with advanced analysis
            return conn.execute("""
                SELECT username, AVG(grammar) as avg_grammar, AVG(vocab) as avg_vocab, 
                       AVG(speed) as avg_speed, AVG(pronunciation) as avg_pronunciation,
                       AVG(ielts_band) as avg_ielts_band, COUNT(*) as attempts
                FROM user_results 
                GROUP BY username 
                HAVING attempts >= 3
                ORDER BY (avg_grammar + avg_vocab + avg_pronunciation + avg_ielts_band) / 4 DESC
                LIMIT 10
            """).fetchall()
        except sqlite3.OperationalError:
            try:
                # Fallback to intermediate schema with pronunciation
                return conn.execute("""
                    SELECT username, AVG(grammar) as avg_grammar, AVG(vocab) as avg_vocab, 
                           AVG(speed) as avg_speed, AVG(pronunciation) as avg_pronunciation,
                           0 as avg_ielts_band, COUNT(*) as attempts
                FROM user_results 
                GROUP BY username 
                HAVING attempts >= 3
                ORDER BY (avg_grammar + avg_vocab + avg_pronunciation) / 3 DESC
                LIMIT 10
            """).fetchall()
            except sqlite3.OperationalError:
                # Fallback to old schema without pronunciation
                return conn.execute("""
                    SELECT username, AVG(grammar) as avg_grammar, AVG(vocab) as avg_vocab, 
                           AVG(speed) as avg_speed, 0 as avg_pronunciation, 0 as avg_ielts_band,
                       COUNT(*) as attempts
                FROM user_results 
                GROUP BY username 
                HAVING attempts >= 3
                ORDER BY (avg_grammar + avg_vocab) / 2 DESC
                LIMIT 10
            """).fetchall()

# --- Accent Feedback ---
def generate_accent_comment(wpm):
    if wpm > 130:
        return "You speak very quickly — try to slow down for clarity."
    elif wpm < 90:
        return "Your pace is slow and clear — now work on rhythm."
    else:
        return "Balanced pace! Focus now on refining pronunciation."

FILLER_WORDS = {"um", "uh", "like", "you know", "so", "o", "basically", "actually", "well"}
def analyze_fluency(text):
    words = text.split()
    sentences = re.split(r'[.!?]', text)
    avg_len = np.mean([len(s.split()) for s in sentences if s.strip()])
    filler_count = sum(word.lower() in FILLER_WORDS for word in words)
    return round(avg_len, 2), filler_count

def generate_enhanced_pdf(grammar, vocab, speed, cefr, tip, accent, fluency, fillers, pronunciation_analysis, emotion_analysis, 
                         idiom_analysis, academic_analysis, complexity_analysis, discourse_analysis, register_analysis, 
                         ielts_analysis, path):
    def clean(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Enhanced English Speaking Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    
    # Basic metrics
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Basic Metrics", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, clean(f"• Grammar: {grammar:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Vocabulary: {vocab:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Speed: {speed:.1f} WPM"))
    pdf.multi_cell(0, 10, clean(f"• CEFR Level: {cefr}"))
    pdf.multi_cell(0, 10, clean(f"• IELTS Band: {ielts_analysis['band']:.1f}"))
    pdf.multi_cell(0, 10, clean(f"• Avg Sentence Length: {fluency}"))
    pdf.multi_cell(0, 10, clean(f"• Filler Words: {fillers}"))
    
    # Advanced metrics
    pdf.ln(5)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Advanced Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, clean(f"• Pronunciation Score: {pronunciation_analysis['pronunciation_score']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Speaking Confidence: {emotion_analysis['confidence_score']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Detected Emotion: {emotion_analysis['detected_emotion'].title()}"))
    pdf.multi_cell(0, 10, clean(f"• Idiom Score: {idiom_analysis['idiom_score']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Academic Vocabulary: {academic_analysis['academic_score']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Sentence Complexity: {complexity_analysis['complexity_score']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Discourse Markers: {discourse_analysis['discourse_score']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Register Type: {register_analysis['register_type'].title()}"))
    
    # IELTS Analysis
    pdf.ln(5)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="IELTS Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    detailed_scores = ielts_analysis['detailed_scores']
    pdf.multi_cell(0, 10, clean(f"• IELTS Band: {ielts_analysis['band']:.1f}"))
    pdf.multi_cell(0, 10, clean(f"• Band Descriptor: {ielts_analysis['band_descriptor']}"))
    pdf.multi_cell(0, 10, clean(f"• Grammar (IELTS): {detailed_scores['grammar']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Vocabulary (IELTS): {detailed_scores['vocabulary']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Pronunciation (IELTS): {detailed_scores['pronunciation']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Fluency (IELTS): {detailed_scores['fluency']:.1f}"))
    pdf.multi_cell(0, 10, clean(f"• Coherence (IELTS): {detailed_scores['coherence']:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"• Task Achievement (IELTS): {detailed_scores['task_achievement']:.1f}%"))
    
    # Feedback
    pdf.ln(5)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Feedback & Tips", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, clean(f"• Tip: {tip}"))
    pdf.multi_cell(0, 10, clean(f"• Accent Feedback: {accent}"))
    pdf.multi_cell(0, 10, clean(f"• Confidence Feedback: {emotion_analysis['feedback']}"))
    pdf.multi_cell(0, 10, clean(f"• IELTS Feedback: {ielts_calculator.get_band_feedback(ielts_analysis['band'])}"))
    
    # Advanced Language Features
    if idiom_analysis['found_idioms']:
        pdf.ln(5)
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(200, 10, txt="Idioms Used", ln=True)
        pdf.set_font("Arial", size=12)
        for idiom in idiom_analysis['found_idioms']:
            pdf.multi_cell(0, 10, clean(f"• {idiom}"))
    
    if academic_analysis['academic_words']:
        pdf.ln(5)
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(200, 10, txt="Academic Words Used", ln=True)
        pdf.set_font("Arial", size=12)
        for word in academic_analysis['academic_words']:
            pdf.multi_cell(0, 10, clean(f"• {word}"))
    
    # Accent improvement tips
    if pronunciation_analysis['accent_indicators']:
        pdf.ln(5)
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(200, 10, txt="Accent Improvement Tips", ln=True)
        pdf.set_font("Arial", size=12)
        for indicator in pronunciation_analysis['accent_indicators']:
            pdf.multi_cell(0, 10, clean(f"• {indicator}"))
    
    pdf.output(path)

def generate_pdf(grammar, vocab, speed, cefr, tip, accent, fluency, fillers, path):
    def clean(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="English Speaking Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, clean(f"- Grammar: {grammar:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"- Vocabulary: {vocab:.1f}%"))
    pdf.multi_cell(0, 10, clean(f"- Speed: {speed:.1f} WPM"))
    pdf.multi_cell(0, 10, clean(f"- CEFR Level: {cefr}"))
    pdf.multi_cell(0, 10, clean(f"- Avg Sentence Length: {fluency}"))
    pdf.multi_cell(0, 10, clean(f"- Filler Words: {fillers}"))
    pdf.multi_cell(0, 10, clean(f"- Tip: {tip}"))
    pdf.multi_cell(0, 10, clean(f"- Accent Feedback: {accent}"))
    pdf.output(path)

def generate_tts_summary(text, path):
    tts = gTTS(text=text, lang='en')
    tts.save(path)

def generate_tip(cefr, grammar, vocab, wpm, fluency, filler_count):
    tips = []

    # CEFR-specific foundational tip
    cefr_tips = {
        "A1/A2": "Practice basic sentence structures and pronunciation daily.",
        "B1": "Work on fluency and reduce hesitation when speaking.",
        "B2": "Enhance vocabulary and reduce filler words.",
        "C1": "Focus on natural flow, idioms, and complex grammar.",
        "C2": "Polish your discourse strategies and professional expressions."
    }
    tips.append(cefr_tips.get(cefr, "Keep practicing regularly."))

    # Grammar
    if grammar < 70:
        tips.append("Review subject-verb agreement and common tense errors.")
    elif grammar < 85:
        tips.append("Try combining simple and complex sentence structures.")

    # Vocabulary
    if vocab < 60:
        tips.append("Read articles and books to improve word variety.")
    elif vocab < 80:
        tips.append("Avoid repetition. Try using synonyms when speaking.")

    # Speaking speed
    if wpm > 140:
        tips.append("Slow down just a bit for better clarity.")
    elif wpm < 90:
        tips.append("Practice speaking aloud to increase confidence and speed.")

    # Fluency
    if fluency < 10:
        tips.append("Try joining ideas into longer, fluent sentences.")
    elif fluency > 20:
        tips.append("Break up long sentences for better rhythm.")

    # Filler words
    if filler_count > 5:
        tips.append("Avoid filler words like 'um', 'like', and 'basically'.")

    return random.choice(tips)

def evaluate_english(text, grammar_matches, wpm):
    words = text.split()
    vocab_score = (len(set(words)) / len(words)) * 100 if words else 0
    grammar_score = max(0, 100 - grammar_matches * 5)

    if vocab_score > 90 and grammar_score > 95 and wpm >= 130:
        cefr = "C2"
    elif vocab_score > 80 and grammar_score > 90 and wpm >= 110:
        cefr = "C1"
    elif vocab_score > 70 and grammar_score > 85 and wpm >= 90:
        cefr = "B2"
    elif vocab_score > 55 and grammar_score > 70 and wpm >= 70:
        cefr = "B1"
    else:
        cefr = "A1/A2"

    return grammar_score, vocab_score, wpm, cefr

def process_and_display_results(audio_data_np):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data_np, fs)
            audio_file = f.name

        st.audio(audio_file, format="audio/wav")
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            try:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
            except Exception as e:
                st.error(f"❌ Could not understand audio: {str(e)}")
                return

        # Enhanced analysis with error handling
        try:
            response = requests.post("https://api.languagetoolplus.com/v2/check", 
                                   data={"text": text, "language": "en-US"}, timeout=10)
            matches = response.json().get("matches", [])
        except Exception as e:
            st.warning(f"⚠️ Grammar check unavailable: {str(e)}")
            matches = []

        duration_sec = len(audio_data_np) / fs
        wpm = (len(text.split()) / duration_sec) * 60 if duration_sec > 0 else 0

        # Basic analysis
        grammar, vocab, wpm, cefr = evaluate_english(text, len(matches), wpm)
        accent_comment = generate_accent_comment(wpm)
        fluency, filler_count = analyze_fluency(text)
        tip = generate_tip(cefr, grammar, vocab, wpm, fluency, filler_count)

        # Advanced analysis with memory optimization
        try:
            pronunciation_analysis = st.session_state.pronunciation_analyzer.analyze_pronunciation(text, audio_data_np, fs)
        except Exception as e:
            st.warning(f"⚠️ Pronunciation analysis failed: {str(e)}")
            pronunciation_analysis = {
                'pronunciation_score': 50,
                'audio_features': {'pitch_std': 0, 'energy_std': 0, 'pitch_mean': 0, 'energy_mean': 0, 'spectral_centroid': 0},
                'accent_indicators': []
            }

        try:
            emotion_analysis = st.session_state.emotion_analyzer.analyze_speaking_confidence(audio_data_np, fs)
        except Exception as e:
            st.warning(f"⚠️ Emotion analysis failed: {str(e)}")
            emotion_analysis = {
                'confidence_score': 50,
                'detected_emotion': 'neutral',
                'confidence_level': 'medium',
                'feedback': 'Analysis unavailable',
                'emotion_probabilities': {'neutral': 1.0}
            }

        # Advanced Language Analysis
        try:
            # Optimize analysis by doing simpler checks first
            idiom_analysis = advanced_analyzer.analyze_idiomatic_expressions(text)
            academic_analysis = advanced_analyzer.analyze_academic_vocabulary(text)
            
            # Only do complex analysis if text is long enough
            if len(text.split()) > 10:
                complexity_analysis = advanced_analyzer.analyze_complex_sentences(text)
                discourse_analysis = advanced_analyzer.analyze_discourse_markers(text)
                register_analysis = advanced_analyzer.analyze_register(text)
            else:
                # Use simplified analysis for short texts
                complexity_analysis = {'complexity_score': 0, 'complex_sentences': 0, 'avg_sentence_length': len(text.split()), 'complexity_ratio': 0}
                discourse_analysis = {'discourse_score': 0, 'marker_count': 0, 'markers_found': {}}
                register_analysis = {'register_score': 50, 'register_type': 'neutral', 'formal_count': 0, 'informal_count': 0}
                
        except Exception as e:
            st.warning(f"⚠️ Advanced language analysis failed: {str(e)}")
            idiom_analysis = {'idiom_score': 0, 'found_idioms': [], 'idiom_count': 0}
            academic_analysis = {'academic_score': 0, 'academic_words': [], 'academic_count': 0, 'academic_ratio': 0}
            complexity_analysis = {'complexity_score': 0, 'complex_sentences': 0, 'avg_sentence_length': 0, 'complexity_ratio': 0}
            discourse_analysis = {'discourse_score': 0, 'marker_count': 0, 'markers_found': {}}
            register_analysis = {'register_score': 50, 'register_type': 'neutral', 'formal_count': 0, 'informal_count': 0}

        # Calculate coherence and task achievement scores
        coherence_score = (discourse_analysis['discourse_score'] + register_analysis['register_score']) / 2
        task_achievement_score = min((grammar + vocab + pronunciation_analysis['pronunciation_score']) / 3, 100)

        # Calculate IELTS Band Score
        try:
            ielts_analysis = ielts_calculator.calculate_ielts_band(
                grammar, vocab, pronunciation_analysis['pronunciation_score'], 
                fluency, coherence_score, task_achievement_score
            )
        except Exception as e:
            st.warning(f"⚠️ IELTS band calculation failed: {str(e)}")
            ielts_analysis = {
                'band': 5,
                'band_descriptor': 'Modest user - Has partial command, coping with overall meaning',
                'weighted_score': 50,
                'detailed_scores': {
                    'grammar': grammar, 'vocabulary': vocab, 'pronunciation': pronunciation_analysis['pronunciation_score'],
                    'fluency': fluency, 'coherence': coherence_score, 'task_achievement': task_achievement_score
                }
            }

        st.text_area("📝 Your Transcription:", text, height=150)

        # Enhanced metrics display
        st.markdown("### 📊 Comprehensive Analysis")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Basic Metrics", "🎯 Advanced Analysis", "🏆 IELTS Analysis", "📊 Detailed Charts"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = ['Grammar', 'Vocabulary', 'Speed', 'Pronunciation', 'Confidence']
            values = [grammar, vocab, min(wpm, 160), pronunciation_analysis['pronunciation_score'], emotion_analysis['confidence_score']]
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
            bars = ax.bar(labels, values, color=colors)
            ax.set_ylim(0, 100)
            ax.set_title("Speaking Performance Metrics", fontsize=14, pad=20)
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, 
                       f'{values[i]:.1f}', ha='center', color='white', fontweight='bold')
            st.pyplot(fig)

        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎤 Pronunciation Analysis")
                st.metric("Pronunciation Score", f"{pronunciation_analysis['pronunciation_score']:.1f}%")
                st.metric("Pitch Variation", f"{pronunciation_analysis['audio_features']['pitch_std']:.1f}")
                st.metric("Energy Consistency", f"{pronunciation_analysis['audio_features']['energy_std']:.3f}")
                
                if pronunciation_analysis['accent_indicators']:
                    st.markdown("#### 🎯 Accent Improvement Tips")
                    for indicator in pronunciation_analysis['accent_indicators']:
                        st.info(f"💡 {indicator}")
                
                st.markdown("#### 🗣️ Language Complexity")
                st.metric("Complexity Score", f"{complexity_analysis['complexity_score']:.1f}%")
                st.metric("Complex Sentences", complexity_analysis['complex_sentences'])
                st.metric("Avg Sentence Length", f"{complexity_analysis['avg_sentence_length']:.1f} words")
            
            with col2:
                st.markdown("#### 😊 Speaking Confidence")
                st.metric("Confidence Score", f"{emotion_analysis['confidence_score']:.1f}%")
                st.metric("Detected Emotion", emotion_analysis['detected_emotion'].title())
                st.metric("Confidence Level", emotion_analysis['confidence_level'].title())
                
                st.markdown("#### 💭 Confidence Feedback")
                st.success(f"💬 {emotion_analysis['feedback']}")
                
                st.markdown("#### 📚 Academic Vocabulary")
                st.metric("Academic Score", f"{academic_analysis['academic_score']:.1f}%")
                st.metric("Academic Words Used", academic_analysis['academic_count'])
                st.metric("Academic Ratio", f"{academic_analysis['academic_ratio']:.1f}%")

        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 IELTS Band Analysis")
                st.metric("IELTS Band", f"{ielts_analysis['band']:.1f}")
                st.metric("Weighted Score", f"{ielts_analysis['weighted_score']:.1f}%")
                st.info(f"**Band Descriptor:** {ielts_analysis['band_descriptor']}")
                
                st.markdown("#### 📊 Detailed IELTS Scores")
                detailed_scores = ielts_analysis['detailed_scores']
                st.metric("Grammar", f"{detailed_scores['grammar']:.1f}%")
                st.metric("Vocabulary", f"{detailed_scores['vocabulary']:.1f}%")
                st.metric("Pronunciation", f"{detailed_scores['pronunciation']:.1f}%")
                st.metric("Fluency", f"{detailed_scores['fluency']:.1f}")
                st.metric("Coherence", f"{detailed_scores['coherence']:.1f}%")
                st.metric("Task Achievement", f"{detailed_scores['task_achievement']:.1f}%")
            
            with col2:
                st.markdown("#### 🎯 Advanced Language Features")
                st.metric("Idiom Score", f"{idiom_analysis['idiom_score']:.1f}%")
                st.metric("Idioms Found", idiom_analysis['idiom_count'])
                
                if idiom_analysis['found_idioms']:
                    st.markdown("#### 💡 Idioms Used")
                    for idiom in idiom_analysis['found_idioms']:
                        st.success(f"✅ {idiom}")
                
                st.markdown("#### 🔗 Discourse Markers")
                st.metric("Discourse Score", f"{discourse_analysis['discourse_score']:.1f}%")
                st.metric("Markers Used", discourse_analysis['marker_count'])
                
                if discourse_analysis['markers_found']:
                    st.markdown("#### 📝 Discourse Categories")
                    for category, count in discourse_analysis['markers_found'].items():
                        st.info(f"**{category.title()}:** {count}")
                
                st.markdown("#### 📖 Register Analysis")
                st.metric("Register Score", f"{register_analysis['register_score']:.1f}%")
                st.metric("Register Type", register_analysis['register_type'].title())
                st.metric("Formal Indicators", register_analysis['formal_count'])
                st.metric("Informal Indicators", register_analysis['informal_count'])

        with tab4:
            # Advanced charts with new metrics
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Emotion probabilities
            emotions = list(emotion_analysis['emotion_probabilities'].keys())
            probs = list(emotion_analysis['emotion_probabilities'].values())
            ax1.bar(emotions, probs, color='skyblue')
            ax1.set_title("Emotion Analysis")
            ax1.tick_params(axis='x', rotation=45)
            
            # IELTS Band Analysis
            ielts_metrics = ['Grammar', 'Vocab', 'Pronunciation', 'Fluency', 'Coherence', 'Task Achievement']
            ielts_values = [
                detailed_scores['grammar'], detailed_scores['vocabulary'], 
                detailed_scores['pronunciation'], detailed_scores['fluency'],
                detailed_scores['coherence'], detailed_scores['task_achievement']
            ]
            ax2.bar(ielts_metrics, ielts_values, color='gold')
            ax2.set_title("IELTS Criteria Scores")
            ax2.tick_params(axis='x', rotation=45)
            
            # Advanced Language Features
            advanced_metrics = ['Idiom', 'Academic', 'Complexity', 'Discourse', 'Register']
            advanced_values = [
                idiom_analysis['idiom_score'], academic_analysis['academic_score'],
                complexity_analysis['complexity_score'], discourse_analysis['discourse_score'],
                register_analysis['register_score']
            ]
            ax3.bar(advanced_metrics, advanced_values, color='lightcoral')
            ax3.set_title("Advanced Language Features")
            ax3.tick_params(axis='x', rotation=45)
            
            # Overall Performance
            overall_metrics = ['Grammar', 'Vocab', 'Speed', 'Pronunciation', 'Confidence', 'IELTS Band']
            overall_values = [
                grammar, vocab, min(wpm, 160), pronunciation_analysis['pronunciation_score'], 
                emotion_analysis['confidence_score'], ielts_analysis['band'] * 10
            ]
            ax4.bar(overall_metrics, overall_values, color='lightgreen')
            ax4.set_title("Overall Performance")
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)

        # Enhanced report box
        st.markdown(f"""
            <div class="report-box">
                <div class="metric">📌 CEFR Level: {cefr}</div>
                <div class="metric">🏆 IELTS Band: {ielts_analysis['band']:.1f}</div>
                <div class="metric">🧠 Grammar: {grammar:.1f}%</div>
                <div class="metric">🗣 Vocabulary: {vocab:.1f}%</div>
                <div class="metric">⏱ Speed: {wpm:.1f} WPM</div>
                <div class="metric">🎤 Pronunciation: {pronunciation_analysis['pronunciation_score']:.1f}%</div>
                <div class="metric">😊 Confidence: {emotion_analysis['confidence_score']:.1f}%</div>
                <div class="metric">📏 Fluency: {fluency} words/sentence</div>
                <div class="metric">❗ Filler Words: {filler_count}</div>
                <div class="metric">💡 Idioms: {idiom_analysis['idiom_count']} found</div>
                <div class="metric">📚 Academic: {academic_analysis['academic_count']} words</div>
                <div class="metric">🔗 Discourse: {discourse_analysis['marker_count']} markers</div>
                <div class="metric">📖 Register: {register_analysis['register_type'].title()}</div>
                <br/>
                <b>💡 Tip:</b> <i>{tip}</i><br/>
                <b>🗣 Accent:</b> <i>{accent_comment}</i><br/>
                <b>😊 Emotion:</b> <i>{emotion_analysis['detected_emotion'].title()}</i><br/>
                <b>🏆 IELTS Feedback:</b> <i>{ielts_calculator.get_band_feedback(ielts_analysis['band'])}</i>
            </div>
        """, unsafe_allow_html=True)

        # Save enhanced results with advanced analysis
        save_result_to_db(username, grammar, vocab, wpm, cefr, 
                         pronunciation_analysis['pronunciation_score'], 
                         emotion_analysis['confidence_score'],
                         st.session_state.current_topic,
                         st.session_state.current_difficulty,
                         idiom_analysis['idiom_score'],
                         academic_analysis['academic_score'],
                         complexity_analysis['complexity_score'],
                         discourse_analysis['discourse_score'],
                         register_analysis['register_score'],
                         ielts_analysis['band'],
                         coherence_score,
                         task_achievement_score)

        # Enhanced PDF and audio
        pdf_path = os.path.join(tempfile.gettempdir(), "english_report.pdf")
        audio_path = os.path.join(tempfile.gettempdir(), "summary.mp3")
        generate_enhanced_pdf(grammar, vocab, wpm, cefr, tip, accent_comment, fluency, filler_count, 
                             pronunciation_analysis, emotion_analysis, idiom_analysis, academic_analysis, 
                             complexity_analysis, discourse_analysis, register_analysis, ielts_analysis, pdf_path)
        generate_tts_summary(f"Your CEFR level is {cefr} and IELTS band is {ielts_analysis['band']:.1f}. Pronunciation: {pronunciation_analysis['pronunciation_score']:.1f}%. Confidence: {emotion_analysis['confidence_score']:.1f}%. {tip}", audio_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Enhanced PDF Report", f, file_name="English_Report.pdf")

        with open(audio_path, "rb") as a:
            st.audio(a.read(), format="audio/mp3")
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Try speaking more clearly or check your microphone settings")

# --- Style ---
st.set_page_config(page_title="🎤 English Analyzer Pro", layout="centered")

# Native Streamlit title
st.markdown("# English Analyzer Pro")
st.markdown("### Speak. Analyze. Improve.")

# CSS Styling Block (correct usage)
st.markdown("""
    <style>
    body { background-color: #0f111a; }
    .title {
        font-size: 44px;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
        color: #aaa;
        margin-bottom: 20px;
    }
    .report-box {
        background: #1e1e2e;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 25px rgba(0,255,255,0.05);
        color: #f0f0f0;
        font-size: 16px;
        margin-top: 30px;
    }
    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 14px;
        color: #666;
    }
    .metric {
        font-size: 22px;
        font-weight: bold;
        color: #66d9ef;
    }
    </style>
""", unsafe_allow_html=True)

# Optional: Also render the same styled header using HTML
st.markdown('<div class="title">English Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">THE ONLY PAGE YOU NEED TO IMPROVE YOUR ENGLISH. WELCOME TO THE FULLY AI AUTOMATED IELTS LEVEL ENGLISH ENHANCER.</div>', unsafe_allow_html=True)
st.markdown("---")

# Initialize placeholders
placeholder = st.empty()
progress_placeholder = st.empty()

# Initialize analyzers and content manager
if "pronunciation_analyzer" not in st.session_state:
    st.session_state.pronunciation_analyzer = PronunciationAnalyzer()
if "emotion_analyzer" not in st.session_state:
    st.session_state.emotion_analyzer = EmotionAnalyzer()
if "content_manager" not in st.session_state:
    st.session_state.content_manager = ContentManager()

# Initialize session state variables
if "selected_sentence" not in st.session_state:
    st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_topic('business')
if "current_topic" not in st.session_state:
    st.session_state.current_topic = "business"
if "current_difficulty" not in st.session_state:
    st.session_state.current_difficulty = "intermediate"
if "current_exercise" not in st.session_state:
    st.session_state.current_exercise = None
if "exercise_mode" not in st.session_state:
    st.session_state.exercise_mode = False

fs = 16000
if "rec_duration" not in st.session_state:
    st.session_state.rec_duration = 8
rec_duration = st.session_state.rec_duration

# Content Selection Interface
st.markdown("### 🎯 Content Settings")
col1, col2, col3 = st.columns(3)

with col1:
    topic = st.selectbox("📚 Topic", st.session_state.content_manager.get_all_topics(), 
                        index=st.session_state.content_manager.get_all_topics().index(st.session_state.current_topic),
                        key="topic_selector")
    if topic != st.session_state.current_topic:
        st.session_state.current_topic = topic
        st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_topic(topic)
        st.session_state.exercise_mode = False
        st.session_state.current_exercise = None
        st.rerun()

with col2:
    difficulty = st.selectbox("📊 Difficulty", st.session_state.content_manager.get_all_difficulties(),
                            index=st.session_state.content_manager.get_all_difficulties().index(st.session_state.current_difficulty),
                            key="difficulty_selector")
    if difficulty != st.session_state.current_difficulty:
        st.session_state.current_difficulty = difficulty
        st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_difficulty(difficulty)
        st.session_state.exercise_mode = False
        st.session_state.current_exercise = None
        st.rerun()

with col3:
    exercise_type = st.selectbox("🏋️ Exercise Type", ["Regular Practice"] + st.session_state.content_manager.get_all_exercise_types(),
                                key="exercise_selector")
    if exercise_type != "Regular Practice":
        if not st.session_state.exercise_mode or st.session_state.current_exercise is None or st.session_state.current_exercise.get('type') != exercise_type:
            st.session_state.exercise_mode = True
            st.session_state.current_exercise = st.session_state.content_manager.get_interactive_exercise(exercise_type)
            st.rerun()
    else:
        if st.session_state.exercise_mode:
            st.session_state.exercise_mode = False
            st.session_state.current_exercise = None
            st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_topic(st.session_state.current_topic)
            st.rerun()

# Custom input option
st.markdown("### ✍️ Custom Practice")
custom_text = st.text_area("Practice your own text:", 
                          placeholder="Type your own sentence or paragraph here...",
                          height=100)

if custom_text.strip():
    st.session_state.selected_sentence = custom_text
    st.session_state.exercise_mode = False
    st.session_state.current_exercise = None

# Display current content
if st.session_state.exercise_mode and st.session_state.current_exercise:
    st.markdown(f"""🎯 **Exercise:** {st.session_state.current_exercise['type'].title()}<br>
    <i>\"{st.session_state.current_exercise['content']}\"</i><br>
    <small>💡 {st.session_state.current_exercise['instruction']}</small>""", unsafe_allow_html=True)
else:
    st.markdown(f"""📘 **Read this aloud:**<br>
    <small>📚 Topic: {st.session_state.current_topic.title()} | 📊 Difficulty: {st.session_state.current_difficulty.title()}</small><br>
    <i>\"{st.session_state.selected_sentence}\"</i>""", unsafe_allow_html=True)

# File upload option for cloud deployment
if not SOUNDDEVICE_AVAILABLE:
    st.markdown("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader("Upload an audio file for analysis (WAV, MP3)", type=['wav', 'mp3'])
    if uploaded_file is not None:
        try:
            # Read the uploaded audio file
            audio_data, fs = sf.read(uploaded_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Convert to mono
            
            st.success("✅ Audio file loaded successfully!")
            
            # Analyze the uploaded audio
            with st.spinner("Analyzing your speech..."):
                process_and_display_results(audio_data)
                
        except Exception as e:
            st.error(f"❌ Error reading audio file: {str(e)}")

# Recording and New Content buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Start Recording", key="start_recording_button"):
        # Clear previous content
        placeholder.empty()
        progress_placeholder.empty()
        
        # Create countdown timer with single box display
        countdown_container = st.container()
        
        # Countdown timer with single box
        countdown_container.info("Get ready to record...")
        time.sleep(1)
        
        # Single countdown box that updates
        countdown_box = countdown_container.empty()
        for i in range(3, 0, -1):
            countdown_box.markdown(f"""
            <div style="
                text-align: center; 
                font-size: 48px; 
                font-weight: bold; 
                color: #ff6b6b; 
                background: #2d3748; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0;
                border: 2px solid #ff6b6b;
            ">
                {i}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
        
        # Clear the countdown box and show final message
        countdown_box.empty()
        countdown_container.markdown("""
        <div style="
            text-align: center; 
            font-size: 36px; 
            font-weight: bold; 
            color: #4ade80; 
            background: #064e3b; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
            border: 2px solid #4ade80;
        ">
            RECORDING NOW!
        </div>
        """, unsafe_allow_html=True)
        
        # Start recording with visual feedback
        if not SOUNDDEVICE_AVAILABLE:
            st.error("❌ Audio recording is not available on this platform.")
            st.info("💡 Please use the file upload option or deploy locally for full functionality.")
            st.stop()
        
        recording = sd.rec(int(rec_duration * fs), samplerate=fs, channels=1)
        
        # Real-time feedback during recording
        for second in range(rec_duration):
            # Create a visual progress indicator
            progress = (second + 1) / rec_duration
            progress_placeholder.progress(progress, text=f"Recording... {second + 1}s / {rec_duration}s")
            
            # Real-time volume monitoring
            if second > 0:  # Start monitoring after first second
                current_audio = recording[:int((second + 1) * fs)]
                try:
                    # Add bounds checking to prevent overflow
                    audio_squared = np.clip(current_audio**2, 0, 1e6)
                    volume_level = np.sqrt(np.mean(audio_squared))
                except:
                    volume_level = 0
                
                if volume_level < 0.01:
                    progress_placeholder.warning("Speak louder!")
                elif volume_level > 0.1:
                    progress_placeholder.info("Good volume level")
                else:
                    progress_placeholder.success("Recording well!")
            
            time.sleep(1)
        
        sd.wait()
        
        # Clear countdown display
        countdown_container.empty()

        # Create a dedicated analysis section
        analysis_container = st.container()
        
        with analysis_container:
            st.success("🎙️ **Recording Complete!**")
            st.info("🔄 **Starting Analysis...**")

        with st.spinner("Analyzing your speech..."):
            try:
                # Show comprehensive analysis message
                st.info("🔍 **Comprehensive Analysis in Progress**")
                st.markdown("""
                **Please wait 20-30 seconds** for our AI to analyze:
                • 🎤 Pronunciation & accent detection
                • 📝 Grammar & vocabulary assessment  
                • 😊 Speaking confidence & emotion
                • 🏆 IELTS band scoring
                • 💡 Advanced language features
                • 📊 Detailed performance metrics
                """)
                
                # Show progress steps
                progress_bar = st.progress(0)
                st.text("Step 1/5: Processing audio...")
                progress_bar.progress(20)
                
                process_and_display_results(recording)
                
                progress_bar.progress(100)
                st.success("✅ **Analysis Complete!**")
                
                # Auto-advance content if enabled
                if st.session_state.get("auto_advance", False):
                    if st.session_state.exercise_mode and st.session_state.current_exercise:
                        st.session_state.current_exercise = st.session_state.content_manager.get_interactive_exercise(
                            st.session_state.current_exercise['type']
                        )
                    else:
                        st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_difficulty(st.session_state.current_difficulty)
                        st.success("🔄 Content auto-advanced for next practice!")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Try speaking more clearly or check your microphone settings")

with col2:
    if st.button("🔄 New Content", key="new_content_button"):
        if st.session_state.exercise_mode and st.session_state.current_exercise:
            # Get new exercise of the same type
            st.session_state.current_exercise = st.session_state.content_manager.get_interactive_exercise(
                st.session_state.current_exercise['type']
            )
        else:
            # Get new sentence based on current topic and difficulty
            st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_difficulty(st.session_state.current_difficulty)
        st.rerun()

st.markdown("---")

# --- Enhanced Progress Tracking ---
user_data = get_user_history(username)

if user_data and len(user_data) >= 2:
    st.markdown("### 📈 Enhanced Progress Tracking")
    
    # Create tabs for different progress views
    progress_tab1, progress_tab2, progress_tab3 = st.tabs(["📊 Basic Progress", "🎯 Advanced Metrics", "🏆 Leaderboard"])
    
    with progress_tab1:
        dates = [x[0] for x in user_data]
        grammar = []
        vocab = []
        speed = []
        
        for x in user_data:
            try:
                grammar.append(float(x[1]) if not isinstance(x[1], bytes) else 0)
                vocab.append(float(x[2]) if not isinstance(x[2], bytes) else 0)
                speed.append(float(x[3]) if not isinstance(x[3], bytes) else 0)
            except (ValueError, TypeError):
                grammar.append(0)
                vocab.append(0)
                speed.append(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, grammar, label="Grammar", marker="o", linewidth=2, color="#4CAF50")
        ax.plot(dates, vocab, label="Vocabulary", marker="s", linewidth=2, color="#2196F3")
        ax.plot(dates, speed, label="Speed", marker="^", linewidth=2, color="#FF9800")

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Score / WPM", fontsize=11)
        ax.set_title("Timeline of Speaking Performance", fontsize=13, color="white", pad=15)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.set_facecolor("#111")
        fig.patch.set_facecolor('#0f111a')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with progress_tab2:
        # Advanced metrics progress - handle both old and new schema
        if len(user_data[0]) >= 8:  # New schema with advanced analysis
            pronunciation = []
            confidence = []
            ielts_bands = []
            idiom_scores = []
            academic_scores = []
            
            for x in user_data:
                # Safely convert and check pronunciation values
                if x[4] is not None:
                    try:
                        pron_val = float(x[4]) if not isinstance(x[4], bytes) else 0
                        if pron_val > 0:
                            pronunciation.append(pron_val)
                    except (ValueError, TypeError):
                        continue
                
                # Safely convert and check confidence values
                if x[5] is not None:
                    try:
                        conf_val = float(x[5]) if not isinstance(x[5], bytes) else 0
                        if conf_val > 0:
                            confidence.append(conf_val)
                    except (ValueError, TypeError):
                        continue
                
                # Safely convert and check IELTS band values
                if x[8] is not None:
                    try:
                        ielts_val = float(x[8]) if not isinstance(x[8], bytes) else 0
                        if ielts_val > 0:
                            ielts_bands.append(ielts_val)
                    except (ValueError, TypeError):
                        continue
                
                # Safely convert and check idiom scores
                if x[9] is not None:
                    try:
                        idiom_val = float(x[9]) if not isinstance(x[9], bytes) else 0
                        if idiom_val > 0:
                            idiom_scores.append(idiom_val)
                    except (ValueError, TypeError):
                        continue
                
                # Safely convert and check academic scores
                if x[10] is not None:
                    try:
                        academic_val = float(x[10]) if not isinstance(x[10], bytes) else 0
                        if academic_val > 0:
                            academic_scores.append(academic_val)
                    except (ValueError, TypeError):
                        continue
        else:  # Old schema without advanced analysis
            pronunciation = []
            confidence = []
            ielts_bands = []
            idiom_scores = []
            academic_scores = []
        
        if len(pronunciation) >= 2 and len(confidence) >= 2:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Pronunciation progress
            ax1.plot(dates[-len(pronunciation):], pronunciation, marker="o", linewidth=2, color="#9C27B0")
            ax1.set_title("Pronunciation Progress", fontsize=12)
            ax1.set_ylabel("Pronunciation Score (%)")
            ax1.grid(True, alpha=0.3)
            
            # Confidence progress
            ax2.plot(dates[-len(confidence):], confidence, marker="s", linewidth=2, color="#F44336")
            ax2.set_title("Speaking Confidence Progress", fontsize=12)
            ax2.set_ylabel("Confidence Score (%)")
            ax2.grid(True, alpha=0.3)
            
            # IELTS Band progress
            if len(ielts_bands) >= 2:
                ax3.plot(dates[-len(ielts_bands):], ielts_bands, marker="^", linewidth=2, color="#FF9800")
                ax3.set_title("IELTS Band Progress", fontsize=12)
                ax3.set_ylabel("IELTS Band")
                ax3.grid(True, alpha=0.3)
            
            # Advanced Language Features
            if len(idiom_scores) >= 2 and len(academic_scores) >= 2:
                ax4.plot(dates[-len(idiom_scores):], idiom_scores, marker="d", linewidth=2, color="#4CAF50", label="Idiom Score")
                ax4.plot(dates[-len(academic_scores):], academic_scores, marker="*", linewidth=2, color="#2196F3", label="Academic Score")
                ax4.set_title("Advanced Language Features", fontsize=12)
                ax4.set_ylabel("Score (%)")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("📈 More practice sessions needed to show advanced metrics progress")
    
    with progress_tab3:
        # Leaderboard
        leaderboard_data = get_leaderboard()
        if leaderboard_data:
            st.markdown("#### 🏆 Top Performers")
            
            # Create DataFrame for better display - handle all schemas
            if len(leaderboard_data[0]) >= 7:  # New schema with IELTS bands
                df = pd.DataFrame(leaderboard_data, columns=['Username', 'Avg Grammar', 'Avg Vocab', 'Avg Speed', 'Avg Pronunciation', 'Avg IELTS Band', 'Attempts'])
                df['Overall Score'] = (df['Avg Grammar'] + df['Avg Vocab'] + df['Avg Pronunciation'] + df['Avg IELTS Band']) / 4
            elif len(leaderboard_data[0]) >= 6:  # Intermediate schema
                df = pd.DataFrame(leaderboard_data, columns=['Username', 'Avg Grammar', 'Avg Vocab', 'Avg Speed', 'Avg Pronunciation', 'Avg IELTS Band', 'Attempts'])
                df['Overall Score'] = (df['Avg Grammar'] + df['Avg Vocab'] + df['Avg Pronunciation']) / 3
            else:  # Old schema
                df = pd.DataFrame(leaderboard_data, columns=['Username', 'Avg Grammar', 'Avg Vocab', 'Avg Speed', 'Avg Pronunciation', 'Avg IELTS Band', 'Attempts'])
                df['Overall Score'] = (df['Avg Grammar'] + df['Avg Vocab']) / 2
            
            # Highlight current user
            if username in df['Username'].values:
                st.success(f"🎉 You're ranked #{df[df['Username'] == username].index[0] + 1} on the leaderboard!")
            
            # Display leaderboard
            st.dataframe(df.sort_values('Overall Score', ascending=False).head(10), use_container_width=True)
        else:
            st.info("🏆 Leaderboard will appear once more users have completed practice sessions")

# Sidebar function
def render_sidebar():
    st.sidebar.success(f"Logged in as {name}")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.auth = {"logged_in": False, "username": None, "name": None}
        st.rerun()

    st.sidebar.markdown("---")

    # User Statistics
    st.sidebar.markdown("### 📊 Your Stats")
    user_data = get_user_history(username)
    if user_data:
        total_sessions = len(user_data)
        if len(user_data[0]) >= 8:  # New schema with advanced analysis
            # Safely calculate averages with proper type conversion
            grammar_values = []
            vocab_values = []
            pronunciation_values = []
            ielts_values = []
            idiom_values = []
            academic_values = []
            
            for x in user_data:
                try:
                    grammar_values.append(float(x[1]) if not isinstance(x[1], bytes) else 0)
                    vocab_values.append(float(x[2]) if not isinstance(x[2], bytes) else 0)
                    if x[4] is not None:
                        pron_val = float(x[4]) if not isinstance(x[4], bytes) else 0
                        if pron_val > 0:
                            pronunciation_values.append(pron_val)
                    if x[8] is not None:
                        ielts_val = float(x[8]) if not isinstance(x[8], bytes) else 0
                        if ielts_val > 0:
                            ielts_values.append(ielts_val)
                    if x[9] is not None:
                        idiom_val = float(x[9]) if not isinstance(x[9], bytes) else 0
                        if idiom_val > 0:
                            idiom_values.append(idiom_val)
                    if x[10] is not None:
                        academic_val = float(x[10]) if not isinstance(x[10], bytes) else 0
                        if academic_val > 0:
                            academic_values.append(academic_val)
                except (ValueError, TypeError):
                    continue
            
            avg_grammar = sum(grammar_values) / len(grammar_values) if grammar_values else 0
            avg_vocab = sum(vocab_values) / len(vocab_values) if vocab_values else 0
            avg_pronunciation = sum(pronunciation_values) / len(pronunciation_values) if pronunciation_values else 0
            avg_ielts = sum(ielts_values) / len(ielts_values) if ielts_values else 0
            avg_idiom = sum(idiom_values) / len(idiom_values) if idiom_values else 0
            avg_academic = sum(academic_values) / len(academic_values) if academic_values else 0
            
            st.sidebar.metric("Total Sessions", total_sessions)
            st.sidebar.metric("Avg Grammar", f"{avg_grammar:.1f}%")
            st.sidebar.metric("Avg Vocabulary", f"{avg_vocab:.1f}%")
            st.sidebar.metric("Avg Pronunciation", f"{avg_pronunciation:.1f}%")
            st.sidebar.metric("Avg IELTS Band", f"{avg_ielts:.1f}")
            st.sidebar.metric("Avg Idiom Score", f"{avg_idiom:.1f}%")
            st.sidebar.metric("Avg Academic Score", f"{avg_academic:.1f}%")
        elif len(user_data[0]) >= 6:  # Intermediate schema
            # Safely calculate averages with proper type conversion
            grammar_values = []
            vocab_values = []
            pronunciation_values = []
            
            for x in user_data:
                try:
                    grammar_values.append(float(x[1]) if not isinstance(x[1], bytes) else 0)
                    vocab_values.append(float(x[2]) if not isinstance(x[2], bytes) else 0)
                    if x[4] is not None:
                        pron_val = float(x[4]) if not isinstance(x[4], bytes) else 0
                        if pron_val > 0:
                            pronunciation_values.append(pron_val)
                except (ValueError, TypeError):
                    continue
            
            avg_grammar = sum(grammar_values) / len(grammar_values) if grammar_values else 0
            avg_vocab = sum(vocab_values) / len(vocab_values) if vocab_values else 0
            avg_pronunciation = sum(pronunciation_values) / len(pronunciation_values) if pronunciation_values else 0
            
            st.sidebar.metric("Total Sessions", total_sessions)
            st.sidebar.metric("Avg Grammar", f"{avg_grammar:.1f}%")
            st.sidebar.metric("Avg Vocabulary", f"{avg_vocab:.1f}%")
            st.sidebar.metric("Avg Pronunciation", f"{avg_pronunciation:.1f}%")
        else:  # Old schema
            # Safely calculate averages for old schema
            grammar_values = []
            vocab_values = []
            
            for x in user_data:
                try:
                    grammar_values.append(float(x[1]) if not isinstance(x[1], bytes) else 0)
                    vocab_values.append(float(x[2]) if not isinstance(x[2], bytes) else 0)
                except (ValueError, TypeError):
                    continue
            
            avg_grammar = sum(grammar_values) / len(grammar_values) if grammar_values else 0
            avg_vocab = sum(vocab_values) / len(vocab_values) if vocab_values else 0
            
            st.sidebar.metric("Total Sessions", total_sessions)
            st.sidebar.metric("Avg Grammar", f"{avg_grammar:.1f}%")
            st.sidebar.metric("Avg Vocabulary", f"{avg_vocab:.1f}%")
    else:
        st.sidebar.info("No practice sessions yet. Start practicing to see your stats!")

    st.sidebar.markdown("---")

    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🎯 Practice Random Topic", key="random_topic"):
        random_topic = random.choice(st.session_state.content_manager.get_all_topics())
        st.session_state.current_topic = random_topic
        st.session_state.selected_sentence = st.session_state.content_manager.get_sentence_by_topic(random_topic)
        st.session_state.exercise_mode = False
        st.rerun()

    if st.sidebar.button("🏋️ Random Exercise", key="random_exercise"):
        random_exercise = random.choice(st.session_state.content_manager.get_all_exercise_types())
        st.session_state.exercise_mode = True
        st.session_state.current_exercise = st.session_state.content_manager.get_interactive_exercise(random_exercise)
        st.rerun()

    if st.sidebar.button("📈 View Progress", key="view_progress"):
        st.sidebar.success("Scroll down to see your progress charts!")

    st.sidebar.markdown("---")

    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    recording_duration = st.sidebar.slider("Recording Duration (seconds)", 5, 15, 8, key="duration_slider")
    if recording_duration != rec_duration:
        st.session_state.rec_duration = recording_duration

    auto_advance = st.sidebar.checkbox("Auto-advance content", value=False, key="auto_advance")
    if auto_advance:
        st.sidebar.info("Content will change automatically after each practice session")

    st.sidebar.markdown("---")

    # Help & Tips
    st.sidebar.markdown("### 💡 Tips")
    with st.sidebar.expander("🎤 Speaking Tips"):
        st.markdown("""
        - **Speak clearly** and at a natural pace
        - **Vary your pitch** for more engaging speech
        - **Practice regularly** for best results
        - **Use the exercises** to improve specific skills
        """)

    with st.sidebar.expander("📊 Understanding Scores"):
        st.markdown("""
        - **Grammar**: Accuracy of sentence structure
        - **Vocabulary**: Word variety and complexity
        - **Pronunciation**: Clarity and accent reduction
        - **Confidence**: Speaking energy and emotion
        - **Speed**: Words per minute (WPM)
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📘 About")
    st.sidebar.info("""
    This app is built by **Muhammad Jarreer** using AI-based evaluation to give real-time English speaking feedback.

    🔗 [Portfolio](https://jarreer.github.io/portfolio/)  
    📨 [Contact](mailto:jareerfootball7@gmail.com)  
    💡 Built with `Streamlit`, `gTTS`, `SQLite`, `python`, `scikit-learn`, and `FPDF`
    """)

# Call sidebar function
render_sidebar()

st.markdown('<div class="footer">© 2025 Muhammad Jarreer — All Rights Reserved</div>', unsafe_allow_html=True)