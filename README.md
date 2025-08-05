# 🎤 English Analyzer Pro

**AI-powered English speaking analyzer with IELTS-level assessment and real-time feedback**

[![Status](https://img.shields.io/badge/Status-Live-green?style=for-the-badge)](https://your-streamlit-app-url.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![AI/ML](https://img.shields.io/badge/AI/ML-Powered-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)

## 🚀 Features

### 🎤 **Real-time Audio Analysis**
- Record and analyze spoken English instantly
- Advanced pronunciation assessment with accent detection
- Speaking confidence and emotion analysis
- Real-time feedback during recording

### 🏆 **IELTS Band Scoring**
- Comprehensive IELTS-level assessment
- Detailed band scoring (1-9) with descriptors
- Grammar, vocabulary, pronunciation evaluation
- Fluency and coherence analysis
- Task achievement assessment

### 📝 **Advanced Language Analysis**
- **Grammar Analysis**: Real-time grammar checking and scoring
- **Vocabulary Assessment**: Word variety and complexity analysis
- **Pronunciation Scoring**: Audio-based pronunciation evaluation
- **Speaking Speed**: Words per minute (WPM) measurement
- **Idiomatic Expressions**: Detection and scoring of idioms usage
- **Academic Vocabulary**: Academic vs. everyday vocabulary analysis
- **Discourse Markers**: Analysis of language flow and coherence
- **Register Analysis**: Formal vs. informal language detection

### 😊 **Speaking Confidence Detection**
- Emotion analysis using AI models
- Confidence level assessment
- Speaking energy and engagement analysis
- Personalized confidence feedback

### 📊 **Progress Tracking & Analytics**
- Historical performance tracking
- Detailed progress charts and visualizations
- Performance comparison over time
- Leaderboard system for motivation
- Comprehensive statistics dashboard

### 📄 **Professional Reports**
- Detailed PDF analysis reports
- Audio summaries using text-to-speech
- Exportable performance data
- Customizable report formats

### 🔐 **User Management**
- Secure user authentication system
- Individual user profiles and progress
- Session management and data persistence
- Privacy-focused data handling

### 🎯 **Interactive Learning**
- Topic-based practice sessions
- Multiple difficulty levels (Beginner to Advanced)
- Interactive exercises and tongue twisters
- Custom text input for personalized practice
- Real-time content suggestions

## 🛠️ Technology Stack

### **Frontend & UI**
- **Streamlit** - Interactive web application framework
- **HTML/CSS** - Custom styling and responsive design
- **Matplotlib** - Data visualization and charts

### **Backend & Processing**
- **Python 3.8+** - Core programming language
- **SQLite** - Lightweight database for user data
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis

### **AI & Machine Learning**
- **scikit-learn** - Machine learning algorithms
- **librosa** - Audio and music analysis
- **NLTK** - Natural language processing
- **Joblib** - Model serialization and loading

### **Audio Processing**
- **sounddevice** - Real-time audio recording
- **soundfile** - Audio file handling
- **speech_recognition** - Speech-to-text conversion
- **gTTS** - Text-to-speech synthesis

### **Security & Authentication**
- **bcrypt** - Password hashing and security
- **SQLite** - Secure user data storage

### **Documentation & Reports**
- **FPDF** - PDF report generation
- **requests** - External API integration

## 🎯 Live Demo

[Try the English Analyzer Pro](https://your-streamlit-app-url.streamlit.app/)

## 📸 Screenshots

### Main Dashboard
![Dashboard](assets/images/dashboard.png)

### Analysis Results
![Analysis](assets/images/analysis.png)

### Progress Tracking
![Progress](assets/images/progress.png)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Microphone access (for audio recording)
- Internet connection (for external APIs)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Jarreer/english-analyzer-pro.git
cd english-analyzer-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📊 Project Structure

```
emotion_app/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── utils/                    # Utility modules
│   ├── pronunciation_analyzer.py    # Audio and pronunciation analysis
│   ├── emotion_integration.py       # Emotion and confidence detection
│   ├── content_manager.py           # Content and exercise management
│   └── extract_features.py          # Feature extraction utilities
├── model/                    # Machine learning models
│   ├── emotion_model.pkl     # Pre-trained emotion detection model
│   └── train_model.py        # Model training script
├── assets/                   # Static assets
│   ├── css/                  # Custom stylesheets
│   │   ├── style.css
│   │   └── space.css
│   ├── js/                   # JavaScript files
│   │   └── space.js
│   └── images/               # Images and icons
│       └── profile.png
├── data/                     # Data storage
└── venv/                     # Virtual environment (not included in repo)
```

## 🎓 Key Features Explained

### **IELTS Band Scoring System**
The app implements a comprehensive IELTS scoring system that evaluates:
- **Grammar** (25% weight): Sentence structure, tense usage, agreement
- **Vocabulary** (20% weight): Word variety, complexity, appropriateness
- **Pronunciation** (20% weight): Clarity, accent, intonation
- **Fluency** (15% weight): Speaking speed, hesitation, flow
- **Coherence** (10% weight): Logical organization, discourse markers
- **Task Achievement** (10% weight): Response relevance and completeness

### **Advanced Language Analysis**
- **Idiomatic Expressions**: Detects and scores usage of common English idioms
- **Academic Vocabulary**: Analyzes formal vs. informal language usage
- **Complexity Analysis**: Evaluates sentence structure complexity
- **Discourse Markers**: Identifies language flow and coherence indicators
- **Register Analysis**: Determines formal vs. informal communication style

### **Real-time Audio Processing**
- **Pitch Analysis**: Evaluates intonation and speaking patterns
- **Energy Detection**: Measures speaking energy and engagement
- **Spectral Features**: Analyzes audio quality and clarity
- **Stress Pattern Analysis**: Evaluates word stress and emphasis

## 🔧 Configuration

### **Environment Variables**
```bash
# Optional: Set custom database path
DATABASE_PATH=user_results.db

# Optional: Set custom model paths
EMOTION_MODEL_PATH=model/emotion_model.pkl
```

### **Customization Options**
- Modify `utils/content_manager.py` to add new topics and exercises
- Adjust scoring weights in `app.py` for different assessment criteria
- Customize UI styling in `assets/css/style.css`

## 📈 Performance Metrics

The app tracks comprehensive performance metrics including:
- **Grammar Score**: 0-100% accuracy
- **Vocabulary Score**: 0-100% variety and complexity
- **Pronunciation Score**: 0-100% clarity and accent
- **Speaking Speed**: Words per minute (WPM)
- **Confidence Score**: 0-100% speaking confidence
- **IELTS Band**: 1-9 band score
- **CEFR Level**: A1-C2 proficiency levels

## 🎓 About the Developer

**Muhammad Jarreer** - AI Engineer & Python Developer

- 🔗 [Portfolio](https://jarreer.github.io/portfolio/)
- 📧 [Email](mailto:jareerfootball7@gmail.com)
- 💼 [LinkedIn](https://linkedin.com/in/your-profile)
- 🐙 [GitHub](https://github.com/Jarreer)

### **Expertise**
- **AI/ML Development**: Building intelligent tools that enhance real-world productivity
- **NLP & Speech Processing**: Specialized in natural language understanding and audio analysis
- **Streamlit Applications**: Creating beautiful, interactive data science applications
- **Full-Stack Development**: End-to-end application development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### **How to Contribute**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **OpenAI** for inspiration in AI-powered applications
- **NLTK** for natural language processing capabilities
- **librosa** for audio analysis features
- **scikit-learn** for machine learning algorithms

## 📞 Support

If you have any questions or need support:
- 📧 Email: jareerfootball7@gmail.com
- 🐙 GitHub Issues: [Create an issue](https://github.com/Jarreer/english-analyzer-pro/issues)
- 📖 Documentation: Check the code comments and docstrings

---

**Built with ❤️ by Muhammad Jarreer**

*"Solving real problems through clean, impactful code."* 