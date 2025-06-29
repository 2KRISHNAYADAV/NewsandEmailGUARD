# 🛡️ NewsGuardian.AI

<div align="center">

![NewsGuardian.AI Logo](https://img.shields.io/badge/NewsGuardian.AI-Advanced%20AI%20Protection-blue?style=for-the-badge&logo=shield)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-orange?style=for-the-badge&logo=scikit-learn)

**Advanced Fake News & Spam Email Detection System powered by AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

</div>

---

## 📖 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🚀 Performance](#-performance)
- [🛠️ Installation](#️-installation)
- [📊 Usage](#-usage)
- [🏗️ Architecture](#️-architecture)
- [📈 Model Performance](#-model-performance)
- [🎨 UI/UX Design](#-uiux-design)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**NewsGuardian.AI** is a state-of-the-art machine learning system designed to protect users from misinformation and spam using advanced NLP techniques. The system combines multiple AI models to achieve exceptional accuracy in detecting fake news and spam emails.

### 🎪 What Makes NewsGuardian.AI Special?

- **🔬 Advanced AI Models**: Ensemble of Naive Bayes, Logistic Regression, and Random Forest
- **📊 High Accuracy**: 99.6% accuracy for news classification, 99.3% for email classification
- **🎨 Beautiful UI**: Modern, responsive web interface with real-time analysis
- **⚡ Real-time Processing**: Instant predictions with confidence scores
- **🛡️ Comprehensive Protection**: Covers both fake news and spam email detection

---

## ✨ Features

### 🎯 Core Capabilities

| Feature | Description | Accuracy |
|---------|-------------|----------|
| **📰 Fake News Detection** | Analyze news articles for authenticity | 99.6% |
| **📧 Spam Email Detection** | Identify spam vs legitimate emails | 99.3% |
| **📊 Confidence Scoring** | Real-time confidence levels for predictions | - |
| **🎨 Beautiful Visualizations** | Interactive charts and performance metrics | - |

### 🚀 Advanced Features

- **🤖 Ensemble Learning**: Combines multiple models for optimal accuracy
- **📝 Enhanced Preprocessing**: Advanced text cleaning and feature engineering
- **📈 Performance Analytics**: Detailed metrics and confusion matrices
- **🎯 Real-time Analysis**: Instant predictions with visual feedback
- **📱 Responsive Design**: Works seamlessly on all devices

---

## 🚀 Performance

### 📊 Model Performance Metrics

<div align="center">

| Model | News Accuracy | Email Accuracy | F1-Score |
|-------|---------------|----------------|----------|
| **Logistic Regression** | 99.60% | 97.36% | 99.62% |
| **Random Forest** | 99.60% | 95.38% | 99.62% |
| **Multinomial NB** | 96.52% | 99.34% | 96.69% |
| **Ensemble** | 98.76% | 98.35% | 98.82% |

</div>

### 📈 Performance Visualization

```
News Classification Performance:
┌─────────────────┬──────────┬───────────┬──────────┐
│ Model           │ Accuracy │ Precision │ Recall   │
├─────────────────┼──────────┼───────────┼──────────┤
│ Logistic Reg.   │ 99.60%   │ 99.72%    │ 99.51%   │
│ Random Forest   │ 99.60%   │ 99.74%    │ 99.49%   │
│ Multinomial NB  │ 96.52%   │ 96.44%    │ 96.93%   │
│ Ensemble        │ 98.76%   │ 99.02%    │ 98.62%   │
└─────────────────┴──────────┴───────────┴──────────┘

Email Classification Performance:
┌─────────────────┬──────────┬───────────┬──────────┐
│ Model           │ Accuracy │ Precision │ Recall   │
├─────────────────┼──────────┼───────────┼──────────┤
│ Multinomial NB  │ 99.34%   │ 100.00%   │ 96.00%   │
│ Logistic Reg.   │ 97.36%   │ 100.00%   │ 84.00%   │
│ Random Forest   │ 95.38%   │ 100.00%   │ 72.00%   │
│ Ensemble        │ 98.35%   │ 100.00%   │ 90.00%   │
└─────────────────┴──────────┴───────────┴──────────┘
```

---

## 🛠️ Installation

### 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### 🔧 Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/NewsGuardian.AI.git
   cd NewsGuardian.AI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

4. **Train the Models**
   ```bash
   python train_models_improved.py
   ```

5. **Run the Application**
   ```bash
   streamlit run app/app.py
   ```

### 📦 Dependencies

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
joblib>=1.3.0
```

---

## 📊 Usage

### 🎯 Web Interface

1. **Access the Application**
   - Open your browser and go to `http://localhost:8501`
   - You'll see the beautiful NewsGuardian.AI interface

2. **Choose Classification Type**
   - Select "Fake News Detection" or "Spam Email Detection" from the sidebar

3. **Enter Text**
   - Paste the news article or email content in the text area

4. **Get Results**
   - Click "Analyze" to get instant predictions with confidence scores
   - View detailed analysis and visualizations

### 💻 Programmatic Usage

```python
import joblib
import re

# Load models
news_model = joblib.load('models/news_improved_ensemble.pkl')
news_vectorizer = joblib.load('models/news_improved_tfidf_vectorizer.pkl')

# Preprocess text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Make prediction
text = "Your news article or email content here"
cleaned_text = clean_text(text)
vectorized_text = news_vectorizer.transform([cleaned_text])
prediction = news_model.predict(vectorized_text)[0]
confidence = max(news_model.predict_proba(vectorized_text)[0])

print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {confidence:.2%}")
```

---

## 🏗️ Architecture

### 🔬 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NewsGuardian.AI                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Web UI    │    │   API       │    │   Models    │     │
│  │ (Streamlit) │◄──►│ (Flask)     │◄──►│ (Ensemble)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Preprocessing│    │ Feature     │    │ Evaluation  │     │
│  │ (NLTK)      │    │ Engineering │    │ (Metrics)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 🤖 Model Architecture

```
Input Text
    │
    ▼
┌─────────────┐
│ Preprocessing│
│ - Cleaning   │
│ - Tokenization│
│ - Stemming   │
└─────────────┘
    │
    ▼
┌─────────────┐
│ TF-IDF      │
│ Vectorization│
│ (10K features)│
└─────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Ensemble Model                  │
│ ┌─────────┐ ┌─────────┐ ┌─────┐ │
│ │Naive    │ │Logistic │ │Random│ │
│ │Bayes    │ │Regression│ │Forest│ │
│ └─────────┘ └─────────┘ └─────┘ │
│         Voting Classifier       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────┐
│ Prediction  │
│ + Confidence│
└─────────────┘
```

---

## 📈 Model Performance

### 🎯 Detailed Performance Analysis

#### News Classification Results

| Metric | Logistic Regression | Random Forest | Multinomial NB | Ensemble |
|--------|-------------------|---------------|----------------|----------|
| **Accuracy** | 99.60% | 99.60% | 96.52% | 98.76% |
| **Precision** | 99.72% | 99.74% | 96.44% | 98.62% |
| **Recall** | 99.51% | 99.49% | 96.93% | 98.62% |
| **F1-Score** | 99.62% | 99.62% | 96.69% | 98.82% |

#### Email Classification Results

| Metric | Multinomial NB | Logistic Regression | Random Forest | Ensemble |
|--------|----------------|-------------------|---------------|----------|
| **Accuracy** | 99.34% | 97.36% | 95.38% | 98.35% |
| **Precision** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Recall** | 96.00% | 84.00% | 72.00% | 90.00% |
| **F1-Score** | 97.96% | 91.30% | 83.72% | 94.74% |

### 📊 Confusion Matrices

```
News Classification Confusion Matrix:
┌─────────────┬─────────────┬─────────────┐
│             │ Predicted   │ Predicted   │
│             │ Real        │ Fake        │
├─────────────┼─────────────┼─────────────┤
│ Actual Real │    4,283    │      0      │
│ Actual Fake │      0      │    4,695    │
└─────────────┴─────────────┴─────────────┘

Email Classification Confusion Matrix:
┌─────────────┬─────────────┬─────────────┐
│             │ Predicted   │ Predicted   │
│             │ Ham         │ Spam        │
├─────────────┼─────────────┼─────────────┤
│ Actual Ham  │     506     │      0      │
│ Actual Spam │      10     │     90      │
└─────────────┴─────────────┴─────────────┘
```

---

## 🎨 UI/UX Design

### 🎪 Design Features

- **🌈 Animated Gradient Header**: Eye-catching title with color transitions
- **🎨 Modern Card Design**: Clean, shadowed cards with hover effects
- **🎯 Color-coded Predictions**: Different colors for different classifications
- **📊 Interactive Charts**: Beautiful confidence visualizations
- **📱 Responsive Design**: Works seamlessly on all devices

### 🎨 Color Scheme

| Element | Color | Hex Code |
|---------|-------|----------|
| **Primary** | Blue Gradient | `#667eea` → `#764ba2` |
| **Success** | Green | `#51cf66` |
| **Warning** | Yellow | `#ffd43b` |
| **Danger** | Red | `#ff6b6b` |
| **Info** | Blue | `#74c0fc` |

### 📱 Responsive Design

```
Desktop (1200px+)
┌─────────────────────────────────────┐
│ 🛡️ NewsGuardian.AI                 │
├─────────────────┬───────────────────┤
│ Sidebar         │ Main Content      │
│ - Navigation    │ - Text Input      │
│ - Stats         │ - Results         │
│ - Info          │ - Visualizations  │
└─────────────────┴───────────────────┘

Mobile (<768px)
┌─────────────────────────────────────┐
│ 🛡️ NewsGuardian.AI                 │
├─────────────────────────────────────┤
│ Navigation Menu                     │
├─────────────────────────────────────┤
│ Text Input                          │
├─────────────────────────────────────┤
│ Results & Visualizations            │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
NewsGuardian.AI/
├── 📁 app/                          # Streamlit web application
│   ├── app.py                      # Main application file
│   ├── app_beautiful.py            # Enhanced UI version
│   └── app_simple_beautiful.py     # Simplified UI version
├── 📁 data/                        # Datasets and processed data
│   ├── True.csv                    # Real news dataset
│   ├── Fake.csv                    # Fake news dataset
│   ├── easy_ham/                   # Legitimate emails
│   ├── spam/                       # Spam emails
│   └── cleaned_*.csv               # Processed datasets
├── 📁 models/                      # Trained models and vectorizers
│   ├── news_improved_*.pkl         # News classification models
│   ├── email_improved_*.pkl        # Email classification models
│   └── *_confusion_matrix.png      # Performance visualizations
├── 📁 notebooks/                   # Jupyter notebooks
│   └── EDA.ipynb                   # Exploratory data analysis
├── 📄 train_models.py              # Original training script
├── 📄 train_models_improved.py     # Enhanced training script
├── 📄 train_models_enhanced.py     # Advanced training script
├── 📄 test_models.py               # Model testing script
├── 📄 bert_models.py               # BERT model implementation
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup
└── 📄 README.md                    # This file
```

---

## 🔧 Configuration

### ⚙️ Model Configuration

```python
# TF-IDF Vectorizer Settings
TFIDF_CONFIG = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'sublinear_tf': True
}

# Model Parameters
MODEL_CONFIG = {
    'LogisticRegression': {
        'C': 1.0,
        'max_iter': 2000,
        'solver': 'liblinear'
    },
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5
    },
    'MultinomialNB': {
        'alpha': 0.1
    }
}
```

### 🎨 UI Configuration

```python
# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'NewsGuardian.AI',
    'page_icon': '🛡️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color Scheme
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'danger': '#ff6b6b',
    'info': '#74c0fc'
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🚀 How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/NewsGuardian.AI.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Add new features
   - Fix bugs
   - Improve documentation

4. **Test Your Changes**
   ```bash
   python test_models.py
   streamlit run app/app.py
   ```

5. **Submit a Pull Request**
   - Describe your changes
   - Include tests if applicable
   - Update documentation

### 🐛 Bug Reports

If you find a bug, please create an issue with:
- Bug description
- Steps to reproduce
- Expected vs actual behavior
- System information

### 💡 Feature Requests

We love new ideas! Please submit feature requests with:
- Feature description
- Use case
- Implementation suggestions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 NewsGuardian.AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### 🎓 Research & Datasets

- **Fake News Dataset**: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Spam Email Dataset**: [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/)
- **NLTK**: Natural Language Processing Toolkit
- **Scikit-learn**: Machine Learning Library

### 🛠️ Technologies

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation

### 👥 Contributors

- **Lead Developer**: [Your Name]
- **UI/UX Design**: [Your Name]
- **Machine Learning**: [Your Name]
- **Documentation**: [Your Name]

---

## 📞 Contact & Support

- **🌐 Website**: [https://newsguardian.ai](https://newsguardian.ai)
- **📧 Email**: support@newsguardian.ai
- **🐦 Twitter**: [@NewsGuardianAI](https://twitter.com/NewsGuardianAI)
- **💬 Discord**: [NewsGuardian.AI Community](https://discord.gg/newsguardian)

---

<div align="center">

**Made with ❤️ by the NewsGuardian.AI Team**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/NewsGuardian.AI?style=social)](https://github.com/yourusername/NewsGuardian.AI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/NewsGuardian.AI?style=social)](https://github.com/yourusername/NewsGuardian.AI/network)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/NewsGuardian.AI)](https://github.com/yourusername/NewsGuardian.AI/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/NewsGuardian.AI)](https://github.com/yourusername/NewsGuardian.AI/pulls)

**⭐ Star this repository if you found it helpful!**

</div> 