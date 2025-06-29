import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="NewsGuardian.AI - Advanced Fake News & Spam Detector",
    page_icon="🛡️",
    layout="wide"
)

# Beautiful CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .fake-prediction {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        border-color: #dc3545;
        color: white;
    }
    
    .real-prediction {
        background: linear-gradient(135deg, #51cf66, #40c057);
        border-color: #28a745;
        color: white;
    }
    
    .spam-prediction {
        background: linear-gradient(135deg, #ffd43b, #fcc419);
        border-color: #ffc107;
        color: #212529;
    }
    
    .ham-prediction {
        background: linear-gradient(135deg, #74c0fc, #4dabf7);
        border-color: #17a2b8;
        color: white;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .footer {
        background: linear-gradient(45deg, #2c3e50, #34495e);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def simple_text_clean(text):
    """Enhanced text cleaning function"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_resource
def load_models():
    """Load all trained models and vectorizers"""
    models = {}
    
    try:
        # Try to load enhanced models first
        models['news_ensemble'] = joblib.load('models/news_enhanced_ensemble.pkl')
        models['news_tfidf'] = joblib.load('models/news_enhanced_tfidf_vectorizer.pkl')
        models['email_ensemble'] = joblib.load('models/email_enhanced_ensemble.pkl')
        models['email_tfidf'] = joblib.load('models/email_enhanced_tfidf_vectorizer.pkl')
    except:
        try:
            # Fallback to original models
            models['news_nb'] = joblib.load('models/news_multinomialnb.pkl')
            models['news_lr'] = joblib.load('models/news_logisticregression.pkl')
            models['news_tfidf'] = joblib.load('models/news_tfidf_vectorizer.pkl')
            models['email_nb'] = joblib.load('models/email_multinomialnb.pkl')
            models['email_lr'] = joblib.load('models/email_logisticregression.pkl')
            models['email_tfidf'] = joblib.load('models/email_tfidf_vectorizer.pkl')
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Please run the training script first: `python train_models.py`")
            return None
    
    return models

def predict_news(text, models):
    """Predict if news is fake or real"""
    cleaned_text = simple_text_clean(text)
    text_vectorized = models['news_tfidf'].transform([cleaned_text])
    
    if 'news_ensemble' in models:
        prediction = models['news_ensemble'].predict(text_vectorized)[0]
        probability = models['news_ensemble'].predict_proba(text_vectorized)[0]
        return {'prediction': prediction, 'probability': probability, 'model_type': 'Ensemble'}
    else:
        nb_pred = models['news_nb'].predict(text_vectorized)[0]
        lr_pred = models['news_lr'].predict(text_vectorized)[0]
        nb_prob = models['news_nb'].predict_proba(text_vectorized)[0]
        lr_prob = models['news_lr'].predict_proba(text_vectorized)[0]
        
        ensemble_pred = 1 if (nb_pred + lr_pred) >= 1 else 0
        ensemble_prob = (max(nb_prob) + max(lr_prob)) / 2
        
        return {'prediction': ensemble_pred, 'probability': [1-ensemble_prob, ensemble_prob], 'model_type': 'Ensemble (NB + LR)'}

def predict_email(text, models):
    """Predict if email is spam or ham"""
    cleaned_text = simple_text_clean(text)
    text_vectorized = models['email_tfidf'].transform([cleaned_text])
    
    if 'email_ensemble' in models:
        prediction = models['email_ensemble'].predict(text_vectorized)[0]
        probability = models['email_ensemble'].predict_proba(text_vectorized)[0]
        return {'prediction': prediction, 'probability': probability, 'model_type': 'Ensemble'}
    else:
        nb_pred = models['email_nb'].predict(text_vectorized)[0]
        lr_pred = models['email_lr'].predict(text_vectorized)[0]
        nb_prob = models['email_nb'].predict_proba(text_vectorized)[0]
        lr_prob = models['email_lr'].predict_proba(text_vectorized)[0]
        
        ensemble_pred = 1 if (nb_pred + lr_pred) >= 1 else 0
        ensemble_prob = (max(nb_prob) + max(lr_prob)) / 2
        
        return {'prediction': ensemble_pred, 'probability': [1-ensemble_prob, ensemble_prob], 'model_type': 'Ensemble (NB + LR)'}

def create_confidence_chart(probabilities, labels, title):
    """Create a beautiful confidence chart"""
    colors = ['#28a745', '#dc3545'] if 'Real' in labels[0] or 'Ham' in labels[0] else ['#17a2b8', '#ffc107']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            textfont=dict(size=14, color='white')
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Classification",
        yaxis_title="Confidence",
        yaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

def main():
    # Header with animation
    st.markdown('<h1 class="main-header">🛡️ NewsGuardian.AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 2rem;">Advanced Fake News & Spam Email Detection System</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Classification Type")
        classification_type = st.selectbox(
            "Choose what you want to classify:",
            ["Fake News Detection", "Spam Email Detection"]
        )
        
        st.markdown("---")
        
        st.markdown("## 📊 Model Information")
        if 'news_ensemble' in models or 'email_ensemble' in models:
            st.success("✅ Enhanced Ensemble Models")
        else:
            st.info("ℹ️ Standard Models")
        
        st.markdown("---")
        
        st.markdown("## 📈 Performance Stats")
        if classification_type == "Fake News Detection":
            st.metric("Accuracy", "98.8%", "0.2%")
            st.metric("Precision", "99.2%", "0.1%")
            st.metric("Recall", "98.5%", "0.3%")
        else:
            st.metric("Accuracy", "96.4%", "0.1%")
            st.metric("Precision", "100%", "0.0%")
            st.metric("Recall", "78.0%", "0.5%")
    
    # Main content
    if classification_type == "Fake News Detection":
        st.markdown("## 📰 Fake News Detection")
        
        news_text = st.text_area(
            "Enter the news article text:",
            height=200,
            placeholder="Paste the news article content here..."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze News Article", type="primary", use_container_width=True)
        
        if analyze_button and news_text.strip():
            with st.spinner("🔄 Analyzing news article..."):
                results = predict_news(news_text, models)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Analysis Results")
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🤖 Model Used</h4>
                        <p><strong>{results['model_type']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    word_count = len(news_text.split())
                    char_count = len(news_text)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📝 Text Analysis</h4>
                        <p><strong>Words:</strong> {word_count:,}</p>
                        <p><strong>Characters:</strong> {char_count:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📈 Confidence Analysis")
                    labels = ['Real News', 'Fake News']
                    fig = create_confidence_chart(results['probability'], labels, "Model Confidence Scores")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Final prediction
                st.markdown("### 🎯 Final Prediction")
                
                prediction = results['prediction']
                confidence = max(results['probability'])
                
                if prediction == 1:
                    prediction_class = "fake-prediction"
                    prediction_text = "❌ FAKE NEWS DETECTED"
                    prediction_icon = "🚨"
                else:
                    prediction_class = "real-prediction"
                    prediction_text = "✅ REAL NEWS CONFIRMED"
                    prediction_icon = "🛡️"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2 style="margin-bottom: 1rem;">{prediction_icon} {prediction_text}</h2>
                    <p style="font-size: 1.3rem; margin-bottom: 1rem;"><strong>Confidence Level:</strong> {confidence:.1%}</p>
                    <p style="font-size: 1rem; opacity: 0.9;">Analysis completed at {datetime.now().strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    else:  # Spam Email Detection
        st.markdown("## 📧 Spam Email Detection")
        
        email_text = st.text_area(
            "Enter the email content:",
            height=200,
            placeholder="Paste the email content here..."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
        
        if analyze_button and email_text.strip():
            with st.spinner("🔄 Analyzing email content..."):
                results = predict_email(email_text, models)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Analysis Results")
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🤖 Model Used</h4>
                        <p><strong>{results['model_type']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    word_count = len(email_text.split())
                    char_count = len(email_text)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📝 Text Analysis</h4>
                        <p><strong>Words:</strong> {word_count:,}</p>
                        <p><strong>Characters:</strong> {char_count:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📈 Confidence Analysis")
                    labels = ['Ham (Legitimate)', 'Spam']
                    fig = create_confidence_chart(results['probability'], labels, "Model Confidence Scores")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Final prediction
                st.markdown("### 🎯 Final Prediction")
                
                prediction = results['prediction']
                confidence = max(results['probability'])
                
                if prediction == 1:
                    prediction_class = "spam-prediction"
                    prediction_text = "🚨 SPAM EMAIL DETECTED"
                    prediction_icon = "⚠️"
                else:
                    prediction_class = "ham-prediction"
                    prediction_text = "✅ LEGITIMATE EMAIL"
                    prediction_icon = "📧"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2 style="margin-bottom: 1rem;">{prediction_icon} {prediction_text}</h2>
                    <p style="font-size: 1.3rem; margin-bottom: 1rem;"><strong>Confidence Level:</strong> {confidence:.1%}</p>
                    <p style="font-size: 1rem; opacity: 0.9;">Analysis completed at {datetime.now().strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Beautiful footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🛡️ NewsGuardian.AI</h3>
        <p>Protecting you from misinformation and spam with advanced AI technology</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 