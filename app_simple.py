import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="NewsGuardian.AI - Fake News & Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .fake-prediction {
        background-color: #ffe6e6;
        border-color: #dc3545;
    }
    .real-prediction {
        background-color: #e6ffe6;
        border-color: #28a745;
    }
    .spam-prediction {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .ham-prediction {
        background-color: #d1ecf1;
        border-color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and preprocessors"""
    models = {}
    
    try:
        # Load news models
        models['news_nb'] = joblib.load('models/news_multinomialnb.pkl')
        models['news_lr'] = joblib.load('models/news_logisticregression.pkl')
        models['news_tfidf'] = joblib.load('models/news_tfidf_vectorizer.pkl')
        
        # Load email models
        models['email_nb'] = joblib.load('models/email_multinomialnb.pkl')
        models['email_lr'] = joblib.load('models/email_logisticregression.pkl')
        models['email_tfidf'] = joblib.load('models/email_tfidf_vectorizer.pkl')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training script first: `python train_models.py`")
        return None

def simple_text_clean(text):
    """Simple text cleaning function"""
    import re
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text, models):
    """Predict if news is fake or real"""
    # Simple text cleaning
    cleaned_text = simple_text_clean(text)
    
    # Vectorize
    text_vectorized = models['news_tfidf'].transform([cleaned_text])
    
    # Get predictions from both models
    nb_pred = models['news_nb'].predict(text_vectorized)[0]
    lr_pred = models['news_lr'].predict(text_vectorized)[0]
    
    # Get probabilities
    nb_prob = models['news_nb'].predict_proba(text_vectorized)[0]
    lr_prob = models['news_lr'].predict_proba(text_vectorized)[0]
    
    return {
        'nb_prediction': nb_pred,
        'lr_prediction': lr_pred,
        'nb_probability': nb_prob,
        'lr_probability': lr_prob
    }

def predict_email(text, models):
    """Predict if email is spam or ham"""
    # Simple text cleaning
    cleaned_text = simple_text_clean(text)
    
    # Vectorize
    text_vectorized = models['email_tfidf'].transform([cleaned_text])
    
    # Get predictions from both models
    nb_pred = models['email_nb'].predict(text_vectorized)[0]
    lr_pred = models['email_lr'].predict(text_vectorized)[0]
    
    # Get probabilities
    nb_prob = models['email_nb'].predict_proba(text_vectorized)[0]
    lr_prob = models['email_lr'].predict_proba(text_vectorized)[0]
    
    return {
        'nb_prediction': nb_pred,
        'lr_prediction': lr_pred,
        'nb_probability': nb_prob,
        'lr_probability': lr_prob
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ NewsGuardian.AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Fake News & Spam Email Detection System</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    if models is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Classification Type")
    classification_type = st.sidebar.selectbox(
        "Choose what you want to classify:",
        ["Fake News Detection", "Spam Email Detection"]
    )
    
    st.sidebar.markdown("## 📊 Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose the model to use:",
        ["Multinomial Naive Bayes", "Logistic Regression", "Both (Ensemble)"]
    )
    
    # Main content
    if classification_type == "Fake News Detection":
        st.markdown('<h2 class="sub-header">📰 Fake News Detection</h2>', unsafe_allow_html=True)
        
        # Input area
        news_text = st.text_area(
            "Enter the news article text:",
            height=200,
            placeholder="Paste the news article content here..."
        )
        
        if st.button("🔍 Analyze News", type="primary"):
            if news_text.strip():
                with st.spinner("Analyzing news article..."):
                    # Get predictions
                    results = predict_news(news_text, models)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Prediction Results")
                        
                        if model_type == "Multinomial Naive Bayes" or model_type == "Both (Ensemble)":
                            nb_pred = results['nb_prediction']
                            nb_prob = results['nb_probability']
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Multinomial Naive Bayes</h4>
                                <p><strong>Prediction:</strong> {'❌ FAKE NEWS' if nb_pred == 1 else '✅ REAL NEWS'}</p>
                                <p><strong>Confidence:</strong> {max(nb_prob):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if model_type == "Logistic Regression" or model_type == "Both (Ensemble)":
                            lr_pred = results['lr_prediction']
                            lr_prob = results['lr_probability']
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Logistic Regression</h4>
                                <p><strong>Prediction:</strong> {'❌ FAKE NEWS' if lr_pred == 1 else '✅ REAL NEWS'}</p>
                                <p><strong>Confidence:</strong> {max(lr_prob):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📈 Confidence Scores")
                        
                        # Create confidence chart
                        if model_type == "Both (Ensemble)":
                            fig = go.Figure()
                            
                            # NB probabilities
                            fig.add_trace(go.Bar(
                                name='Naive Bayes',
                                x=['Real News', 'Fake News'],
                                y=[nb_prob[0], nb_prob[1]],
                                marker_color=['#28a745', '#dc3545']
                            ))
                            
                            # LR probabilities
                            fig.add_trace(go.Bar(
                                name='Logistic Regression',
                                x=['Real News', 'Fake News'],
                                y=[lr_prob[0], lr_prob[1]],
                                marker_color=['#20c997', '#fd7e14']
                            ))
                            
                            fig.update_layout(
                                title="Model Confidence Scores",
                                yaxis_title="Probability",
                                barmode='group'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Final prediction
                    st.markdown("### 🎯 Final Prediction")
                    
                    if model_type == "Both (Ensemble)":
                        # Ensemble prediction (majority vote)
                        ensemble_pred = 1 if (results['nb_prediction'] + results['lr_prediction']) >= 1 else 0
                        ensemble_prob = (max(results['nb_probability']) + max(results['lr_probability'])) / 2
                        
                        prediction_class = "fake-prediction" if ensemble_pred == 1 else "real-prediction"
                        prediction_text = "❌ FAKE NEWS" if ensemble_pred == 1 else "✅ REAL NEWS"
                        
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            <h3 style="text-align: center; margin-bottom: 1rem;">{prediction_text}</h3>
                            <p style="text-align: center; font-size: 1.1rem;"><strong>Ensemble Confidence:</strong> {ensemble_prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Single model prediction
                        if model_type == "Multinomial Naive Bayes":
                            pred = results['nb_prediction']
                            prob = max(results['nb_probability'])
                        else:
                            pred = results['lr_prediction']
                            prob = max(results['lr_probability'])
                        
                        prediction_class = "fake-prediction" if pred == 1 else "real-prediction"
                        prediction_text = "❌ FAKE NEWS" if pred == 1 else "✅ REAL NEWS"
                        
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            <h3 style="text-align: center; margin-bottom: 1rem;">{prediction_text}</h3>
                            <p style="text-align: center; font-size: 1.1rem;"><strong>Confidence:</strong> {prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Text analysis
                    st.markdown("### 📝 Text Analysis")
                    st.info(f"Text length: {len(news_text)} characters")
                    st.info(f"Word count: {len(news_text.split())} words")
                    
            else:
                st.warning("Please enter some text to analyze.")
    
    else:  # Spam Email Detection
        st.markdown('<h2 class="sub-header">📧 Spam Email Detection</h2>', unsafe_allow_html=True)
        
        # Input area
        email_text = st.text_area(
            "Enter the email content:",
            height=200,
            placeholder="Paste the email content here..."
        )
        
        if st.button("🔍 Analyze Email", type="primary"):
            if email_text.strip():
                with st.spinner("Analyzing email content..."):
                    # Get predictions
                    results = predict_email(email_text, models)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Prediction Results")
                        
                        if model_type == "Multinomial Naive Bayes" or model_type == "Both (Ensemble)":
                            nb_pred = results['nb_prediction']
                            nb_prob = results['nb_probability']
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Multinomial Naive Bayes</h4>
                                <p><strong>Prediction:</strong> {'🚨 SPAM' if nb_pred == 1 else '✅ HAM'}</p>
                                <p><strong>Confidence:</strong> {max(nb_prob):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if model_type == "Logistic Regression" or model_type == "Both (Ensemble)":
                            lr_pred = results['lr_prediction']
                            lr_prob = results['lr_probability']
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Logistic Regression</h4>
                                <p><strong>Prediction:</strong> {'🚨 SPAM' if lr_pred == 1 else '✅ HAM'}</p>
                                <p><strong>Confidence:</strong> {max(lr_prob):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📈 Confidence Scores")
                        
                        # Create confidence chart
                        if model_type == "Both (Ensemble)":
                            fig = go.Figure()
                            
                            # NB probabilities
                            fig.add_trace(go.Bar(
                                name='Naive Bayes',
                                x=['Ham', 'Spam'],
                                y=[nb_prob[0], nb_prob[1]],
                                marker_color=['#17a2b8', '#ffc107']
                            ))
                            
                            # LR probabilities
                            fig.add_trace(go.Bar(
                                name='Logistic Regression',
                                x=['Ham', 'Spam'],
                                y=[lr_prob[0], lr_prob[1]],
                                marker_color=['#6f42c1', '#fd7e14']
                            ))
                            
                            fig.update_layout(
                                title="Model Confidence Scores",
                                yaxis_title="Probability",
                                barmode='group'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Final prediction
                    st.markdown("### 🎯 Final Prediction")
                    
                    if model_type == "Both (Ensemble)":
                        # Ensemble prediction (majority vote)
                        ensemble_pred = 1 if (results['nb_prediction'] + results['lr_prediction']) >= 1 else 0
                        ensemble_prob = (max(results['nb_probability']) + max(results['lr_probability'])) / 2
                        
                        prediction_class = "spam-prediction" if ensemble_pred == 1 else "ham-prediction"
                        prediction_text = "🚨 SPAM EMAIL" if ensemble_pred == 1 else "✅ HAM EMAIL"
                        
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            <h3 style="text-align: center; margin-bottom: 1rem;">{prediction_text}</h3>
                            <p style="text-align: center; font-size: 1.1rem;"><strong>Ensemble Confidence:</strong> {ensemble_prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Single model prediction
                        if model_type == "Multinomial Naive Bayes":
                            pred = results['nb_prediction']
                            prob = max(results['nb_probability'])
                        else:
                            pred = results['lr_prediction']
                            prob = max(results['lr_probability'])
                        
                        prediction_class = "spam-prediction" if pred == 1 else "ham-prediction"
                        prediction_text = "🚨 SPAM EMAIL" if pred == 1 else "✅ HAM EMAIL"
                        
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            <h3 style="text-align: center; margin-bottom: 1rem;">{prediction_text}</h3>
                            <p style="text-align: center; font-size: 1.1rem;"><strong>Confidence:</strong> {prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Text analysis
                    st.markdown("### 📝 Text Analysis")
                    st.info(f"Text length: {len(email_text)} characters")
                    st.info(f"Word count: {len(email_text.split())} words")
                    
            else:
                st.warning("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🛡️ <strong>NewsGuardian.AI</strong> - Protecting you from misinformation and spam</p>
        <p>Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 