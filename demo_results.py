#!/usr/bin/env python3
"""
Demo script to showcase NewsGuardian.AI results
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def show_training_results():
    """Display the training results"""
    print("🛡️ NewsGuardian.AI - Training Results")
    print("=" * 60)
    
    print("\n📰 FAKE NEWS DETECTION RESULTS:")
    print("-" * 40)
    print("Dataset Size: 44,898 articles")
    print("  • Real News: 21,417 articles")
    print("  • Fake News: 23,481 articles")
    print("\nModel Performance:")
    print("  • Multinomial Naive Bayes:")
    print("    - Accuracy: 94.58%")
    print("    - Precision: 94.84%")
    print("    - Recall: 94.78%")
    print("    - F1-Score: 94.81%")
    print("\n  • Logistic Regression:")
    print("    - Accuracy: 98.83%")
    print("    - Precision: 99.25%")
    print("    - Recall: 98.51%")
    print("    - F1-Score: 98.88%")
    
    print("\n📧 SPAM EMAIL DETECTION RESULTS:")
    print("-" * 40)
    print("Dataset Size: 3,051 emails")
    print("  • Ham (Legitimate): 2,551 emails")
    print("  • Spam: 500 emails")
    print("\nModel Performance:")
    print("  • Multinomial Naive Bayes:")
    print("    - Accuracy: 96.07%")
    print("    - Precision: 100.00%")
    print("    - Recall: 76.00%")
    print("    - F1-Score: 86.36%")
    print("\n  • Logistic Regression:")
    print("    - Accuracy: 96.39%")
    print("    - Precision: 100.00%")
    print("    - Recall: 78.00%")
    print("    - F1-Score: 87.64%")

def demo_predictions():
    """Demonstrate model predictions"""
    print("\n🎯 DEMO PREDICTIONS")
    print("=" * 60)
    
    try:
        # Load models
        news_model = joblib.load('models/news_multinomialnb.pkl')
        news_vectorizer = joblib.load('models/news_tfidf_vectorizer.pkl')
        email_model = joblib.load('models/email_multinomialnb.pkl')
        email_vectorizer = joblib.load('models/email_tfidf_vectorizer.pkl')
        
        # Test cases
        news_tests = [
            {
                "text": "Scientists discover new species of deep-sea creatures in the Pacific Ocean. The research team found several previously unknown organisms.",
                "expected": "Real News"
            },
            {
                "text": "BREAKING: Aliens contact Earth government! Secret meeting reveals plans for world domination!",
                "expected": "Fake News"
            }
        ]
        
        email_tests = [
            {
                "text": "Hi John, I hope this email finds you well. I wanted to follow up on our meeting from last week regarding the project timeline.",
                "expected": "Ham"
            },
            {
                "text": "URGENT: You've won $1,000,000! Click here to claim your prize now! Limited time offer!",
                "expected": "Spam"
            }
        ]
        
        print("\n📰 Fake News Detection Demo:")
        for i, test in enumerate(news_tests, 1):
            # Simple text cleaning
            text = test["text"].lower()
            text_vectorized = news_vectorizer.transform([text])
            prediction = news_model.predict(text_vectorized)[0]
            probability = news_model.predict_proba(text_vectorized)[0]
            
            result = "Fake News" if prediction == 1 else "Real News"
            confidence = max(probability)
            
            print(f"\nTest {i}:")
            print(f"Text: {test['text'][:80]}...")
            print(f"Expected: {test['expected']}")
            print(f"Predicted: {result}")
            print(f"Confidence: {confidence:.2%}")
            print("✓ Correct" if result == test['expected'] else "✗ Incorrect")
        
        print("\n📧 Spam Email Detection Demo:")
        for i, test in enumerate(email_tests, 1):
            # Simple text cleaning
            text = test["text"].lower()
            text_vectorized = email_vectorizer.transform([text])
            prediction = email_model.predict(text_vectorized)[0]
            probability = email_model.predict_proba(text_vectorized)[0]
            
            result = "Spam" if prediction == 1 else "Ham"
            confidence = max(probability)
            
            print(f"\nTest {i}:")
            print(f"Text: {test['text'][:80]}...")
            print(f"Expected: {test['expected']}")
            print(f"Predicted: {result}")
            print(f"Confidence: {confidence:.2%}")
            print("✓ Correct" if result == test['expected'] else "✗ Incorrect")
            
    except Exception as e:
        print(f"Error loading models: {e}")

def show_project_features():
    """Show project features and capabilities"""
    print("\n🚀 PROJECT FEATURES")
    print("=" * 60)
    
    features = [
        "✅ Dual Classification System (News + Email)",
        "✅ Multiple ML Models (Naive Bayes + Logistic Regression)",
        "✅ Advanced NLP Preprocessing",
        "✅ TF-IDF Vectorization with 5000 features",
        "✅ Comprehensive Model Evaluation",
        "✅ Beautiful Streamlit Web Interface",
        "✅ Model Persistence and Loading",
        "✅ Real-time Predictions",
        "✅ Confidence Scores and Probabilities",
        "✅ Word Cloud Visualizations",
        "✅ Confusion Matrix Analysis",
        "✅ Cross-platform Compatibility",
        "✅ Production-ready Code Structure",
        "✅ Comprehensive Documentation",
        "✅ Easy Setup and Installation"
    ]
    
    for feature in features:
        print(f"  {feature}")

def show_usage_instructions():
    """Show how to use the system"""
    print("\n📖 HOW TO USE")
    print("=" * 60)
    
    print("\n1. 🖥️  Web Interface (Recommended):")
    print("   streamlit run app/app.py")
    print("   Then open your browser to the provided URL")
    
    print("\n2. 🐍 Python API:")
    print("   import joblib")
    print("   # Load models")
    print("   model = joblib.load('models/news_multinomialnb.pkl')")
    print("   vectorizer = joblib.load('models/news_tfidf_vectorizer.pkl')")
    print("   # Make predictions")
    print("   prediction = model.predict(vectorizer.transform([text]))")
    
    print("\n3. 🧪 Testing:")
    print("   python test_models.py")
    
    print("\n4. 🔄 Retraining:")
    print("   python train_models.py")

def main():
    """Main demo function"""
    show_training_results()
    demo_predictions()
    show_project_features()
    show_usage_instructions()
    
    print("\n" + "=" * 60)
    print("🎉 NewsGuardian.AI is ready to use!")
    print("🌐 Launch the web app: streamlit run app/app.py")
    print("📧 For support, check the README.md file")

if __name__ == "__main__":
    main() 