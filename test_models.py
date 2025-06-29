import joblib
import pandas as pd
import numpy as np
import os

def test_news_classifier():
    """Test the fake news classifier"""
    print("=== Testing Fake News Classifier ===")
    
    # Load models
    try:
        news_model = joblib.load('models/news_multinomialnb.pkl')
        news_vectorizer = joblib.load('models/news_tfidf_vectorizer.pkl')
        news_preprocessor = joblib.load('models/news_preprocessor.pkl')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "text": "Scientists discover new species of deep-sea creatures in the Pacific Ocean. The research team from the University of Marine Biology found several previously unknown organisms during their expedition.",
            "expected": "Real News"
        },
        {
            "text": "BREAKING: Aliens contact Earth government! Secret meeting reveals plans for world domination. Sources say the government is hiding the truth from citizens.",
            "expected": "Fake News"
        },
        {
            "text": "New study shows that regular exercise can improve mental health and reduce stress levels. The research involved over 10,000 participants across multiple countries.",
            "expected": "Real News"
        },
        {
            "text": "SHOCKING: One simple trick to lose 50 pounds in a week! Doctors hate this! Click here to find out the secret that big pharma doesn't want you to know!",
            "expected": "Fake News"
        }
    ]
    
    print("\n--- Test Results ---")
    for i, test_case in enumerate(test_cases, 1):
        # Preprocess text
        cleaned_text = news_preprocessor.clean_text(test_case["text"])
        vectorized_text = news_vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = news_model.predict(vectorized_text)[0]
        probability = news_model.predict_proba(vectorized_text)[0]
        
        # Determine result
        result = "Fake News" if prediction == 1 else "Real News"
        confidence = max(probability)
        
        print(f"\nTest {i}:")
        print(f"Text: {test_case['text'][:100]}...")
        print(f"Expected: {test_case['expected']}")
        print(f"Predicted: {result}")
        print(f"Confidence: {confidence:.2%}")
        print(f"✓ Correct" if result == test_case['expected'] else "✗ Incorrect")

def test_email_classifier():
    """Test the spam email classifier"""
    print("\n=== Testing Spam Email Classifier ===")
    
    # Load models
    try:
        email_model = joblib.load('models/email_multinomialnb.pkl')
        email_vectorizer = joblib.load('models/email_tfidf_vectorizer.pkl')
        email_preprocessor = joblib.load('models/email_preprocessor.pkl')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "text": "Hi John, I hope this email finds you well. I wanted to follow up on our meeting from last week regarding the project timeline. Please let me know if you have any questions.",
            "expected": "Ham"
        },
        {
            "text": "URGENT: You've won $1,000,000! Click here to claim your prize now! Limited time offer! Don't miss out on this amazing opportunity!",
            "expected": "Spam"
        },
        {
            "text": "Dear team, Please find attached the quarterly report for Q3. We have made significant progress on all key metrics. Let's discuss this in our next meeting.",
            "expected": "Ham"
        },
        {
            "text": "FREE VIAGRA! ENLARGE YOUR PENIS NOW! 100% GUARANTEED! CLICK HERE FOR AMAZING RESULTS! SPECIAL OFFER EXPIRES TODAY!",
            "expected": "Spam"
        }
    ]
    
    print("\n--- Test Results ---")
    for i, test_case in enumerate(test_cases, 1):
        # Preprocess text
        cleaned_text = email_preprocessor.clean_text(test_case["text"])
        vectorized_text = email_vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = email_model.predict(vectorized_text)[0]
        probability = email_model.predict_proba(vectorized_text)[0]
        
        # Determine result
        result = "Spam" if prediction == 1 else "Ham"
        confidence = max(probability)
        
        print(f"\nTest {i}:")
        print(f"Text: {test_case['text'][:100]}...")
        print(f"Expected: {test_case['expected']}")
        print(f"Predicted: {result}")
        print(f"Confidence: {confidence:.2%}")
        print(f"✓ Correct" if result == test_case['expected'] else "✗ Incorrect")

def test_model_performance():
    """Test model performance on sample data"""
    print("\n=== Model Performance Test ===")
    
    # Check if cleaned data exists
    if not os.path.exists('data/cleaned_news_data.csv') or not os.path.exists('data/cleaned_email_data.csv'):
        print("✗ Cleaned data not found. Please run train_models.py first.")
        return
    
    # Load cleaned data
    news_df = pd.read_csv('data/cleaned_news_data.csv')
    email_df = pd.read_csv('data/cleaned_email_data.csv')
    
    # Sample data for testing
    news_sample = news_df.sample(n=100, random_state=42)
    email_sample = email_df.sample(n=100, random_state=42)
    
    print(f"Testing on {len(news_sample)} news samples and {len(email_sample)} email samples")
    
    # Test news classifier
    try:
        news_model = joblib.load('models/news_multinomialnb.pkl')
        news_vectorizer = joblib.load('models/news_tfidf_vectorizer.pkl')
        
        X_news = news_sample['cleaned_text']
        y_news = news_sample['label']
        
        X_news_vectorized = news_vectorizer.transform(X_news)
        news_predictions = news_model.predict(X_news_vectorized)
        news_accuracy = (news_predictions == y_news).mean()
        
        print(f"News Classifier Accuracy: {news_accuracy:.2%}")
    except Exception as e:
        print(f"✗ Error testing news classifier: {e}")
    
    # Test email classifier
    try:
        email_model = joblib.load('models/email_multinomialnb.pkl')
        email_vectorizer = joblib.load('models/email_tfidf_vectorizer.pkl')
        
        X_email = email_sample['cleaned_text']
        y_email = email_sample['label']
        
        X_email_vectorized = email_vectorizer.transform(X_email)
        email_predictions = email_model.predict(X_email_vectorized)
        email_accuracy = (email_predictions == y_email).mean()
        
        print(f"Email Classifier Accuracy: {email_accuracy:.2%}")
    except Exception as e:
        print(f"✗ Error testing email classifier: {e}")

def main():
    """Main test function"""
    print("🧪 NewsGuardian.AI Model Testing")
    print("=" * 50)
    
    # Check if models exist
    if not os.path.exists('models'):
        print("✗ Models directory not found. Please run train_models.py first.")
        return
    
    # Test individual classifiers
    test_news_classifier()
    test_email_classifier()
    
    # Test performance
    test_model_performance()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")

if __name__ == "__main__":
    main() 