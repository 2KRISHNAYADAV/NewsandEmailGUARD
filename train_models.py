import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

class EmailPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def extract_email_content(self, file_path):
        """Extract email content from file, removing headers"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split by lines and find where body starts
            lines = content.split('\n')
            body_start = 0
            
            for i, line in enumerate(lines):
                if line.strip() == '':
                    body_start = i + 1
                    break
            
            # Extract body content
            body = '\n'.join(lines[body_start:])
            return body
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and preprocess email text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)

def load_news_data():
    """Load and prepare news data"""
    print("Loading news data...")
    
    # Load CSV files from data directory
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    
    # Add labels
    true_df['label'] = 0  # Real news
    fake_df['label'] = 1  # Fake news
    
    # Combine datasets
    news_df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Combine title and text
    news_df['combined_text'] = news_df['title'] + ' ' + news_df['text']
    
    print(f"News dataset shape: {news_df.shape}")
    print(f"Real news: {len(true_df)}, Fake news: {len(fake_df)}")
    
    return news_df

def load_email_data():
    """Load and prepare email data"""
    print("Loading email data...")
    
    email_data = []
    
    # Load spam emails from data directory
    spam_dir = 'data/spam'
    for filename in os.listdir(spam_dir):
        if filename.endswith('.txt') or '.' in filename:
            file_path = os.path.join(spam_dir, filename)
            content = EmailPreprocessor().extract_email_content(file_path)
            if content.strip():
                email_data.append({
                    'content': content,
                    'label': 1  # Spam
                })
    
    # Load ham emails from data directory
    ham_dir = 'data/easy_ham'
    for filename in os.listdir(ham_dir):
        if filename.endswith('.txt') or '.' in filename:
            file_path = os.path.join(ham_dir, filename)
            content = EmailPreprocessor().extract_email_content(file_path)
            if content.strip():
                email_data.append({
                    'content': content,
                    'label': 0  # Ham
                })
    
    email_df = pd.DataFrame(email_data)
    
    print(f"Email dataset shape: {email_df.shape}")
    print(f"Spam emails: {len(email_df[email_df['label'] == 1])}")
    print(f"Ham emails: {len(email_df[email_df['label'] == 0])}")
    
    return email_df

def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name):
    """Train and evaluate models for a given dataset"""
    print(f"\n=== Training models for {dataset_name} ===")
    
    # TF-IDF Vectorization
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save vectorizer
    joblib.dump(tfidf, f'models/{dataset_name}_tfidf_vectorizer.pkl')
    
    models = {
        'MultinomialNB': MultinomialNB(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Save model
        joblib.dump(model, f'models/{dataset_name}_{name.lower()}.pkl')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real/Ham', 'Fake/Spam'], 
                   yticklabels=['Real/Ham', 'Fake/Spam'])
        plt.title(f'Confusion Matrix - {name} ({dataset_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'models/{dataset_name}_{name.lower()}_confusion_matrix.png')
        plt.close()
        
        # Classification Report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['Real/Ham', 'Fake/Spam']))
    
    return results, tfidf

def main():
    """Main function to run the entire pipeline"""
    print("=== NLP Fake News and Spam Email Classifier ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # 1. Load and preprocess news data
    news_df = load_news_data()
    news_preprocessor = TextPreprocessor()
    news_df['cleaned_text'] = news_df['combined_text'].apply(news_preprocessor.clean_text)
    
    # Remove empty texts
    news_df = news_df[news_df['cleaned_text'].str.len() > 10]
    
    # Split news data
    X_news = news_df['cleaned_text']
    y_news = news_df['label']
    X_news_train, X_news_test, y_news_train, y_news_test = train_test_split(
        X_news, y_news, test_size=0.2, random_state=42, stratify=y_news
    )
    
    # 2. Load and preprocess email data
    email_df = load_email_data()
    email_preprocessor = EmailPreprocessor()
    email_df['cleaned_text'] = email_df['content'].apply(email_preprocessor.clean_text)
    
    # Remove empty texts
    email_df = email_df[email_df['cleaned_text'].str.len() > 10]
    
    # Split email data
    X_email = email_df['cleaned_text']
    y_email = email_df['label']
    X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(
        X_email, y_email, test_size=0.2, random_state=42, stratify=y_email
    )
    
    # 3. Train and evaluate news models
    news_results, news_tfidf = train_and_evaluate_models(
        X_news_train, X_news_test, y_news_train, y_news_test, 'news'
    )
    
    # 4. Train and evaluate email models
    email_results, email_tfidf = train_and_evaluate_models(
        X_email_train, X_email_test, y_email_train, y_email_test, 'email'
    )
    
    # 5. Save preprocessors
    joblib.dump(news_preprocessor, 'models/news_preprocessor.pkl')
    joblib.dump(email_preprocessor, 'models/email_preprocessor.pkl')
    
    # 6. Save cleaned data
    news_df.to_csv('data/cleaned_news_data.csv', index=False)
    email_df.to_csv('data/cleaned_email_data.csv', index=False)
    
    print("\n=== Training Complete ===")
    print("Models saved in 'models/' directory")
    print("Cleaned data saved in 'data/' directory")
    
    # Print summary
    print("\n=== Model Performance Summary ===")
    print("News Classification:")
    for name, results in news_results.items():
        print(f"  {name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
    
    print("\nEmail Classification:")
    for name, results in email_results.items():
        print(f"  {name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")

if __name__ == "__main__":
    main() 