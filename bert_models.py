import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_bert_model(X_train, X_test, y_train, y_test, dataset_name, epochs=3, batch_size=16):
    """Train BERT model for text classification"""
    print(f"\n=== Training BERT model for {dataset_name} ===")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertClassifier()
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Setup optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        test_accuracies.append(accuracy)
        
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'models/{dataset_name}_bert_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    # Save tokenizer
    tokenizer.save_pretrained(f'models/{dataset_name}_bert_tokenizer')
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'models/{dataset_name}_bert_model.pth'))
    model.eval()
    
    # Final evaluation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Final metrics
    final_accuracy = accuracy_score(all_labels, all_predictions)
    final_precision = precision_score(all_labels, all_predictions)
    final_recall = recall_score(all_labels, all_predictions)
    final_f1 = f1_score(all_labels, all_predictions)
    
    print(f"\n=== Final BERT Results for {dataset_name} ===")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    print(f"F1-Score: {final_f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Real/Ham', 'Fake/Spam']))
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'f1_score': final_f1,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }

def predict_with_bert(text, model, tokenizer, device):
    """Make prediction using trained BERT model"""
    model.eval()
    
    # Tokenize input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
    
    return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]

def main():
    """Main function to train BERT models"""
    print("=== BERT-based Fake News and Spam Email Classifier ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if cleaned data exists
    if not os.path.exists('data/cleaned_news_data.csv') or not os.path.exists('data/cleaned_email_data.csv'):
        print("Cleaned data not found. Please run train_models.py first to prepare the data.")
        return
    
    # Load cleaned data
    print("Loading cleaned data...")
    news_df = pd.read_csv('data/cleaned_news_data.csv')
    email_df = pd.read_csv('data/cleaned_email_data.csv')
    
    # Prepare news data for BERT
    print("Preparing news data for BERT training...")
    news_df = news_df[news_df['cleaned_text'].str.len() > 50]  # Filter out very short texts
    
    X_news = news_df['cleaned_text'].values
    y_news = news_df['label'].values
    
    # Split news data
    X_news_train, X_news_test, y_news_train, y_news_test = train_test_split(
        X_news, y_news, test_size=0.2, random_state=42, stratify=y_news
    )
    
    # Prepare email data for BERT
    print("Preparing email data for BERT training...")
    email_df = email_df[email_df['cleaned_text'].str.len() > 50]  # Filter out very short texts
    
    X_email = email_df['cleaned_text'].values
    y_email = email_df['label'].values
    
    # Split email data
    X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(
        X_email, y_email, test_size=0.2, random_state=42, stratify=y_email
    )
    
    # Train BERT models
    print("Training BERT models...")
    
    # Train news BERT model
    news_bert_results = train_bert_model(
        X_news_train, X_news_test, y_news_train, y_news_test, 'news', epochs=2
    )
    
    # Train email BERT model
    email_bert_results = train_bert_model(
        X_email_train, X_email_test, y_email_train, y_email_test, 'email', epochs=2
    )
    
    print("\n=== BERT Training Complete ===")
    print("BERT models saved in 'models/' directory")
    
    # Print summary
    print("\n=== BERT Model Performance Summary ===")
    print(f"News Classification BERT: Accuracy={news_bert_results['accuracy']:.4f}, F1={news_bert_results['f1_score']:.4f}")
    print(f"Email Classification BERT: Accuracy={email_bert_results['accuracy']:.4f}, F1={email_bert_results['f1_score']:.4f}")

if __name__ == "__main__":
    main() 