import os
import sys
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class SpamPredictor:
    def __init__(self):
        self.model_path = os.path.join(project_root, 'model', 'model.pkl')
        self.vectorizer_path = os.path.join(project_root, 'model', 'vectorizer.pkl')
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def load_model_and_vectorizer(self):
        """Load the saved model and vectorizer"""
        try:
            print("Loading model and vectorizer...")
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("Model and vectorizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model or vectorizer: {e}")
            sys.exit(1)
    
    def preprocess_text(self, text):
        """Preprocess text using the same steps as in training"""
        try:
            # Handle null/empty input
            if pd.isna(text) or not str(text).strip():
                return ""
            
            # Step 1: Convert to lowercase
            text = str(text).lower()
            
            # Step 2: Word tokenize
            tokens = word_tokenize(text)
            
            # Step 3: Remove punctuation using isalnum() for each word
            cleaned_tokens = []
            for word in tokens:
                cleaned_word = ''.join(char for char in word if char.isalnum())
                if cleaned_word:
                    cleaned_tokens.append(cleaned_word)
            
            if not cleaned_tokens:
                return ""
            
            # Step 4: Remove stopwords
            cleaned_tokens = [word for word in cleaned_tokens if word not in self.stop_words]
            
            if not cleaned_tokens:
                return ""
            
            # Step 5: Apply stemming
            stemmed_tokens = [self.stemmer.stem(word) for word in cleaned_tokens]
            
            # Step 6: Join words back into text
            preprocessed_text = ' '.join(stemmed_tokens)
            
            return preprocessed_text if preprocessed_text.strip() else ""
            
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return ""
    
    def predict(self, text):
        """Make prediction for a given text"""
        try:
            # Preprocess the input text
            preprocessed_text = self.preprocess_text(text)
            
            # Transform the text using the vectorizer
            text_vectorized = self.vectorizer.transform([preprocessed_text])
            
            # Make prediction
            prediction = self.model.predict(text_vectorized.toarray())[0]
            prediction_proba = self.model.predict_proba(text_vectorized.toarray())[0]
            
            # Get the probability of spam
            spam_probability = prediction_proba[1]
            
            return {
                'prediction': 'SPAM' if prediction == 1 else 'HAM',
                'spam_probability': round(spam_probability * 100, 2),
                'ham_probability': round(prediction_proba[0] * 100, 2)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

def main():
    # Sample input data
    sample_texts = [
        "Congratulations! You've won a free iPhone! Click here to claim your prize!",
        "Kindly put your Bank Details in the message",
        "Hey, can you send me the meeting notes from yesterday?",
        "URGENT: Your account has been suspended. Click here to verify your identity!",
        "Thanks for your email. I'll get back to you soon.",
        "FREE VIAGRA NOW!!! Click here to get your supply!"
    ]
    
    # Initialize predictor
    predictor = SpamPredictor()
    predictor.load_model_and_vectorizer()
    
    # Test predictions
    print("\nTesting model with sample texts:")
    print("-" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}:")
        print(f"Text: {text}")
        result = predictor.predict(text)
        
        if result:
            print(f"Prediction: {result['prediction']}")
            print(f"Spam Probability: {result['spam_probability']}%")
            print(f"Ham Probability: {result['ham_probability']}%")
        else:
            print("Failed to make prediction")
        
        print("-" * 80)

if __name__ == "__main__":
    main() 