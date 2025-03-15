import os
import sys
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import from src
from src.logger import logger
from src.exception import CustomException

class DataPreprocessor:
    def __init__(self, cleaned_data_path=None, processed_data_path=None):
        try:
            # Set default paths if none provided
            if cleaned_data_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.cleaned_data_path = os.path.join(project_root, "data", "preprocessed", "cleaned.csv")
            else:
                self.cleaned_data_path = cleaned_data_path
                
            if processed_data_path is None:
                # Save processed data in the processed folder
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.processed_data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
            else:
                self.processed_data_path = processed_data_path
                
            logger.info(f"[DATA PREPROCESSING] Initialized DataPreprocessor")
            logger.info(f"[DATA PREPROCESSING] Will load cleaned data from: {self.cleaned_data_path}")
            logger.info(f"[DATA PREPROCESSING] Will save processed data to: {self.processed_data_path}")
            
            # Download NLTK resources if not already present
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("[DATA PREPROCESSING] Downloading NLTK resources...")
                nltk.download('punkt')
                nltk.download('stopwords')
                
            # Initialize stemmer
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def read_cleaned_data(self):
        """
        Read the cleaned SMS dataset
        """
        try:
            logger.info(f"[DATA PREPROCESSING] Reading cleaned data from {self.cleaned_data_path}")
            df = pd.read_csv(self.cleaned_data_path, encoding="latin1")
            logger.info(f"[DATA PREPROCESSING] Cleaned data read successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[DATA PREPROCESSING] Error reading cleaned data: {e}")
            raise CustomException(e, sys)
    
    def rename_and_encode_columns(self, df):
        """
        Rename columns and encode the target column:
        1. Rename 'v1' to 'target'
        2. Rename 'v2' to 'message'
        3. Encode 'target' to binary values (0 for ham, 1 for spam)
        """
        try:
            logger.info(f"[DATA PREPROCESSING] Renaming and encoding columns")
            
            # Create a copy to avoid modifying the original dataframe
            processed_df = df.copy()
            
            # Rename columns
            processed_df = processed_df.rename(columns={'v1': 'target', 'v2': 'message'})
            logger.info(f"[DATA PREPROCESSING] Renamed columns: v1 -> target, v2 -> message")
            
            # Encode the target column (ham=0, spam=1)
            processed_df['target'] = processed_df['target'].map({'ham': 0, 'spam': 1})
            logger.info(f"[DATA PREPROCESSING] Encoded target column: ham -> 0, spam -> 1")
            
            # Check if encoding was successful
            logger.info(f"[DATA PREPROCESSING] Target value counts: {processed_df['target'].value_counts().to_dict()}")
            
            return processed_df
        except Exception as e:
            logger.error(f"[DATA PREPROCESSING] Error renaming and encoding columns: {e}")
            raise CustomException(e, sys)
    
    def add_feature_columns(self, df):
        """
        Add new feature columns:
        1. Text length
        2. Word count
        3. Sentence count
        """
        try:
            logger.info(f"[DATA PREPROCESSING] Adding new feature columns")
            
            # Add text length column
            df['text_length'] = df['message'].apply(len)
            logger.info(f"[DATA PREPROCESSING] Added text_length column")
            
            # Add word count column
            df['word_count'] = df['message'].apply(lambda x: len(word_tokenize(str(x))))
            logger.info(f"[DATA PREPROCESSING] Added word_count column")
            
            # Add sentence count column
            df['sentence_count'] = df['message'].apply(lambda x: len(sent_tokenize(str(x))))
            logger.info(f"[DATA PREPROCESSING] Added sentence_count column")
            
            logger.info(f"[DATA PREPROCESSING] Feature columns added. New shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[DATA PREPROCESSING] Error adding feature columns: {e}")
            raise CustomException(e, sys)
    
    def preprocess_text(self, text):
        """
        Preprocess text by applying multiple transformations
        """
        try:
            # Handle null/empty input
            if pd.isna(text) or not str(text).strip():
                return ""
                
            # Convert to string (in case it's not already)
            text = str(text).lower()
            
            # Tokenize
            text = nltk.word_tokenize(text)
            
            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)
            
            if not y:
                return ""
                
            text = y[:]
            y.clear()

            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)
            
            if not y:
                return ""
                
            text = y[:]
            y.clear()

            for i in text:
                y.append(self.stemmer.stem(i))
            
            # Return joined text or empty string if no tokens left
            return " ".join(y) if y else ""
            
        except Exception as e:
            logger.error(f"[FEATURE ENGINEERING] Error in text transformation: {e}")
            # Return empty string instead of trying to return 'text'
            return ""
    
    def transform_text(self, df):
        """
        Create transformed_text column with preprocessed text
        """
        try:
            logger.info(f"[DATA PREPROCESSING] Creating transformed_text column")
            print(f"Creating transformed_text column for {len(df)} rows...")
            
            # Create a copy to avoid modifying the original
            df_processed = df.copy()
            
            # Process in batches to show progress
            total_rows = len(df)
            batch_size = 500  # Process 500 rows at a time
            num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
            
            logger.info(f"[DATA PREPROCESSING] Processing {total_rows} rows in {num_batches} batches")
            print(f"Processing text transformation in {num_batches} batches...")
            
            transformed_texts = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_rows)
                current_batch = df['message'].iloc[start_idx:end_idx]
                
                # Process each text in the batch
                batch_transformed = []
                for text in current_batch:
                    transformed = self.preprocess_text(text)
                    batch_transformed.append(transformed)
                
                transformed_texts.extend(batch_transformed)
                
                # Report progress
                progress = (i + 1) / num_batches * 100
                logger.info(f"[DATA PREPROCESSING] Batch {i+1}/{num_batches} completed ({progress:.1f}%)")
                print(f"Batch {i+1}/{num_batches} completed ({progress:.1f}%)")
            
            # Add the transformed texts as a new column
            df_processed['transformed_text'] = transformed_texts
            
            logger.info(f"[DATA PREPROCESSING] Text transformation completed")
            print("Text transformation completed!")
            return df_processed
        except Exception as e:
            logger.error(f"[DATA PREPROCESSING] Error transforming text: {e}")
            print(f"Error transforming text: {e}")
            raise CustomException(e, sys)
    
    def save_processed_data(self, df):
        """
        Save the processed dataframe to CSV
        """
        try:
            # Ensure the directory exists
            data_dir = os.path.dirname(self.processed_data_path)
            if not os.path.exists(data_dir):
                logger.info(f"[DATA PREPROCESSING] Creating directory: {data_dir}")
                os.makedirs(data_dir, exist_ok=True)
            
            # Save the dataframe
            df.to_csv(self.processed_data_path, index=False)
            logger.info(f"[DATA PREPROCESSING] Processed data saved to {self.processed_data_path}")
            return self.processed_data_path
        except Exception as e:
            logger.error(f"[DATA PREPROCESSING] Error saving processed data: {e}")
            raise CustomException(e, sys)
    
    def preprocess(self):
        """
        Execute the complete preprocessing pipeline in the same order as the notebook
        """
        try:
            # Step 1: Read the cleaned data
            logger.info("[DATA PREPROCESSING] Starting preprocessing pipeline")
            df = self.read_cleaned_data()
            
            # Step 2: Rename and encode columns
            logger.info("[DATA PREPROCESSING] Renaming columns and encoding target")
            df = self.rename_and_encode_columns(df)
            
            # Step 3: Add feature columns
            logger.info("[DATA PREPROCESSING] Adding feature columns")
            df = self.add_feature_columns(df)
            
            # Step 4: Create transformed_text column
            logger.info("[DATA PREPROCESSING] Transforming text")
            df = self.transform_text(df)
            
            # Step 5: Save the processed data
            logger.info("[DATA PREPROCESSING] Saving processed data")
            processed_data_path = self.save_processed_data(df)
            
            logger.info("[DATA PREPROCESSING] Preprocessing pipeline completed successfully")
            return processed_data_path
        except Exception as e:
            logger.error(f"[DATA PREPROCESSING] Error in preprocessing pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test data preprocessing
    try:
        import time
        print("Starting Data Preprocessing...", flush=True)
        time.sleep(1)  # Give time for print to flush
        
        preprocessor = DataPreprocessor()
        
        # Read the cleaned data
        print("Reading cleaned data...", flush=True)
        time.sleep(1)
        
        print(f"Cleaned data path: {preprocessor.cleaned_data_path}", flush=True)
        if not os.path.exists(preprocessor.cleaned_data_path):
            print(f"ERROR: Cleaned data file not found at {preprocessor.cleaned_data_path}", flush=True)
            sys.exit(1)
            
        df = preprocessor.read_cleaned_data()
        print(f"Cleaned data shape: {df.shape}", flush=True)
        print("\nSample of cleaned data (first 5 rows):", flush=True)
        print(df.head(5), flush=True)
        
        # Rename columns and encode target
        print("\nRenaming columns and encoding target...", flush=True)
        renamed_df = preprocessor.rename_and_encode_columns(df)
        print("\nSample after renaming and encoding (first 5 rows):", flush=True)
        print(renamed_df.head(5), flush=True)
        
        # Add feature columns
        print("\nAdding feature columns...", flush=True)
        df_with_features = preprocessor.add_feature_columns(renamed_df)
        print("\nSample with features (first 5 rows):", flush=True)
        print(df_with_features[['target', 'message', 'text_length', 'word_count', 'sentence_count']].head(5), flush=True)
        
        # Transform text
        print("\nTransforming text...", flush=True)
        processed_df = preprocessor.transform_text(df_with_features)
        print("\nSample with transformed text (first 5 rows):", flush=True)
        print(processed_df[['target', 'message', 'transformed_text']].head(5), flush=True)
        
        # Save the processed dataset
        print("\nSaving processed data...", flush=True)
        processed_data_path = preprocessor.save_processed_data(processed_df)
        print(f"\nProcessed dataset saved to: {processed_data_path}", flush=True)
        
        print("\nData Preprocessing Completed Successfully!", flush=True)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}", flush=True)
        print("\nTraceback:", flush=True)
        traceback.print_exc()
        print("", flush=True)
