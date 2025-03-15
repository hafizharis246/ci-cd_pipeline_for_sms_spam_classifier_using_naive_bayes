import pandas as pd
import numpy as np
import os
import sys
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

# Import from src
from src.logger import logger
from src.exception import CustomException

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    logger.info("[FEATURE ENGINEERING] NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"[FEATURE ENGINEERING] Error downloading NLTK resources: {e}")
    raise CustomException(e, sys)

class DataTransformation:
    def __init__(self, processed_data_path=None, transformed_data_dir=None):
        try:
            # Set default paths if none provided
            if processed_data_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.processed_data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
            else:
                self.processed_data_path = processed_data_path
                
            if transformed_data_dir is None:
                # Save transformed data in the transformed folder
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.transformed_data_dir = os.path.join(project_root, "data", "transformed")
            else:
                self.transformed_data_dir = transformed_data_dir
                
            logger.info(f"[DATA TRANSFORMATION] Initialized DataTransformation")
            logger.info(f"[DATA TRANSFORMATION] Will load processed data from: {self.processed_data_path}")
            logger.info(f"[DATA TRANSFORMATION] Will save transformed data to: {self.transformed_data_dir}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def read_processed_data(self):
        """
        Read the processed dataset
        """
        try:
            logger.info(f"[DATA TRANSFORMATION] Reading processed data from {self.processed_data_path}")
            df = pd.read_csv(self.processed_data_path)
            logger.info(f"[DATA TRANSFORMATION] Processed data read successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[DATA TRANSFORMATION] Error reading processed data: {e}")
            raise CustomException(e, sys)
    
    def split_data(self, df):
        """
        Split the data into train and test sets
        """
        try:
            logger.info("[DATA TRANSFORMATION] Starting train-test split")
            
            # Split features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            # Combine features and target back
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            logger.info(f"[DATA TRANSFORMATION] Train data shape: {train_data.shape}")
            logger.info(f"[DATA TRANSFORMATION] Test data shape: {test_data.shape}")
            
            return train_data, test_data
        except Exception as e:
            logger.error(f"[DATA TRANSFORMATION] Error in train-test split: {e}")
            raise CustomException(e, sys)
    
    def save_transformed_data(self, train_data, test_data):
        """
        Save train and test datasets
        """
        try:
            # Create transformed directory if it doesn't exist
            if not os.path.exists(self.transformed_data_dir):
                logger.info(f"[DATA TRANSFORMATION] Creating directory: {self.transformed_data_dir}")
                os.makedirs(self.transformed_data_dir, exist_ok=True)
            
            # Save train and test data
            train_path = os.path.join(self.transformed_data_dir, "train.csv")
            test_path = os.path.join(self.transformed_data_dir, "test.csv")
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            logger.info(f"[DATA TRANSFORMATION] Train data saved to: {train_path}")
            logger.info(f"[DATA TRANSFORMATION] Test data saved to: {test_path}")
            
            return train_path, test_path
        except Exception as e:
            logger.error(f"[DATA TRANSFORMATION] Error saving transformed data: {e}")
            raise CustomException(e, sys)
    
    def transform(self):
        """
        Execute the complete transformation pipeline
        """
        try:
            # Read processed data
            df = self.read_processed_data()
            
            # Split data into train and test sets
            train_data, test_data = self.split_data(df)
            
            # Save transformed data
            train_path, test_path = self.save_transformed_data(train_data, test_data)
            
            logger.info(f"[DATA TRANSFORMATION] Transformation completed successfully")
            return train_path, test_path
        except Exception as e:
            logger.error(f"[DATA TRANSFORMATION] Error in transformation pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test data transformation
    try:
        import time
        print("Starting Data Transformation...", flush=True)
        time.sleep(1)  # Give time for print to flush
        
        transformer = DataTransformation()
        
        # Read the processed data
        print("Reading processed data...", flush=True)
        time.sleep(1)
        
        print(f"Processed data path: {transformer.processed_data_path}", flush=True)
        if not os.path.exists(transformer.processed_data_path):
            print(f"ERROR: Processed data file not found at {transformer.processed_data_path}", flush=True)
            sys.exit(1)
            
        df = transformer.read_processed_data()
        print(f"Processed data shape: {df.shape}", flush=True)
        
        # Split the data
        print("\nPerforming train-test split...", flush=True)
        train_data, test_data = transformer.split_data(df)
        print(f"Train data shape: {train_data.shape}", flush=True)
        print(f"Test data shape: {test_data.shape}", flush=True)
        
        # Save the transformed data
        print("\nSaving transformed data...", flush=True)
        train_path, test_path = transformer.save_transformed_data(train_data, test_data)
        print(f"Train data saved to: {train_path}", flush=True)
        print(f"Test data saved to: {test_path}", flush=True)
        
        print("\nData Transformation Completed Successfully!", flush=True)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}", flush=True)
        print("\nTraceback:", flush=True)
        traceback.print_exc()
        print("", flush=True) 