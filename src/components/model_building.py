import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import joblib
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import from src
from src.logger import logger
from src.exception import CustomException

class ModelBuilder:
    def __init__(self, train_data_path=None, test_data_path=None, model_dir=None):
        try:
            # Set default paths if none provided
            if train_data_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.train_data_path = os.path.join(project_root, "data", "transformed", "train.csv")
            else:
                self.train_data_path = train_data_path
                
            if test_data_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.test_data_path = os.path.join(project_root, "data", "transformed", "test.csv")
            else:
                self.test_data_path = test_data_path
                
            if model_dir is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.model_dir = os.path.join(project_root, "model")
            else:
                self.model_dir = model_dir
                
            logger.info(f"[MODEL BUILDING] Initialized ModelBuilder")
            logger.info(f"[MODEL BUILDING] Will load train data from: {self.train_data_path}")
            logger.info(f"[MODEL BUILDING] Will load test data from: {self.test_data_path}")
            logger.info(f"[MODEL BUILDING] Will save models to: {self.model_dir}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def read_data(self):
        """
        Read train and test datasets
        """
        try:
            logger.info("[MODEL BUILDING] Reading train and test data")
            
            # Read train data
            train_df = pd.read_csv(self.train_data_path)
            logger.info(f"[MODEL BUILDING] Train data shape: {train_df.shape}")
            
            # Read test data
            test_df = pd.read_csv(self.test_data_path)
            logger.info(f"[MODEL BUILDING] Test data shape: {test_df.shape}")
            
            return train_df, test_df
        except Exception as e:
            logger.error(f"[MODEL BUILDING] Error reading data: {e}")
            raise CustomException(e, sys)
    
    def prepare_data(self, train_df, test_df):
        """
        Prepare data for model training
        """
        try:
            logger.info("[MODEL BUILDING] Preparing data for model training")
            
            # Print initial info about nulls
            logger.info(f"[MODEL BUILDING] Initial null counts in train data:")
            for col in train_df.columns:
                null_count = train_df[col].isnull().sum()
                if null_count > 0:
                    logger.info(f"- {col}: {null_count} nulls")
            
            # Drop rows with null values
            train_df = train_df.dropna(subset=['transformed_text', 'target'])
            test_df = test_df.dropna(subset=['transformed_text', 'target'])
            
            logger.info(f"[MODEL BUILDING] Shape after dropping nulls - Train: {train_df.shape}, Test: {test_df.shape}")
            
            # Check if required columns exist
            required_columns = ['transformed_text', 'target']
            for col in required_columns:
                if col not in train_df.columns:
                    raise ValueError(f"Required column '{col}' not found in train data")
                if col not in test_df.columns:
                    raise ValueError(f"Required column '{col}' not found in test data")
            
            # Separate features and target
            X_train = train_df['transformed_text']
            y_train = train_df['target']
            X_test = test_df['transformed_text']
            y_test = test_df['target']
            
            # Convert target to numeric if needed
            if y_train.dtype == 'object':
                logger.info("[MODEL BUILDING] Converting target to numeric")
                y_train = pd.to_numeric(y_train, errors='coerce')
                y_test = pd.to_numeric(y_test, errors='coerce')
                
                # Drop any rows where conversion failed
                valid_train_idx = ~y_train.isnull()
                valid_test_idx = ~y_test.isnull()
                
                X_train = X_train[valid_train_idx]
                y_train = y_train[valid_train_idx]
                X_test = X_test[valid_test_idx]
                y_test = y_test[valid_test_idx]
            
            # Initialize and fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=3000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            logger.info(f"[MODEL BUILDING] Final shapes:")
            logger.info(f"X_train: {X_train_tfidf.shape}")
            logger.info(f"X_test: {X_test_tfidf.shape}")
            logger.info(f"y_train: {y_train.shape}")
            logger.info(f"y_test: {y_test.shape}")
            
            return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
        except Exception as e:
            logger.error(f"[MODEL BUILDING] Error preparing data: {e}")
            raise CustomException(e, sys)
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all three Naive Bayes models
        """
        try:
            logger.info("[MODEL BUILDING] Training and evaluating models")
            
            # Initialize models
            models = {
                'GaussianNB': GaussianNB(),
                'MultinomialNB': MultinomialNB(),
                'BernoulliNB': BernoulliNB()
            }
            
            # Train and evaluate each model
            results = {}
            for name, model in models.items():
                logger.info(f"[MODEL BUILDING] Training {name}")
                
                # Train model
                start_time = time.time()
                model.fit(X_train.toarray(), y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_test.toarray())
                
                # Calculate metrics
                precision = precision_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f1': f1,
                    'train_time': train_time
                }
                
                logger.info(f"[MODEL BUILDING] {name} Results:")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1 Score: {f1:.4f}")
                logger.info(f"Training Time: {train_time:.2f} seconds")
            
            return results
        except Exception as e:
            logger.error(f"[MODEL BUILDING] Error training and evaluating models: {e}")
            raise CustomException(e, sys)
    
    def save_best_model(self, results, vectorizer):
        """
        Save the best model (based on precision) and vectorizer
        """
        try:
            # Find best model based on precision
            best_model_name = max(results.items(), key=lambda x: x[1]['precision'])[0]
            best_model = results[best_model_name]['model']
            
            logger.info(f"[MODEL BUILDING] Best model: {best_model_name}")
            logger.info(f"[MODEL BUILDING] Best precision: {results[best_model_name]['precision']:.4f}")
            
            # Create model directory if it doesn't exist
            if not os.path.exists(self.model_dir):
                logger.info(f"[MODEL BUILDING] Creating model directory: {self.model_dir}")
                os.makedirs(self.model_dir, exist_ok=True)
            
            # Save best model
            model_path = os.path.join(self.model_dir, "model.pkl")
            joblib.dump(best_model, model_path)
            logger.info(f"[MODEL BUILDING] Best model saved to: {model_path}")
            
            # Save vectorizer
            vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"[MODEL BUILDING] Vectorizer saved to: {vectorizer_path}")
            
            return model_path, vectorizer_path
        except Exception as e:
            logger.error(f"[MODEL BUILDING] Error saving models: {e}")
            raise CustomException(e, sys)
    
    def build(self):
        """
        Execute the complete model building pipeline
        """
        try:
            # Read data
            train_df, test_df = self.read_data()
            
            # Prepare data
            X_train, X_test, y_train, y_test, vectorizer = self.prepare_data(train_df, test_df)
            
            # Train and evaluate models
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            # Save best model and vectorizer
            model_path, vectorizer_path = self.save_best_model(results, vectorizer)
            
            logger.info(f"[MODEL BUILDING] Model building completed successfully")
            return model_path, vectorizer_path
        except Exception as e:
            logger.error(f"[MODEL BUILDING] Error in model building pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test model building
    try:
        import time
        print("Starting Model Building...", flush=True)
        time.sleep(1)  # Give time for print to flush
        
        builder = ModelBuilder()
        
        # Read the data
        print("Reading train and test data...", flush=True)
        time.sleep(1)
        
        print(f"Train data path: {builder.train_data_path}", flush=True)
        print(f"Test data path: {builder.test_data_path}", flush=True)
        
        if not os.path.exists(builder.train_data_path) or not os.path.exists(builder.test_data_path):
            print("ERROR: Train or test data files not found", flush=True)
            sys.exit(1)
            
        train_df, test_df = builder.read_data()
        print(f"\nInitial shapes:")
        print(f"Train data shape: {train_df.shape}", flush=True)
        print(f"Test data shape: {test_df.shape}", flush=True)
        print("\nTrain data columns:", flush=True)
        print(train_df.columns.tolist(), flush=True)
        print("\nTrain data info:", flush=True)
        print(train_df.info(), flush=True)
        print("\nSample of train data:", flush=True)
        print(train_df.head(), flush=True)
        
        # Prepare data
        print("\nPreparing data for model training...", flush=True)
        X_train, X_test, y_train, y_test, vectorizer = builder.prepare_data(train_df, test_df)
        print(f"TF-IDF features shape: {X_train.shape}", flush=True)
        
        # Train and evaluate models
        print("\nTraining and evaluating models...", flush=True)
        results = builder.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save best model
        print("\nSaving best model and vectorizer...", flush=True)
        model_path, vectorizer_path = builder.save_best_model(results, vectorizer)
        print(f"Best model saved to: {model_path}", flush=True)
        print(f"Vectorizer saved to: {vectorizer_path}", flush=True)
        
        print("\nModel Building Completed Successfully!", flush=True)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}", flush=True)
        print("\nTraceback:", flush=True)
        traceback.print_exc()
        print("", flush=True)
