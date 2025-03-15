import os
import sys
import time
import logging
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import components
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessor
from src.components.data_transformation import DataTransformation
from src.components.model_building import ModelBuilder
from src.logger import logger

def run_pipeline():
    """
    Run the complete data processing and model training pipeline
    """
    try:
        print("="*50)
        print("STARTING SMS SPAM CLASSIFIER PIPELINE")
        print("="*50)
        logger.info("="*50)
        logger.info("STARTING SMS SPAM CLASSIFIER PIPELINE")
        logger.info("="*50)
        
        # Step 1: Data Ingestion
        print("\n\n" + "="*20 + " STEP 1: DATA INGESTION " + "="*20)
        logger.info("\n\n" + "="*20 + " STEP 1: DATA INGESTION " + "="*20)
        try:
            ingestion = DataIngestion()
            # Use the correct sequence of methods for data ingestion
            raw_data = ingestion.read_data()
            cleaned_data = ingestion.clean_data(raw_data)
            cleaned_data_path = ingestion.save_cleaned_data(cleaned_data)
            print(f"Data ingestion completed. Cleaned data saved to: {cleaned_data_path}")
            logger.info(f"Data ingestion completed. Cleaned data saved to: {cleaned_data_path}")
        except Exception as e:
            print(f"ERROR IN DATA INGESTION: {str(e)}")
            logger.error(f"ERROR IN DATA INGESTION: {str(e)}")
            traceback.print_exc()
            raise e
        
        time.sleep(1)  # Small pause for logging
        
        # Step 2: Data Preprocessing
        print("\n\n" + "="*20 + " STEP 2: DATA PREPROCESSING " + "="*20)
        logger.info("\n\n" + "="*20 + " STEP 2: DATA PREPROCESSING " + "="*20)
        try:
            preprocessor = DataPreprocessor(cleaned_data_path=cleaned_data_path)
            processed_data_path = preprocessor.preprocess()
            print(f"Data preprocessing completed. Processed data saved to: {processed_data_path}")
            logger.info(f"Data preprocessing completed. Processed data saved to: {processed_data_path}")
        except Exception as e:
            print(f"ERROR IN DATA PREPROCESSING: {str(e)}")
            logger.error(f"ERROR IN DATA PREPROCESSING: {str(e)}")
            traceback.print_exc()
            raise e
        
        time.sleep(1)  # Small pause for logging
        
        # Step 3: Data Transformation
        print("\n\n" + "="*20 + " STEP 3: DATA TRANSFORMATION " + "="*20)
        logger.info("\n\n" + "="*20 + " STEP 3: DATA TRANSFORMATION " + "="*20)
        try:
            transformer = DataTransformation(processed_data_path=processed_data_path)
            train_data_path, test_data_path = transformer.transform()
            print(f"Data transformation completed.")
            print(f"Training data saved to: {train_data_path}")
            print(f"Testing data saved to: {test_data_path}")
            logger.info(f"Data transformation completed.")
            logger.info(f"Training data saved to: {train_data_path}")
            logger.info(f"Testing data saved to: {test_data_path}")
        except Exception as e:
            print(f"ERROR IN DATA TRANSFORMATION: {str(e)}")
            logger.error(f"ERROR IN DATA TRANSFORMATION: {str(e)}")
            traceback.print_exc()
            raise e
        
        time.sleep(1)  # Small pause for logging
        
        # Step 4: Model Building
        print("\n\n" + "="*20 + " STEP 4: MODEL BUILDING " + "="*20)
        logger.info("\n\n" + "="*20 + " STEP 4: MODEL BUILDING " + "="*20)
        try:
            builder = ModelBuilder(train_data_path=train_data_path, test_data_path=test_data_path)
            model_path, vectorizer_path = builder.build()
            print(f"Model building completed.")
            print(f"Best model saved to: {model_path}")
            print(f"Vectorizer saved to: {vectorizer_path}")
            logger.info(f"Model building completed.")
            logger.info(f"Best model saved to: {model_path}")
            logger.info(f"Vectorizer saved to: {vectorizer_path}")
        except Exception as e:
            print(f"ERROR IN MODEL BUILDING: {str(e)}")
            logger.error(f"ERROR IN MODEL BUILDING: {str(e)}")
            traceback.print_exc()
            raise e
        
        print("\n\n" + "="*20 + " PIPELINE COMPLETED SUCCESSFULLY " + "="*20)
        print("="*50)
        logger.info("\n\n" + "="*20 + " PIPELINE COMPLETED SUCCESSFULLY " + "="*20)
        logger.info("="*50)
        
        return {
            "cleaned_data_path": cleaned_data_path,
            "processed_data_path": processed_data_path,
            "train_data_path": train_data_path,
            "test_data_path": test_data_path,
            "model_path": model_path,
            "vectorizer_path": vectorizer_path
        }
        
    except Exception as e:
        print(f"ERROR IN PIPELINE: {str(e)}")
        logger.error(f"ERROR IN PIPELINE: {str(e)}")
        print("Pipeline execution failed!")
        logger.error("Pipeline execution failed!")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run_pipeline() 