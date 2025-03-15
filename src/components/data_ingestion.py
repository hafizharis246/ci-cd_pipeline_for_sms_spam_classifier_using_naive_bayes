import os
import sys
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import from src
from src.logger import logger
from src.exception import CustomException

class DataIngestion:
    def __init__(self, raw_data_path=None, cleaned_data_path=None):
        try:
            # Set default paths if none provided
            if raw_data_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.raw_data_path = os.path.join(project_root, "data", "raw", "spam.csv")
            else:
                self.raw_data_path = raw_data_path
                
            if cleaned_data_path is None:
                # Save cleaned data in the processed folder
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.cleaned_data_path = os.path.join(project_root, "data", "preprocessed", "cleaned.csv")
            else:
                self.cleaned_data_path = cleaned_data_path
                
            logger.info(f"[DATA INGESTION] Initialized DataIngestion with raw_data_path: {self.raw_data_path}")
            logger.info(f"[DATA INGESTION] Cleaned data will be saved to: {self.cleaned_data_path}")
        except Exception as e:
            raise CustomException(e, sys)
    
    def read_data(self):
        """
        Read the entire SMS dataset
        """
        try:
            logger.info(f"[DATA INGESTION] Reading data from {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path, encoding="latin1")
            logger.info(f"[DATA INGESTION] Data read successfully with shape: {df.shape} - Total rows: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"[DATA INGESTION] Error reading data: {e}")
            raise CustomException(e, sys)
    
    def clean_data(self, df):
        """
        Clean the entire dataset - remove unnecessary columns, duplicates, and nulls
        """
        try:
            logger.info(f"[DATA INGESTION] Starting basic data cleaning on entire dataset ({len(df)} rows)")
            
            # Count original rows
            original_rows = len(df)
            original_columns = df.shape[1]
            logger.info(f"[DATA INGESTION] Original data has {original_rows} rows and {original_columns} columns")
            
            # FIRST STEP: Drop the Unnamed columns
            columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
            df_dropped_cols = df.drop(columns=columns_to_drop, errors='ignore')
            
            cols_dropped = original_columns - df_dropped_cols.shape[1]
            logger.info(f"[DATA INGESTION] Dropped {cols_dropped} unnecessary columns")
            logger.info(f"[DATA INGESTION] Remaining columns: {df_dropped_cols.columns.tolist()}")
            
            # Check for NaN values in the dataset
            nan_count = df_dropped_cols.isna().sum()
            logger.info(f"[DATA INGESTION] Number of NaN values in dataset: {nan_count}")
            
            # Remove duplicates from the entire dataset
            df_no_duplicates = df_dropped_cols.drop_duplicates(keep='first')
            duplicates_removed = len(df_dropped_cols) - len(df_no_duplicates)
            logger.info(f"[DATA INGESTION] Removed {duplicates_removed} duplicate rows")
            
            # Check columns with NaN values
            for col in df_no_duplicates.columns:
                nan_in_col = df_no_duplicates[col].isna().sum()
                if nan_in_col > 0:
                    logger.info(f"[DATA INGESTION] Column '{col}' has {nan_in_col} NaN values")
            
            # Drop rows with null values in critical columns
            df_cleaned = df_no_duplicates.dropna(subset=['v1', 'v2'])
            nulls_removed = len(df_no_duplicates) - len(df_cleaned)
            logger.info(f"[DATA INGESTION] Removed {nulls_removed} rows with critical null values")
            
            # Final stats
            final_rows = len(df_cleaned)
            final_columns = df_cleaned.shape[1]
            rows_reduced = original_rows - final_rows
            percent_reduced = (rows_reduced / original_rows) * 100 if original_rows > 0 else 0
            
            logger.info(f"[DATA INGESTION] Data cleaning completed on entire dataset.")
            logger.info(f"[DATA INGESTION] Final shape: {df_cleaned.shape} - {final_rows} rows, {final_columns} columns")
            logger.info(f"[DATA INGESTION] Total rows reduced: {rows_reduced} ({percent_reduced:.2f}%)")
            
            return df_cleaned
        except Exception as e:
            logger.error(f"[DATA INGESTION] Error in data cleaning: {e}")
            raise CustomException(e, sys)
    
    def save_cleaned_data(self, df):
        """
        Save the entire cleaned dataframe to CSV in the processed folder
        """
        try:
            # Check if the dataframe is empty
            if df.empty:
                logger.error("[DATA INGESTION] Cannot save empty dataframe!")
                raise ValueError("Dataframe is empty, cannot save to CSV")
                
            # Print absolute file path for clarity
            abs_path = os.path.abspath(self.cleaned_data_path)
            logger.info(f"[DATA INGESTION] Attempting to save to absolute path: {abs_path}")
            
            # Ensure the directory exists
            data_dir = os.path.dirname(self.cleaned_data_path)
            if not os.path.exists(data_dir):
                logger.info(f"[DATA INGESTION] Creating directory: {data_dir}")
                os.makedirs(data_dir, exist_ok=True)
            
            # Check if directory is writable
            if not os.access(data_dir, os.W_OK):
                logger.error(f"[DATA INGESTION] Directory {data_dir} is not writable!")
                raise PermissionError(f"Directory {data_dir} is not writable")
            
            # Save with explicit error handling
            try:
                # Print dataframe info before saving
                logger.info(f"[DATA INGESTION] DataFrame info before saving:\n{df.info()}")
                logger.info(f"[DATA INGESTION] Saving {len(df)} rows to {self.cleaned_data_path}")
                
                # Save the entire dataframe
                df.to_csv(self.cleaned_data_path, index=False)
                
                # Verify the file was created
                if os.path.exists(self.cleaned_data_path):
                    file_size = os.path.getsize(self.cleaned_data_path)
                    logger.info(f"[DATA INGESTION] File successfully created! Size: {file_size} bytes")
                else:
                    logger.error(f"[DATA INGESTION] File was not created despite no errors!")
                
                logger.info(f"[DATA INGESTION] Entire cleaned dataset ({len(df)} rows) saved to {self.cleaned_data_path}")
                logger.info(f"[DATA INGESTION] This cleaned data file will be used as input for Data Transformation")
                
                return self.cleaned_data_path
            except Exception as inner_e:
                logger.error(f"[DATA INGESTION] Specific error during file saving: {str(inner_e)}")
                raise
        except Exception as e:
            logger.error(f"[DATA INGESTION] Error saving cleaned data: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test data ingestion and cleaning
    try:
        # Force print output to display immediately
        import sys
        
        print("Starting Data Ingestion Process...", flush=True)
        data_ingestion = DataIngestion()
        
        # Read the entire raw dataset
        print("Reading raw data...", flush=True)
        df = data_ingestion.read_data()
        print(f"Raw data shape: {df.shape} - Total rows: {len(df)}", flush=True)
        print(f"\nReading data from: {data_ingestion.raw_data_path}", flush=True)
        print("\nSample of raw data (first 5 rows for display only):", flush=True)
        print(df.head(5), flush=True)
        print("\n(Note: Processing will be performed on ALL rows, not just these 5 samples)", flush=True)
        
        # Clean the entire dataset
        print("\nCleaning data...", flush=True)
        cleaned_df = data_ingestion.clean_data(df)
        print(f"\nCleaned data shape: {cleaned_df.shape} - Total rows: {len(cleaned_df)}", flush=True)
        print("\nSample of cleaned data (first 5 rows for display only):", flush=True)
        print(cleaned_df.head(5), flush=True)
        
        # Save the entire cleaned dataset
        print("\nSaving cleaned data...", flush=True)
        cleaned_data_path = data_ingestion.save_cleaned_data(cleaned_df)
        print(f"\nEntire cleaned dataset ({len(cleaned_df)} rows) saved to: {cleaned_data_path}", flush=True)
        print(f"Absolute path: {os.path.abspath(cleaned_data_path)}", flush=True)
        print("This file will be used as input for the feature engineering step.", flush=True)
        
        print("\nData Ingestion Process Completed Successfully!", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True) 