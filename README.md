# SMS Spam Classifier

A machine learning project that classifies SMS messages as spam or legitimate (ham) using Natural Language Processing (NLP) techniques.

## Project Overview

This project implements an end-to-end machine learning pipeline for SMS spam detection with the following components:

- Data ingestion and cleaning
- Text preprocessing with NLP techniques
- Feature engineering
- Model training with various Naive Bayes classifiers
- Model evaluation and selection
- Web application for real-time predictions

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model](#model)
- [Web Application](#web-application)
- [Usage](#usage)
- [Testing](#testing)

## Installation

### Requirements

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package and dependencies:
   ```
   pip install -e .
   ```

## Project Structure

```
sms-spam-classifier/
│
├── app.py                     # Streamlit web application
├── setup.py                   # Package installation configuration
├── requirements.txt           # Project dependencies
│
├── data/                      # Data directory
│   ├── raw/                   # Original dataset
│   ├── preprocessed/          # Preprocessed data
│   ├── processed/             # Processed data with features
│   └── transformed/           # Train-test split data
│
├── model/                     # Saved models
│   ├── model.pkl              # Trained model
│   └── vectorizer.pkl         # TF-IDF vectorizer
│
├── notebooks/                 # Jupyter notebooks for exploration
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── logger.py              # Logging utility
│   ├── exception.py           # Custom exception handling
│   ├── pipeline.py            # End-to-end ML pipeline
│   ├── predict.py             # Prediction module
│   │
│   └── components/            # Pipeline components
│       ├── __init__.py
│       ├── data_ingestion.py  # Data loading and cleaning
│       ├── data_preprocessing.py  # Text preprocessing and feature creation
│       ├── data_transformation.py  # Train-test splitting
│       └── model_building.py  # Model training and evaluation
│
└── logs/                      # Log files
```

## Data Pipeline

The pipeline consists of the following stages:

1. **Data Ingestion**: Reads and cleans the SMS dataset from source.
2. **Data Preprocessing**: 
   - Converts text to lowercase
   - Tokenizes text
   - Removes non-alphabetic characters
   - Removes stopwords
   - Applies stemming
   - Adds features like text length, word count, and sentence count
3. **Data Transformation**: Splits data into training (80%) and testing (20%) sets.
4. **Model Building**: 
   - Vectorizes text using TF-IDF
   - Trains multiple Naive Bayes models
   - Evaluates and selects the best model based on precision

## Model

The project uses the following models:

- Multinomial Naive Bayes
- Gaussian Naive Bayes
- Bernoulli Naive Bayes

These models are evaluated on precision, accuracy, recall, and F1 score, with the best model selected based on precision for spam detection.

## Web Application

A Streamlit web application provides a user-friendly interface for:

- Entering SMS messages
- Getting real-time spam classification predictions
- Simple and intuitive UI for immediate feedback

## Usage

### Running the Pipeline

To run the complete data pipeline:

```
python src/pipeline.py
```

This will:
- Load and process the data
- Train the models
- Save the best model and vectorizer to the model/ directory

### Running the Web Application

To launch the Streamlit application:

```
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.

### Package Functions

After installation, you can use the package in Python:

```python
from src.pipeline import run_pipeline

# Run the complete pipeline
results = run_pipeline()
```



## Author

- Haris Mehmood - harismehmood948@gmail.com
