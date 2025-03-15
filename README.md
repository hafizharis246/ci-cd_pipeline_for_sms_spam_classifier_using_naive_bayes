# SMS Spam Classifier

A machine learning application that classifies SMS messages as spam or ham (not spam) using Natural Language Processing (NLP) and Naive Bayes classification algorithms.

## Project Overview

This project implements a complete end-to-end machine learning pipeline for SMS spam classification, including:

- Data ingestion and cleaning
- Text preprocessing and feature engineering
- Model training with multiple Naive Bayes classifiers
- Model evaluation and selection
- Web application for real-time predictions
- CI/CD pipeline with GitHub Actions

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager
- virtualenv (optional but recommended)

### Installation

#### Using virtual environment (recommended for development)

**For Windows:**
```bash
# Run the setup script
setup_env.bat
```

**For macOS/Linux:**
```bash
# Make the script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

## Usage

### Running the Streamlit Web App

```bash
# Activate virtual environment first
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app will be available at http://localhost:8501

### Running the Complete Pipeline

To run the full data processing and model training pipeline:

```bash
# Activate virtual environment first
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run the pipeline
python src/pipeline.py
```

## Project Structure

```
SMS_Spam_Classifier/
├── .github/            # GitHub Actions workflows
├── data/               # Data directories
│   ├── raw/            # Original dataset
│   ├── preprocessed/   # Cleaned dataset
│   ├── processed/      # Feature-engineered dataset
│   └── transformed/    # Train-test split datasets
├── model/              # Saved model and vectorizer
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit tests
├── src/                # Source code
│   ├── components/     # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_transformation.py
│   │   └── model_building.py
│   ├── pipeline.py     # Complete pipeline
│   ├── logger.py       # Logging utilities
│   └── exception.py    # Custom exception handling
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
├── setup.py            # Package installation setup
└── README.md           # Project documentation
```

## CI/CD Pipeline

This project uses GitHub Actions for Continuous Integration and Continuous Deployment:

1. **Testing**: Runs linting and unit tests
2. **Deployment**: Deploys to Heroku for production

## Make Commands

The project includes a Makefile with helpful commands:

```bash
# Set up the environment
make setup

# Run tests
make test

# Run linting
make lint

# Run the app
make run-app

# Run the pipeline
make run-pipeline

# Create necessary directories
make create-dirs
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Hafiz Haris Mehmood** - harismehmood948@gmail.com
- **Project Link**: [https://github.com/harismehmood948/sms-spam-classifier](https://github.com/harismehmood948/sms-spam-classifier) 