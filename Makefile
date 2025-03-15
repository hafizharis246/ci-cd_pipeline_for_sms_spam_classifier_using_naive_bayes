.PHONY: clean setup test lint run-app run-pipeline create-dirs

# Set up virtual environment and install dependencies
setup:
	python -m venv venv
	venv\Scripts\activate && pip install -r requirements.txt && pip install -e .

# Run tests
test:
	pytest tests/

# Run linting
lint:
	flake8 src/ app.py

# Run the Streamlit app
run-app:
	streamlit run app.py

# Run the complete pipeline
run-pipeline:
	python src/pipeline.py

# Clean up compiled Python files and cache
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Create directories needed for the project
create-dirs:
	mkdir -p data/raw data/preprocessed data/processed data/transformed model logs