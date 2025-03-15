@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Installing project in development mode...
pip install -e .

echo Virtual environment setup complete!
echo To activate the environment, run: venv\Scripts\activate 