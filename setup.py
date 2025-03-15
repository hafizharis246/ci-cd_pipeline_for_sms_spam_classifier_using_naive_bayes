from setuptools import find_packages, setup
from typing import List

# Function to get requirements from the requirements.txt file
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Remove -e . if present (it's used for local installs)
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

# Project metadata
setup(
    name="sms_spam_classifier",
    version="0.1.0",
    author="Hafiz Haris Mehmood",
    author_email="harismehmood948@gmail.com",
    description="SMS Spam Classifier using Machine Learning",
    long_description=open("README.md").read() if "README.md" in find_packages() else "A machine learning model to classify SMS messages as spam or ham",
    url="https://github.com/harismehmood948/sms-spam-classifier",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    
    # Define entry points for running the application
    entry_points={
        'console_scripts': [
            'run_data_pipeline=src.pipeline:run_pipeline',
            'run_spam_app=app:main'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 