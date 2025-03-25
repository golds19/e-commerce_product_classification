import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_variables():
    """
    Loading env variables from .env file
    """
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)

def get_mlflow_credentials():
    return{
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI'),
        'username': os.getenv('MLFLOW_TRACKING_USERNAME'),
        'password': os.getenv('MLFLOW_TRACKING_PASSWORD')
    }