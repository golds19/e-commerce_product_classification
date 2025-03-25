import pandas as pd
from src.config import Config
import os

def load_raw_data(path, nrows=5000):
    """
    Load the raw CSV data
    
    Args:
        path (str): Path to the CSV file
        nrows (int): Number of rows to read
    
    Returns:
        pd.DataFrame: Raw dataframe
    """
    df = pd.read_csv(path, nrows=nrows, on_bad_lines='warn')
    return df

def filter_categories(df, categories):
    """
    Filter dataframe for specific subcategories
    
    Args:
        df (pd.DataFrame): Input dataframe
        categories (list): List of categories to keep
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    return df.loc[lambda df: df['subCategory'].isin(categories)]

def preprocess_data():
    """
    Main preprocessing function that loads and processes the data
    
    Returns:
        pd.DataFrame: Processed and filtered dataframe
    """
    # Load raw data
    df = load_raw_data(
        path=os.path.join(Config.DATASET_PATH, "styles.csv"),
    )
    
    # Create image filename column if it doesn't exist
    if 'image' not in df.columns:
        df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    
    # Filter for chosen subcategories
    df_filtered = filter_categories(df, Config.CHOSEN_SUBCATEGORIES)
    
    # Shuffle the dataframe
    df_filtered = df_filtered.sample(
        frac=1, 
    ).reset_index(drop=True)
    
    return df_filtered