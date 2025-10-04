import pandas as pd

def clean_data(df: pd.DataFrame):
    """
    Cleans data by deleting rows with missing value, sort dateindex to ascending order
    
    Parameters
    df (pd.DataFrame): Raw stocks data frame

    Returns
    pd.DataFrame: Cleaned stocks data frame
    """
    # Convert 'Date' to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna() #Delete rows with any missing value
    df = df.sort_index()

    return df