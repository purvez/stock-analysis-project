import pandas as pd

def clean_data(df: pd.DataFrame):
    """
    Cleans data by 
    1. Convert Date column to DateTime
    2. Delete any rows that cannot be converted to DateTime
    3. Convert to '%m/%d/%Y' format
    4. Deleting rows with missing value
    5. Sort dateindex to ascending order
    6. Only capitalise the first char of the column header
    
    Parameters
    df (pd.DataFrame): Raw stocks data frame

    Returns
    pd.DataFrame: Cleaned stocks data frame
    """
    df['Date'] = pd.to_datetime(df['Date'], errors="coerce") #Parse dates (handles mixed formats); invalid → NaT
    df = df.dropna(subset=["Date"]) #Drop any date that cannot be converted to date time
    df = df.sort_values('Date').reset_index(drop=True) #Sort date by ascending order
    df = df.dropna() #Delete rows with any missing value
    df.columns = df.columns.astype(str).str.strip().str.capitalize() #Captilise the first char of the column header, and the rest as lower char
    df = check_column_var_type(df)
    return df

def check_column_var_type(df):
    #Input validation: Check if df has the column Open, High, Low, Close, Volume
    column_header_names = ["Open", "High", "Low", "Close", "Volume"]
    for header_name in column_header_names:
        if header_name not in df.columns:
            raise KeyError(f"DataFrame must contain a {header_name} column")
        #Input validation: Check if column contains numeric value
        df[header_name] = pd.to_numeric(df[header_name], errors='coerce') #Remove all rows in the column where the value cannot be converted to numeric
        if df['Close'].isna().all(): #If column name does not contain any numeric value
            raise ValueError(f"{header_name} column contains no numeric values")
    return df