import pandas as pd

def clean_data(df: pd.DataFrame):
    """
    Cleans data by 
    1. Column normalisation and Capitalize the first char of the column header
    2. Raise KeyError:  if column Date, Open, High, Low, Close, Volume is missing 
    3. Convert date to the variable type DateTime
    4. Sort the DateTime index via ascending order.
    5. Raise error: if column contains no values after conversion to correct datatype.
    6. Delete rows with missing value
    
    Parameters
    df (pd.DataFrame): Raw stocks data frame

    Returns
    pd.DataFrame: Cleaned stocks data frame
    """
    df.columns = df.columns.astype(str).str.strip().str.capitalize() #Strip and Captilise the first char of the column header, and the rest as lower char
    column_header_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for header_name in column_header_names:
        if header_name not in df.columns:
            raise KeyError(f"DataFrame must contain a {header_name} column")
        if header_name == "Date":
            df[header_name] = pd.to_datetime(df[header_name], errors="coerce") #Parse dates (handles mixed formats); invalid → NaT
            check_empty_col(df, header_name)
            df = df.sort_values(header_name).reset_index(drop=True) #Sort date by ascending order
        else:
            df[header_name] = pd.to_numeric(df[header_name], errors='coerce') #Remove all rows in the column where the value cannot be converted to numeric
            check_empty_col(df, header_name)
    df = df.dropna() #Delete rows with any missing value
    return df


def check_empty_col(df, header_name):
    """
    Validate that a DataFrame column contains at least one non-missing value.
    
    Parameters
    df(pandas.DataFrame):DataFrame to validate.
    header_name : str
        Name of the column to check.

    Returns
    None
        This function performs validation only and does not return a value.

    Raises
    ValueError
        If the specified column exists but all values are missing (NaN/None).
    """
    if df[header_name].isna().all(): #If column name does not contain any datetime values
        raise ValueError(f"{header_name} column contains no values")