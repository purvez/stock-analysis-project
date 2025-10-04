import pandas as pd
import numpy as np
from analysis import compute_sma, compute_daily_returns, count_runs, max_profit

# def validate_sma():
#     test_prices = [10, 20, 30, 40, 50]
#     sma_manual = [None, None, 20.0, 30.0, 40.0]  # window=3
#     sma_computed = compute_sma(test_prices, 3)
#     return sma_manual == sma_computed

def validate_sma(df, window, implemented_sma):
    """
    Validates our own implemented_sma against pandas built-in rolling mean.

    Args:
    df (pd.DataFrame): Stock dataset
    window (int): Number of days to average over
    implemented_sma (pd.DataFrame): SMA of our own calculated SMA

    Returns:
    print whether SMA matches to expected
    """
    pandas_sma = df["Close"].rolling(window=window).mean().tolist() #Calculate SMA using pandas built in rolling mean
    implemented_sma_value = implemented_sma(df['Close'].tolist(), window)

    mismatch = [] #Store any mismatch for debug
    if len(pandas_sma) != len(implemented_sma_value):
        print("Found mismatch!")
    for i in range(window, len(pandas_sma)): #Reason for window is to skip the first few window index where SMA cannot be calculated
        each_pd_sma_value = round(pandas_sma[i],4) #Get each individual pandas SMA value, round to 4dp
        each_implemented_sma_value = round(implemented_sma_value[i],4) #Get each individual implemented SMA value, round to 4dp
        if each_pd_sma_value != each_implemented_sma_value: #If values are !=, store date, pd sma value and implemented sma value into mismatch list
            mismatch.append(each_pd_sma_value, each_implemented_sma_value) 

    #If there are mismatch, the mistmatch list will be printed for debugging
    if len(mismatch) == 0: 
        print("SMA matches: pandas rolling mean")
    else:
        print("SMA mismatch: pandas rolling mean")
        print(mismatch)

def validate_daily_returns(df: pd.DataFrame, implemented_daily_returns, price_col="Close", return_col="Return", tol=1e-8):
    """
    Compare custom daily returns in df[return_col] with pandas' pct_change().

    Args:
        df (pd.DataFrame): DataFrame containing price and return columns.
        price_col (str): Column name for prices (default: 'Close').
        return_col (str): Column name for custom returns (default: 'Return').
        tol (float): Tolerance for floating-point comparison.
    """
    df = implemented_daily_returns(df)
    pandas_returns = df[price_col].pct_change()
    comparison = pd.DataFrame({
        "Custom": df[return_col].dropna(),
        "Pandas": pandas_returns.dropna(),
    })
    comparison["Diff"] = comparison["Custom"] - comparison["Pandas"]
    comparison["Match"] = comparison["Diff"].abs() <= tol

    if comparison["Match"].all():
        print("Daily returns matches within tolerance.")
    else:
        print("Daily returns mismatch\n")
        print(comparison[~comparison["Match"]])

def validate_max_profit(implemented_max_profit):
    """
    Validates our own implemented max profit function against test case

    Args:
    implemented_max_profit(list): Own implemented function to calculate max profit

    Returns:
    print whether max profit matches to expected
    """
    prices = [7, 1, 5, 3, 6, 4]
    expected = 7
    result = implemented_max_profit(prices)
    if expected == result:
        print("Max prices matches")
    else:
        print("Max prices mismatch")

def validate_count_runs(df: pd.DataFrame, implemented_count_runs):
    """
    Validates our own implemented count runs against test case

    Args:
    implemented_count_runs(df:pd.Dataframe): Own implemented function to count runs

    Returns:
    print whether count runs matches to expected
    """

    def make_df(values):
        return pd.DataFrame({'Close': values})
    
    test_set = [
        {
            'name': 'Monotonic Up',
            'df': make_df([1, 2, 3, 4, 5]),
            'runs': {'upward': [4], 'downward': []}
        },
        {
            'name': 'Monotonic Down',
            'df': make_df([5, 4, 3, 2, 1]),
            'runs': {'upward': [0], 'downward': [4]}
        },
        {
            'name': 'Alternating Up/Down',
            'df': make_df([1, 2, 1, 2, 1, 2]),  # diffs: +1, -1, +1, -1, +1
            'runs': {'upward': [1,1,1], 'downward': [1,1]}
        },
        {
            'name': 'Flats Break Runs',
            'df': make_df([1, 1, 2, 2, 2, 1, 1, 1, 1, 1]),
            # diffs: 0, +1, 0, 0, -1, 0, 0, 0, 0
            'runs': {'upward': [1, 1], 'downward': [1, 1]}
        },
        {
            'name': 'NaN Breaks Runs',
            'df': make_df([10, 11, 12, np.nan, 11, 12, 11, 10]),
            # diffs: +1, +1, NaN, NaN, +1, -1, -1
            'runs': {'upward': [2, 3], 'downward': [1, 2]}
        },
        {
            'name': 'Short Series (len=1)',
            'df': make_df([100]),
            'runs': {'upward': [0, 0], 'downward': [0, 0]}
        },
    ]

    for case in test_set:
        name = case['name']
        df = case['df']
        expected = case['runs']

        actual = implemented_count_runs(df)
        if actual == expected:
            print(f"{name}, Count runs match")
        else:
            print(f"Count runs mismatch: {name}")

def validate_sma_test_cases(df, window, implemented_sma):
    """
    Validates our own implemented_sma against data shorter than window and Nan in data

    Args:
    df (pd.DataFrame): Stock dataset
    window (int): Number of days to average over
    implemented_sma (pd.DataFrame): SMA of our own calculated SMA

    Returns:
    print whether SMA matches to expected
    """
    # Case 1: Data shorter than window
    prices_short = [100, 101]  # len=2 < window
    expected = [None, None]
    implemented_sma_value = implemented_sma(prices_short, window)
    if implemented_sma_value == expected:
        print("SMA test cases matches: Data shorter than window")
    else:
        print("SMA test case do not matches: Data shorter than window")

    #Case 2: Data with nan
    prices_with_nan = [10, 11, float('nan'), 14, 15]
    Expected = [None, None, np.nan, np.nan, np.nan]
    implemented_sma_value = implemented_sma(prices_short, window)
    if implemented_sma_value == expected:
        print("SMA test cases matches: data with nan")
    else:
        print("SMA test case do not matches: data with nan")


def run_all_validation():
    df = pd.read_csv(r"C:\Users\aloys\Downloads\SIT\INF1002_programming_fundementals\proj\stock-analysis-project\stock_data.csv")
    validate_sma(df, 3, compute_sma)
    validate_sma_test_cases(df, 3, compute_sma)
    validate_daily_returns(df, compute_daily_returns)
    validate_max_profit(max_profit)
    validate_count_runs(df, count_runs)


run_all_validation()