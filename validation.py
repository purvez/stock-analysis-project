import pandas as pd
import numpy as np
import os
from analysis import *
from dataclasses import dataclass

@dataclass
class TestResult:
    name: str
    passed: bool
    details: any = None


def validate_sma(df, window, implemented_sma):
    """
    Validates our own implemented_sma against pandas built-in rolling mean.

    Args:
    df (pd.DataFrame): Stock dataset
    window (int): Number of days to average over
    implemented_sma (pd.DataFrame): SMA of our own calculated SMA

    Returns:
    passed: bool, details: list
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
        passed = True
    else:
        passed = False
    return passed, mismatch

def validate_daily_returns(df: pd.DataFrame, implemented_daily_returns, price_col="Close", return_col="Return", tol=1e-8):
    """
    Compare custom daily returns in df[return_col] with pandas' pct_change().

    Args:
        df (pd.DataFrame): DataFrame containing price and return columns.
        price_col (str): Column name for prices (default: 'Close').
        return_col (str): Column name for custom returns (default: 'Return').
        tol (float): Tolerance for floating-point comparison.

    Returns:
        passed: bool, details: list
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
        return True, []
    else:
        return False, comparison[~comparison["Match"]].to_dict(orient="records")

def validate_max_profit(implemented_max_profit):
    """
    Validates our own implemented max profit function against test case

    Args:
    implemented_max_profit(list): Own implemented function to calculate max profit

    Returns:
    passed: bool, details: list
    """
    prices = [7, 1, 5, 3, 6, 4]
    expected = 7
    actual = implemented_max_profit(prices)
    if expected == actual:
        return True, []
    else:
        return False, {"Expected": expected, "actual": actual}

def validate_count_runs(df: pd.DataFrame, implemented_count_runs):
    """
    Validates our own implemented count runs against test case

    Args:
    implemented_count_runs(df:pd.Dataframe): Own implemented function to count runs

    Returns:
    passed: bool, details: list
    """
    details = []
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
            'runs': {'upward': [], 'downward': [4]}
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
            'runs': {'upward': [1], 'downward': [1]}
        },
        {
            'name': 'NaN Breaks Runs',
            'df': make_df([10, 11, 12, np.nan, 11, 12, 11, 10]),
            # diffs: +1, +1, NaN, NaN, +1, -1, -1
            'runs': {'upward': [3], 'downward': [2]}
        },
        {
            'name': 'Short Series (len=1)',
            'df': make_df([100]),
            'runs': {'upward': [], 'downward': []}
        },
    ]

    for case in test_set:
        name = case['name']
        df = case['df']
        expected = case['runs']

        actual = implemented_count_runs(df)
        if actual != expected:
            details.append([name, expected, actual])
    if len(details) == 0:
        return True, []
    else:
        return False, details

def validate_sma_test_cases(df, window, implemented_sma):
    """
    Validates our own implemented_sma against data shorter than window and Nan in data

    Args:
    df (pd.DataFrame): Stock dataset
    window (int): Number of days to average over
    implemented_sma (pd.DataFrame): SMA of our own calculated SMA

    Returns:
    passed: bool, details: list
    """

    def equal_with_nan(a, b):
        """
        Robust equality for two lists where:
          - NaNs in the same position are considered equal,
          - None and NaN are considered equivalent (common for 'undefined' SMA slots),
          - Otherwise uses normal equality.
        """
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            x_is_nan = isinstance(x, float) and np.isnan(x)
            y_is_nan = isinstance(y, float) and np.isnan(y)
            # Treat NaN == NaN
            if x_is_nan and y_is_nan:
                continue
            # Treat None and NaN as equivalent for SMA "not defined yet"
            if (x is None and y_is_nan) or (y is None and x_is_nan):
                continue
            if x != y:
                return False
        return True

    details = []
    # Case 1: Data shorter than window
    prices_short = [100, 101]  # len=2 < window
    expected = [None, None]
    actual = implemented_sma(prices_short, window)
    if not equal_with_nan(expected, actual):
        details.append({
            "case": "Data shorter than window",
            "window": window,
            "expected": expected,
            "actual": actual,
        })

    #Case 2: Data with nan
    prices_with_nan = [10, 11, float('nan'), 14, 15]
    expected = [None, None, np.nan, np.nan, np.nan]
    actual = implemented_sma(prices_with_nan, window)
    if not equal_with_nan(expected, actual):
        details.append({
            "case": "Data with NaN",
            "window": window,
            "expected": expected,
            "actual": actual,
        })
    if len(details) == 0:
        passed = True
    else:
        passed = False
    return passed, details


def run_all_validation(csv_path: str, window: int, validators: list):
    """
    General validation runner that collects structured results.
    """
    print(f"\nRunning validations (window={window}) on csv path: {csv_path}")
    results: list[TestResult] = []

    for name, fn in validators:
        try:
            passed, details = fn()
        except Exception as e:
            results.append(TestResult(name=name, passed=False, details=f"Exception: {e}"))
        else:
            results.append(TestResult(name=name, passed=passed, details=details))

    # summary
    print("\nSummary")
    for r in results:
        status = " PASS" if r.passed else " FAIL"
        print(f"{status} — {r.name}")
        if not r.passed and r.details:
            print("   Details:", r.details)

    total = len(results)
    failures = sum(not r.passed for r in results)
    print(f"\nCompleted {total} test(s). {failures} failure(s).")


def main():
    path = os.path.join(os.getcwd(), "stock_data.csv") #Get csv file path that is default in folder for testing
    window = 3 #SMA window
    df = pd.read_csv(path)
    validators = [
        ("SMA vs pandas", lambda: validate_sma(df, window, compute_sma)),
        ("SMA edge cases", lambda: validate_sma_test_cases(df, window, compute_sma)),    
        ("Daily returns", lambda: validate_daily_returns(df, compute_daily_returns)),     
        ("Max profit", lambda: validate_max_profit(max_profit)),                   
        ("Count runs", lambda: validate_count_runs(df, count_runs)),                    
    ]

    run_all_validation(path, window, validators)


if __name__ == "__main__":
    main()
