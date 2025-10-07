import pandas as pd
import numpy as np
import os
from analysis import *
from dataclasses import dataclass
from typing import Any, Callable, Iterable

@dataclass
class TestResult:
    name: str
    passed: bool
    details: any = None

def equal_with_nan(a: Iterable[Any], b: Iterable[Any]) -> bool:
    """
    NaNs in the same position are considered equal Nan == Nan
    None and NaN are considered equivalent, None == Nan
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


def validate_sma(df: pd.DataFrame, window:int, implemented_sma: Callable) ->tuple[bool, list]:
    """
    Validates our own implemented_sma against pandas built-in rolling mean.

    Args:
    df (pd.DataFrame): Stock dataset
    window (int): Number of days to average over
    implemented_sma (callable): SMA of our own calculated SMA

    Returns:
    passed (bool): whether test cased pass
    details (bool): list of mismatch with index and values of expected and actual SMA
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

def validate_sma_edge_cases(window: int, implemented_sma: Callable) ->tuple[bool, list]:
    """
    Validates our own implemented_sma against edge cases:
    1) Data shorter than window
    2) Data with NaN
    3) Data with same constant prices

    Args:
    window (int): Number of days to average over
    implemented_sma (Callable): SMA of our own calculated SMA

    Returns:
    passed (bool): whether test cased pass
    details (bool): list of mismatch with index and values of expected and actual SMA
    """

    details = []

    def case_checker(data: list, expected: list, label: str):
        """
        Runs each test case and updates details(list) if there are any erros.

        Args: 
        data(list): test case data
        expected(list): expected answers based on data
        label(str): name of test case

        """
        actual = implemented_sma(data, window)
        if not equal_with_nan(expected, actual):
            details.append({
                "case": label,
                "window": window,
                "expected": expected,
                "actual": actual,
            })
    
    # Case 1: Data shorter than window
    label = "Shorter than window"
    prices_short = [100, 101]  # len=2 < window
    expected = [None, None]
    case_checker(prices_short, expected, label)

    #Case 2: Data with nan
    label = "Data with NaN"
    prices_with_nan = [10, 11, float('nan'), 14, 15]
    expected = [None, None, np.nan, np.nan, np.nan]
    case_checker(prices_with_nan, expected, label)

    #Case 3: Data with constant prices
    label = "Constant prices"
    prices_constant = [5,5,5,5]
    expected = [None, None, 5.0, 5.0]
    case_checker(prices_constant, expected, label)

    return (len(details) == 0), details


def validate_daily_returns(df: pd.DataFrame, implemented_daily_returns: Callable, price_col: str="Close", return_col: str="Return", tol: float=1e-8) ->tuple[bool, list]:
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
    #Deep copy df since its mutable
    df_copy = df.copy(deep=True)

    df_copy = implemented_daily_returns(df_copy) #Creates the column returns 
    pandas_returns = df_copy [price_col].pct_change()

    comparison = pd.DataFrame({
        "Custom": df_copy [return_col].dropna(),
        "Pandas": pandas_returns.dropna(),
    })
    comparison["Diff"] = comparison["Custom"] - comparison["Pandas"]
    comparison["Match"] = comparison["Diff"].abs() <= tol

    if comparison["Match"].all():
        return True, []
    else:
        return False, comparison[~comparison["Match"]].to_dict(orient="records")

def validate_max_profit(implemented_max_profit:Callable)->tuple[bool, list]:
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

def validate_count_runs(df: pd.DataFrame, implemented_count_runs: Callable)->tuple[bool, list]:
    """
    Validates our own implemented count runs against test case

    Args:
    implemented_count_runs(df:pd.Dataframe): Own implemented function to count runs

    Returns:
    passed: bool, details: list
    """
    details = []
    def make_df(values: list)-> pd.DataFrame: 
        """
        Creates a df with the column close and values of the test case to compare with implemented count runs
        
        Args:
        values(list): values for test cases

        Returns:
        pd.DataFrame: df with column close and values
        """
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

    return (len(details) == 0), details




def run_all_validation(csv_path: str, window: int, validators: list):
    """
    General validation runner that collects results for each test case.
    """
    print(f"\nRunning validations on csv path: {csv_path}")
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
        ("SMA edge cases", lambda: validate_sma_edge_cases(window, compute_sma)),    
        ("Daily returns", lambda: validate_daily_returns(df, compute_daily_returns)),     
        ("Max profit", lambda: validate_max_profit(max_profit)),                   
        ("Count runs", lambda: validate_count_runs(df, count_runs)),                    
    ]

    run_all_validation(path, window, validators)


if __name__ == "__main__":
    main()
