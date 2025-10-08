import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

def compute_sma(prices: list, window: int) -> list:
    """
    Compute SMA using sliding window for efficiency.
    
    Parameters
    prices (list): Stocks closing prices in a list format
    window (int): number of days to calculate SMA

    Returns
    sma (list): list of the calculated SMA
    
    Raises
    TypeError: If prices is not a list or window is not an integer
    ValueError: If window is not positive or prices contain non-numeric values
    """
    # Validate prices parameter type
    if not isinstance(prices, list):
        raise TypeError(f"prices must be a list, got {type(prices).__name__}")
    
    # Handle empty list
    if not prices:
        return []
    
    # Validate window parameter type
    if not isinstance(window, int):
        raise TypeError(f"window must be an integer, got {type(window).__name__}")
    
    # Validate window is positive
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    
    # Validate all prices are numeric (int or float, including NaN)
    for i, price in enumerate(prices):
        if not isinstance(price, (int, float)):
            raise ValueError(f"prices[{i}] must be numeric (int or float), got {type(price).__name__}: {price}")
        # Note: We allow NaN values as they're valid floats and handled by the algorithm
    
    # If window is larger than prices, return all None
    if window > len(prices):
        return [None] * len(prices)
    
    # Function body(above validations passed)
    sma = []
    window_sum = 0
    for i in range(len(prices)):
        window_sum += prices[i]
        if i >= window:
            window_sum -= prices[i - window]
        if i >= window - 1:
            sma.append(window_sum / window)
        else:
            sma.append(None)  # Not enough data yet
    return sma


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns as percentage change in close price. 

    Parameters
    df (pd.DataFrame): Stocks data frame

    Returns
    pd.DataFrame: Stocks data frame with new column "Return"

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If 'Close' column is missing.
    ValueError
        If 'Close' has no numeric values after coercion.
    """
    #Input validation check
    out = df.copy() #Ensure that df is not changed when doing input validation since df is a mutable variabel type
    #Input validation: Check if df is pandas.dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    #Input validation: Check if df has the column Close
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column")
    #Input validation: Check if Close column contains numeric value
    out['Close'] = pd.to_numeric(out['Close'], errors='coerce')
    if out['Close'].isna().all():
        raise ValueError("'Close' column contains no numeric values")
    
    #Calculation of daily returns
    prices = df["Close"]
    df["Return"] = ((prices - prices.shift(1)) / prices.shift(1))
    return df

def daily_returns_stats(df: pd.DataFrame) -> pd.Series:
    """
    Compute summary statistics: Mean, Median, Standard Deviation, Minimum, Maximum for daily returns

    Parameters
    df (pd.DataFrame): data frame

    Returns
    pd.Series: Panda series of the various stats
    """
    summary = df['Return'].describe()[['mean', '50%', 'std', 'min', 'max']]
    summary.index = ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum']
    return summary


def count_runs(df: pd.DataFrame) -> dict:
    """
    Identify upward and downward runs.

    Parameters
    df (pd.DataFrame): data frame

    Returns
    dict: count of upwards and downwards runs 
    """
    runs = {'upward': [], 'downward': []}
    current_run = 0
    direction = None
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            if direction == 'upward':
                current_run += 1
            else:
                if direction == 'downward':
                    runs['downward'].append(current_run)
                direction = 'upward'
                current_run = 1
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            if direction == 'downward':
                current_run += 1
            else:
                if direction == 'upward':
                    runs['upward'].append(current_run)
                direction = 'downward'
                current_run = 1
    if direction:
        runs[direction].append(current_run)
    return runs

def max_profit(prices: list) -> float:
    """
    Leetcode Best Time to Buy and Sell Stock II.

    Parameters
    prices (list): list of closing prices

    Returns
    profit (float): max profit
    """
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

def compute_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def train_and_predict(df, threshold=0.005):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Technical indicators and engineered features
    df['SMA_5'] = compute_sma(df['Close'].tolist(), 5)
    df['SMA_20'] = compute_sma(df['Close'].tolist(), 20)
    df['EMA_5'] = compute_ema(df['Close'], 5)
    df['EMA_20'] = compute_ema(df['Close'], 20)
    df['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Rolling_Max_10'] = df['Close'].rolling(10).max()
    df['Rolling_Min_10'] = df['Close'].rolling(10).min()
    df = compute_daily_returns(df)
    for lag in range(1, 11):
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

    # Next-day returns as target
    df['Target'] = df['Return'].shift(-1)

    # Drop NA rows
    df = df.dropna().reset_index(drop=True)

    features = [
        'Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20',
        'Volatility_10', 'Momentum_10', 'Rolling_Max_10', 'Rolling_Min_10', 'Return'
    ] + [f'Close_Lag{i}' for i in range(1, 11)] + [f'Return_Lag{i}' for i in range(1, 11)]

    X = df[features]
    y = df['Target']

    tscv = TimeSeriesSplit(n_splits=5)
    # Store all test set predictions for plotting across all folds
    test_dates_all = []
    y_test_all = []
    y_pred_all = []

    model = XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.01, subsample=0.8,
        colsample_bytree=0.8, gamma=0.01, random_state=42, verbosity=0
    )

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_dates_all.append(df['Date'].iloc[test_idx])
        y_test_all.append(y_test)
        y_pred_all.append(y_pred)

    test_dates_all = pd.concat(test_dates_all)
    y_test_all = np.concatenate(y_test_all)
    y_pred_all = np.concatenate(y_pred_all)

    # Evaluate using last fold
    last_X_train, last_X_test = X.iloc[train_idx], X.iloc[test_idx]
    last_y_train, last_y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(last_X_train, last_y_train)
    y_pred_last = model.predict(last_X_test)
    mae = mean_absolute_error(last_y_test, y_pred_last)
    mse = mean_squared_error(last_y_test, y_pred_last)
    r2 = r2_score(last_y_test, y_pred_last)

    # Next-day return prediction -> convert to price
    last_row = df.iloc[[-1]][features]
    predicted_return = model.predict(last_row)[0]
    current_close = df.iloc[-1]['Close']
    next_day_prediction = current_close * (1 + predicted_return)

    diff = next_day_prediction - current_close
    pct_change = diff / current_close
    if pct_change > threshold:
        signal = "BUY"
    elif pct_change < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "next_day_prediction": next_day_prediction,
        "current_close": current_close,
        "signal": signal,
        "pct_change": pct_change,
        "test_dates": test_dates_all,
        "y_test": y_test_all,      # actual returns, test set
        "y_pred": y_pred_all       # predicted returns, test set
    }


