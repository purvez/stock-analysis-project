import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_sma(prices, window):
    """Compute SMA using sliding window for efficiency."""
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

def compute_daily_returns(df):
    """Compute daily returns as percentage change in close price."""
    df['Return'] = df['Close'].pct_change()
    return df

def count_runs(df):
    """Identify upward and downward runs."""
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

def max_profit(prices):
    """Leetcode Best Time to Buy and Sell Stock II."""
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

def train_and_predict(df, threshold=0.005):
     # Strip column names (remove hidden spaces or carriage returns)
    df.columns = df.columns.str.strip()

    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Features and target
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features]
    y = df['Close']

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Predict next-day using last row
    last_row = df.iloc[-1]
    next_day_features = pd.DataFrame([{
        'Open': last_row['Open'],
        'High': last_row['High'],
        'Low': last_row['Low'],
        'Volume': last_row['Volume']
    }])[features]

    next_day_prediction = model.predict(next_day_features)[0]
    current_close = last_row['Close']

    # Generate signal
    diff = next_day_prediction - current_close
    pct_change = diff / current_close

    if pct_change > threshold:
        signal = "BUY"
    elif pct_change < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Return all needed info for Streamlit + plotting
    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "next_day_prediction": next_day_prediction,
        "current_close": current_close,
        "signal": signal,
        "pct_change": pct_change,
        "y_test": y_test.values,    # for plotting
        "y_pred": y_pred,           # for plotting
        "test_dates": df['Date'].iloc[test_idx]  # for plotting
    }
