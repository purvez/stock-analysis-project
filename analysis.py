import pandas as pd

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
