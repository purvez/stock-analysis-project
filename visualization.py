import plotly.graph_objects as go
import pandas as pd

def plot_price_with_sma(df, sma):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                             mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=sma,
                             mode='lines', name='SMA'))
    fig.update_layout(title="Closing Price vs SMA", xaxis_title="Date", yaxis_title="Price")
    return fig

def highlight_runs(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                             mode='lines', name='Closing Price'))

    up_dates = []
    up_prices = []
    down_dates = []
    down_prices = []
    flat_dates = []
    flat_prices = []

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            up_dates.append(df['Date'].iloc[i])
            up_prices.append(df['Close'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            down_dates.append(df['Date'].iloc[i])
            down_prices.append(df['Close'].iloc[i])
        else:
            flat_dates.append(df['Date'].iloc[i])
            flat_prices.append(df['Close'].iloc[i])  

    fig.add_trace(go.Scatter(
        x=down_dates, y=down_prices,
        mode='markers', marker=dict(color='red', size=4),
        name='Down'
    ))

    fig.add_trace(go.Scatter(
        x=up_dates, y=up_prices,
        mode='markers', marker=dict(color='green', size=4),
        name='Up'
    ))

    fig.add_trace(go.Scatter(
        x=flat_dates, y=flat_prices,
        mode='markers', marker=dict(color='gray', size=4),
        name='Flat'
    ))


    fig.update_layout(title="Upward and Downward Runs", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_actual_vs_predicted(dates, actual, predicted):
    """Plot actual vs predicted closing prices."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual,
                             mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=dates, y=predicted,
                             mode='lines', name='Predicted Close'))
    fig.update_layout(title="Actual vs Predicted Closing Prices",
                      xaxis_title="Date", yaxis_title="Price")
    return fig

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
