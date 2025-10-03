import plotly.graph_objects as go

def plot_price_with_sma(df, sma):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                             mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=sma,
                             mode='lines', name='SMA'))
    fig.update_layout(title="Closing Price vs SMA", xaxis_title="Date", yaxis_title="Price")
    return fig

def highlight_runs(df, runs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                             mode='lines', name='Closing Price'))
    colors = []
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            colors.append("green")
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            colors.append("red")
        else:
            colors.append("gray")
    fig.add_trace(go.Scatter(x=df['Date'][1:], y=df['Close'][1:], 
                             mode='markers', marker=dict(color=colors, size=8),
                             name="Up/Down Runs"))
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