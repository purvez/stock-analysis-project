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
