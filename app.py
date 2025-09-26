import streamlit as st
import pandas as pd
from analysis import compute_sma, compute_daily_returns, count_runs, max_profit
from visualization import plot_price_with_sma, highlight_runs

st.title("📈 Stock Market Trend Analysis")

# File upload
uploaded_file = st.file_uploader("Upload Stock CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    st.write("### Raw Data", df.head())

    # SMA Calculation
    window = st.slider("Select SMA Window", 2, 50, 5)
    sma = compute_sma(df['Close'].tolist(), window)
    df['SMA'] = sma

    st.plotly_chart(plot_price_with_sma(df, df['SMA']))

    # Daily Returns
    df = compute_daily_returns(df)
    st.write("### Daily Returns", df[['Date', 'Return']].head())

    # Runs
    runs = count_runs(df)
    st.write("### Runs", runs)
    st.plotly_chart(highlight_runs(df, runs))

    # Max Profit
    profit = max_profit(df['Close'].tolist())
    st.success(f"💰 Maximum Profit (multiple transactions): {profit:.2f}")
