import streamlit as st
import pandas as pd
from data_cleaning import *
from analysis import *
from visualization import *
import numpy as np
from import_data import StockDataLoader

#Init to keep data
if "df" not in st.session_state:
    st.session_state.df = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "ticker" not in st.session_state:
    st.session_state.ticker = "NVDA"

def fetch_ticker_df(symbol: str) -> pd.DataFrame:
    """
    Loads the class StockDataLoader to load data when user inputs ticker symbol

    Args:
    symbol (str): symbol for ticker

    Return:
    pd.DataFrame: Stock data

    """
    loader = StockDataLoader(tickers=symbol)  
    return loader.load()

def run_pipeline(df):
    """
    Runs the core functunalities
    """
    df.columns = df.columns.str.strip() # Strip any hidden spaces or carriage returns from column names
    # Ensure 'Date' exists
    if 'Date' not in df.columns:
        st.error("CSV must contain a 'Date' column")
    else:
        df = clean_data(df) #Clean data

        st.write("### Raw Data", df.head())

        # SMA Calculation
        window = st.slider("Select SMA Window", 2, 50, 5)
        sma = compute_sma(df['Close'].tolist(), window)
        df['SMA'] = sma

        st.plotly_chart(plot_price_with_sma(df, df['SMA']))

        # Daily Returns
        df = compute_daily_returns(df)
        st.subheader("Daily returns")
        summary = daily_returns_stats(df)
        st.write(summary)

        # Runs
        runs = count_runs(df)
        st.write("### Runs", runs)
        st.plotly_chart(highlight_runs(df))

        # Max Profit
        profit = max_profit(df['Close'].tolist())
        st.success(f"💰 Maximum Profit (multiple transactions): {profit:.2f}")

        # --- ML Prediction ---
        results = train_and_predict(df)

        prev_closes = []
        test_date_indices = results['test_dates'].index if isinstance(results['test_dates'], pd.Series) else np.arange(len(results['test_dates']))
        for idx in test_date_indices:
            prev_idx = idx - 1 if idx - 1 >= 0 else 0
            prev_closes.append(df['Close'].iloc[prev_idx])
        prev_closes = np.array(prev_closes)

        # Calculate actual and predicted price series
        actual_prices = prev_closes * (1 + results['y_test'])
        predicted_prices = prev_closes * (1 + results['y_pred'])

        st.write("### Model Evaluation")
        st.metric("MAE", f"{results['mae']:.4f}")
        st.metric("MSE", f"{results['mse']:.4f}")
        st.metric("R²", f"{results['r2']:.4f}")

        st.write("### Next-Day Prediction")
        st.metric("Predicted Close", f"{results['next_day_prediction']:.2f}")
        st.metric("Current Close", f"{results['current_close']:.2f}")
        st.metric("Signal", results['signal'])
        st.metric("Expected Change (%)", f"{results['pct_change']*100:.2f}%")

        # --- Optional: Actual vs Predicted ---
        from visualization import plot_actual_vs_predicted
        st.plotly_chart(plot_actual_vs_predicted(
            results['test_dates'],  # Dates of test set
            actual_prices,          # Actual closes
            predicted_prices        # Predicted closes
        ))

st.title("📈 Stock Market Trend Analysis")

# File upload
source = st.radio("Choose data source", ["Upload CSV", "Fetch by Ticker"], horizontal=True) #UI for user to choose data source, csv or ticker symbol

# Reset df when switching modes
if source != st.session_state.mode:
    st.session_state.mode = source
    st.session_state.df = None

if source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Stock CSV", type="csv") 
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file) #load the data
            st.success(f"Loaded CSV: {len(st.session_state.df)} rows.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

else:
    # Use session-state default for convenience
    st.session_state.ticker = st.text_input("Ticker", value=st.session_state.ticker, placeholder="e.g., AAPL")
    if st.button("Load data"):
        with st.spinner(f"Fetching {st.session_state.ticker}"):
            try:
                df = fetch_ticker_df(st.session_state.ticker)
                if df is None or df.empty:
                    st.warning("No data returned. Check the ticker symbol.")
                else:
                    st.session_state.df = df  #load the data
                    st.success(f"Loaded {st.session_state.ticker} data: {len(df)} rows.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

# Always render pipeline if we have data in session. Whenever there is a widget change, streamlit reruns, thus we keep a session state such that for every rerun, we can reuse it, instead of asking the user to input the stock ticker again
if st.session_state.df is not None:
    run_pipeline(st.session_state.df)
