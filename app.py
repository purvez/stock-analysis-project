import streamlit as st
import pandas as pd
from data_cleaning import *
from analysis import *
from visualization import *
import numpy as np

st.title("📈 Stock Market Trend Analysis")

# File upload
uploaded_file = st.file_uploader("Upload Stock CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
     # Strip any hidden spaces or carriage returns from column names
    df.columns = df.columns.str.strip()

    # Ensure 'Date' exists
    if 'Date' not in df.columns:
        st.error("CSV must contain a 'Date' column")
    else:
        clean_data(df)

        st.write("### Raw Data", df.head())
        # Sidebar date filter
        # st.sidebar.header("Filter by Date")
        # start_date = st.sidebar.date_input("Start Date", df['Date'].min())
        # end_date = st.sidebar.date_input("End Date", df['Date'].max())

        # Filter data
        # mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        # df = df.loc[mask]

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
        st.plotly_chart(highlight_runs(df, runs))

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
