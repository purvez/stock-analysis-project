import streamlit as st
import pandas as pd
from analysis import compute_sma, compute_daily_returns, count_runs, max_profit, train_and_predict
from visualization import plot_price_with_sma, highlight_runs, plot_actual_vs_predicted

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
        # Convert 'Date' to datetime and sort
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

        # --- ML Prediction ---
        results = train_and_predict(df)

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
            results['y_test'],      # Actual closes
            results['y_pred']       # Predicted closes
        ))
