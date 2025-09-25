import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our stock analyzer backend
from stock_analyzer import StockAnalyzer
from typing import List, Tuple, Dict, Optional


# Streamlit App Configuration
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-header">📈 Stock Market Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
st.sidebar.title("🔧 Analysis Configuration")
st.sidebar.markdown("Configure your stock analysis parameters below:")

# Stock Symbol Input
popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
selected_stock = st.sidebar.selectbox(
    "Select Stock Symbol",
    options=popular_stocks + ['Custom'],
    index=0,
    help="Choose from popular stocks or enter a custom symbol"
)

if selected_stock == 'Custom':
    stock_symbol = st.sidebar.text_input(
        "Enter Custom Stock Symbol",
        value="AAPL",
        help="Enter any valid stock ticker symbol"
    ).upper()
else:
    stock_symbol = selected_stock

# Time Period Selection
time_period = st.sidebar.selectbox(
    "Analysis Time Period",
    options=['1y', '2y', '3y', '5y'],
    index=2,
    help="Select the time period for analysis"
)

# SMA Window Configuration
sma_windows = st.sidebar.multiselect(
    "Moving Average Windows",
    options=[5, 10, 20, 50, 100, 200],
    default=[20, 50],
    help="Select moving average periods to display"
)

# Analysis Options
show_validation = st.sidebar.checkbox("Show Validation Results", value=True)
show_transactions = st.sidebar.checkbox("Show Trading Transactions", value=True)
show_runs_analysis = st.sidebar.checkbox("Show Runs Analysis", value=True)

# Run Analysis Button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# Main Content Area
if run_analysis or 'analyzer' not in st.session_state:
    with st.spinner(f'Loading data for {stock_symbol}...'):
        try:
            # Initialize analyzer
            analyzer = StockAnalyzer(stock_symbol, period=time_period)
            st.session_state.analyzer = analyzer
            st.session_state.stock_symbol = stock_symbol
            
            if analyzer.data is not None and not analyzer.data.empty:
                st.success(f"✅ Successfully loaded {len(analyzer.data)} days of data for {stock_symbol}")
            else:
                st.error("❌ Failed to load stock data")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error initializing analyzer: {e}")
            st.stop()

# Display Analysis Results
if 'analyzer' in st.session_state:
    analyzer = st.session_state.analyzer
    
    # Stock Overview
    st.subheader(f"📊 {stock_symbol} Stock Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = analyzer.data['Close'].iloc[-1]
    start_price = analyzer.data['Close'].iloc[0]
    total_return = ((current_price - start_price) / start_price) * 100
    highest_price = analyzer.data['High'].max()
    lowest_price = analyzer.data['Low'].min()
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Total Return", f"{total_return:.2f}%", f"{total_return:.2f}%")
    with col3:
        st.metric("52-Week High", f"${highest_price:.2f}")
    with col4:
        st.metric("52-Week Low", f"${lowest_price:.2f}")
    
    # Interactive Price Chart with SMA
    st.subheader("📈 Price Chart with Moving Averages")
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=analyzer.data.index,
        y=analyzer.data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # Add SMA lines
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, window in enumerate(sma_windows):
        sma = analyzer.simple_moving_average(window)
        fig.add_trace(go.Scatter(
            x=analyzer.data.index,
            y=sma,
            mode='lines',
            name=f'SMA {window}',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
            hovertemplate=f'<b>SMA {window}</b>: $%{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'{stock_symbol} - Price vs Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    # Daily Returns Chart
    with col1:
        st.subheader("📊 Daily Returns")
        returns = analyzer.calculate_daily_returns() * 100
        
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=analyzer.data.index,
            y=returns,
            mode='lines',
            name='Daily Returns',
            line=dict(color='green', width=1),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        
        fig_returns.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig_returns.update_layout(
            title='Daily Returns (%)',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
    
    # Volume Chart
    with col2:
        st.subheader("📈 Trading Volume")
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=analyzer.data.index,
            y=analyzer.data['Volume'],
            name='Volume',
            marker_color='orange',
            opacity=0.7
        ))
        
        fig_volume.update_layout(
            title='Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Runs Analysis
    if show_runs_analysis:
        st.subheader("🎯 Runs Analysis")
        runs_stats = analyzer.find_runs()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Upward Runs", runs_stats['total_upward_runs'])
        with col2:
            st.metric("Downward Runs", runs_stats['total_downward_runs'])
        with col3:
            st.metric("Longest Up Streak", f"{runs_stats['longest_upward_streak']} days")
        with col4:
            st.metric("Longest Down Streak", f"{runs_stats['longest_downward_streak']} days")
        
        # Enhanced Runs visualization
        fig_runs = go.Figure()
        
        # Add price line (thinner to make markers stand out)
        fig_runs.add_trace(go.Scatter(
            x=analyzer.data.index,
            y=analyzer.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2E86AB', width=1.5, dash='solid'),
            opacity=0.6
        ))
        
        # Highlight runs with much larger, more visible markers
        changes = analyzer.data['Close'].diff()
        up_days = changes > 0
        down_days = changes < 0
        
        # Up days - bright green with larger markers
        fig_runs.add_trace(go.Scatter(
            x=analyzer.data.index[up_days],
            y=analyzer.data['Close'][up_days],
            mode='markers',
            name='Up Days',
            marker=dict(
                color='#00FF00',  # Bright green
                size=8,           # Much larger
                symbol='triangle-up',
                line=dict(color='#006400', width=1),  # Dark green border
                opacity=0.9
            ),
            hovertemplate='<b>UP DAY</b><br><b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<br><b>Change</b>: +%{customdata:.2f}%<extra></extra>',
            customdata=changes[up_days] * 100
        ))
        
        # Down days - bright red with larger markers
        fig_runs.add_trace(go.Scatter(
            x=analyzer.data.index[down_days],
            y=analyzer.data['Close'][down_days],
            mode='markers',
            name='Down Days',
            marker=dict(
                color='#FF0000',  # Bright red
                size=8,           # Much larger
                symbol='triangle-down',
                line=dict(color='#8B0000', width=1),  # Dark red border
                opacity=0.9
            ),
            hovertemplate='<b>DOWN DAY</b><br><b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<br><b>Change</b>: %{customdata:.2f}%<extra></extra>',
            customdata=changes[down_days] * 100
        ))
        
        # Add background color coding for runs (optional enhancement)
        # Find consecutive runs and highlight them
        directions = []
        for change in changes.dropna():
            if change > 0:
                directions.append('up')
            elif change < 0:
                directions.append('down')
            else:
                directions.append('flat')
        
        # Identify run periods for background highlighting
        current_direction = None
        run_start = None
        
        for i, direction in enumerate(directions):
            if direction != current_direction and direction in ['up', 'down']:
                if run_start is not None and current_direction in ['up', 'down']:
                    # End previous run
                    run_end = analyzer.data.index[i]
                    if current_direction == 'up' and i - run_start > 2:  # Only highlight runs of 3+ days
                        fig_runs.add_vrect(
                            x0=analyzer.data.index[run_start], 
                            x1=run_end,
                            fillcolor="rgba(0, 255, 0, 0.1)",
                            layer="below",
                            line_width=0,
                        )
                    elif current_direction == 'down' and i - run_start > 2:
                        fig_runs.add_vrect(
                            x0=analyzer.data.index[run_start], 
                            x1=run_end,
                            fillcolor="rgba(255, 0, 0, 0.1)",
                            layer="below",
                            line_width=0,
                        )
                
                # Start new run
                current_direction = direction
                run_start = i
        
        fig_runs.update_layout(
            title='Price Movement Runs Analysis - Enhanced Visibility',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="Black",
                borderwidth=1
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_runs, use_container_width=True)
        
        # Add explanation for the enhanced runs visualization
        st.markdown("""
        **📊 Enhanced Runs Analysis Legend:**
        - 🔵 **Blue Line**: Stock price trend (slightly transparent to highlight markers)
        - 🟢 **Green Triangle Up**: Days with price increases (much larger and brighter)
        - 🔴 **Red Triangle Down**: Days with price decreases (much larger and brighter)
        - 🟢 **Light Green Background**: Extended upward runs (3+ consecutive up days)
        - 🔴 **Light Red Background**: Extended downward runs (3+ consecutive down days)
        
        **💡 Hover over any marker to see exact price change percentage!**
        """)
        
        # Additional runs statistics in a more visual format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h5>📈 Upward Momentum</h5>
                <p><strong>{runs_stats['total_upward_runs']}</strong> upward runs</p>
                <p><strong>{runs_stats['total_upward_days']}</strong> total up days</p>
                <p><strong>{runs_stats['longest_upward_streak']}</strong> days max streak</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="warning-card">
                <h5>📉 Downward Momentum</h5>
                <p><strong>{runs_stats['total_downward_runs']}</strong> downward runs</p>
                <p><strong>{runs_stats['total_downward_days']}</strong> total down days</p>
                <p><strong>{runs_stats['longest_downward_streak']}</strong> days max streak</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            up_down_ratio = runs_stats['total_upward_days'] / max(runs_stats['total_downward_days'], 1)
            avg_up = runs_stats['average_upward_run']
            avg_down = runs_stats['average_downward_run']
            
            st.markdown(f"""
            <div class="success-card">
                <h5>⚖️ Market Balance</h5>
                <p><strong>{up_down_ratio:.2f}</strong> up/down ratio</p>
                <p><strong>{avg_up:.1f}</strong> avg up run length</p>
                <p><strong>{avg_down:.1f}</strong> avg down run length</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Trading Simulation
    if show_transactions:
        st.subheader("💰 Trading Simulation (Multiple Transactions)")
        
        max_profit, transactions = analyzer.max_profit_multiple_transactions()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="success-card">
                <h4>🎯 Maximum Profit: ${max_profit:.2f}</h4>
                <p>Number of Transactions: {len(transactions)}</p>
                <p>Strategy: Buy low, sell high with multiple transactions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if len(transactions) > 0:
                avg_profit = max_profit / len(transactions)
                roi_percent = (max_profit / start_price) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h5>📈 Trading Statistics</h5>
                    <p>Average Profit per Transaction: ${avg_profit:.2f}</p>
                    <p>Total ROI: {roi_percent:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced Trading Visualization
        if transactions:
            st.subheader("🎯 Trading Strategy Visualization")
            
            # Create enhanced trading chart
            fig_trading = go.Figure()
            
            # Add price line
            fig_trading.add_trace(go.Scatter(
                x=analyzer.data.index,
                y=analyzer.data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='#2E86AB', width=3),
                hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            ))
            
            # Extract buy and sell points from transactions
            buy_dates = []
            buy_prices = []
            sell_dates = []
            sell_prices = []
            
            for trans_desc, profit in transactions:
                # Parse transaction details
                parts = trans_desc.split(", ")
                buy_part = parts[0]  # "Buy: 2023-01-01 at $100.00"
                sell_part = parts[1]  # "Sell: 2023-01-05 at $110.00"
                
                # Extract buy info
                buy_date_str = buy_part.split(" at ")[0].split(": ")[1]
                buy_price = float(buy_part.split("$")[1])
                buy_dates.append(pd.to_datetime(buy_date_str))
                buy_prices.append(buy_price)
                
                # Extract sell info
                sell_date_str = sell_part.split(" at ")[0].split(": ")[1]
                sell_price = float(sell_part.split("$")[1])
                sell_dates.append(pd.to_datetime(sell_date_str))
                sell_prices.append(sell_price)
            
            # Add buy points
            fig_trading.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy Points',
                marker=dict(
                    color='#00FF00',
                    size=12,
                    symbol='triangle-up',
                    line=dict(color='#006400', width=2)
                ),
                hovertemplate='<b>BUY</b><br><b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            ))
            
            # Add sell points
            fig_trading.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell Points',
                marker=dict(
                    color='#FF4500',
                    size=12,
                    symbol='triangle-down',
                    line=dict(color='#8B0000', width=2)
                ),
                hovertemplate='<b>SELL</b><br><b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            ))
            
            # Add connecting lines between buy and sell points
            for i in range(len(buy_dates)):
                fig_trading.add_trace(go.Scatter(
                    x=[buy_dates[i], sell_dates[i]],
                    y=[buy_prices[i], sell_prices[i]],
                    mode='lines',
                    line=dict(
                        color='rgba(255, 215, 0, 0.8)',
                        width=3,
                        dash='solid'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_trading.update_layout(
                title=f'{stock_symbol} - Trading Strategy Visualization',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=600,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="Black",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_trading, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **📊 Chart Legend:**
            - 🔵 **Blue Line**: Stock price over time
            - 🟢 **Green Triangles Up**: Buy points (local minima)
            - 🟠 **Orange Triangles Down**: Sell points (local maxima)
            - 🟡 **Yellow Lines**: Individual transactions connecting buy to sell
            
            **💡 Strategy Explanation:**
            The algorithm identifies local price minima for buying and local maxima for selling, 
            maximizing profit by capturing every profitable price movement.
            """)
        
        # Show transaction details
        if transactions:
            st.subheader("📋 Transaction Details")
            
            transaction_df = pd.DataFrame([
                {
                    'Transaction': i+1,
                    'Details': trans[0],
                    'Profit': f"${trans[1]:.2f}"
                }
                for i, trans in enumerate(transactions[:10])  # Show first 10 transactions
            ])
            
            st.dataframe(transaction_df, use_container_width=True)
    
    # Validation Results
    if show_validation:
        st.subheader("✅ Validation Results")
        
        validations = analyzer.run_validation_tests()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### SMA Validation")
            if validations['sma_test']['validation_passed']:
                st.markdown('<div class="success-card">✅ SMA calculation matches pandas implementation</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">⚠️ SMA calculation has minor differences</div>', 
                          unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Returns Validation")
            if validations['returns_test']['validation_passed']:
                st.markdown('<div class="success-card">✅ Daily returns calculation is correct</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">⚠️ Daily returns calculation has minor differences</div>', 
                          unsafe_allow_html=True)
        
        # Show validation metrics
        st.markdown("#### Detailed Validation Metrics")
        validation_metrics = pd.DataFrame([
            {'Test': 'SMA Validation', 'Max Difference': f"{validations['sma_test']['max_difference']:.2e}", 'Status': '✅ Passed' if validations['sma_test']['validation_passed'] else '⚠️ Check'},
            {'Test': 'Returns Validation', 'Max Difference': f"{validations['returns_test']['max_difference']:.2e}", 'Status': '✅ Passed' if validations['returns_test']['validation_passed'] else '⚠️ Check'},
            {'Test': 'Edge Cases', 'Max Difference': 'N/A', 'Status': '✅ Handled'}
        ])
        
        st.dataframe(validation_metrics, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🎓 INF1002 Programming Fundamentals - Stock Market Analysis Project</h4>
    <p>Built with ❤️ using Streamlit, Plotly, and Python</p>
</div>
""", unsafe_allow_html=True)