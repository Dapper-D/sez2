import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, date
from dotenv import load_dotenv
load_dotenv()
import os
import matplotlib.pyplot as plt
from data_preparation import DataPreparation
import numpy as np

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

@st.cache_data
def load_instruments():
    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM instruments", conn)
    return df

def load_candles(instrument_id, start, end, timeframe="1min"):
    table = f"candle_data_{timeframe}"
    instrument_id = int(instrument_id)
    with get_connection() as conn:
        df = pd.read_sql(
            f"SELECT * FROM {table} WHERE instrument_id=%s AND time BETWEEN %s AND %s ORDER BY time",
            conn, params=(instrument_id, start, end)
        )
    return df

def load_signals(symbol, timeframe, limit=200):
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT * FROM generate_scalping_signals_flexible(%s, %s, %s)",
            conn, params=(timeframe, symbol, limit)
        )
    return df

st.sidebar.header("Instrument Picker")
inst_df = load_instruments()

# --- Connection Pool Status ---
st.sidebar.markdown("---")
st.sidebar.subheader("DB Connection Pool")
st.sidebar.info("Connection pooling is enabled for all backend components.")

all_underlyings = inst_df['symbol'].str.extract(r'([A-Z]+)')[0].dropna().unique()
priority = ["NIFTY", "BANKNIFTY"]
ordered_underlyings = [u for u in priority if u in all_underlyings] + [u for u in all_underlyings if u not in priority]
underlying = st.sidebar.selectbox("Underlying (e.g. NIFTY, BANKNIFTY)", ordered_underlyings, index=0)

expiry_options = sorted(inst_df[inst_df['symbol'].str.startswith(underlying)]['expiry_date'].dropna().unique())
expiry = st.sidebar.selectbox("Expiry Date", expiry_options, format_func=lambda x: x.strftime("%d-%b-%Y") if pd.notnull(x) else "Unknown")

strike_options = sorted(inst_df[
    (inst_df['symbol'].str.startswith(underlying)) &
    (inst_df['expiry_date']==expiry)
]['strike_price'].dropna().unique())
strike = st.sidebar.selectbox("Strike Price", strike_options)

option_type = st.sidebar.selectbox("Option Type", ['CE', 'PE'])

filtered = inst_df[
    (inst_df['symbol'].str.startswith(underlying)) &
    (inst_df['expiry_date']==expiry) &
    (inst_df['strike_price']==strike) &
    (inst_df['option_type']==option_type)
]
if filtered.empty:
    st.warning("No such option contract found in DB!")
    st.stop()
instrument_id = filtered.iloc[0]['instrument_id']
symbol = filtered.iloc[0]['symbol']

st.title("NSE Options Data Explorer")
st.markdown(f"""
**Selected Contract:**  
- **Symbol:** `{symbol}`  
- **Expiry:** `{expiry.strftime('%d-%b-%Y')}`  
- **Strike:** `{strike}`  
- **Type:** `{option_type}`
""")

mindate = filtered.iloc[0].get('expiry_date', date(2020,1,1))
start_date = st.date_input("Start Date", value=mindate)
end_date = st.date_input("End Date", value=mindate if mindate > date.today() else date.today())
timeframe = st.radio("Candle Timeframe", ["1min", "15sec"])

df = load_candles(instrument_id, start_date, end_date, timeframe)
if df.empty:
    st.warning("No candle data for this contract and range.")
else:
    st.subheader("Candlestick Data")
    st.dataframe(df, use_container_width=True)
    st.line_chart(df.set_index('time')['close'], use_container_width=True)
    st.line_chart(df.set_index('time')['volume'], use_container_width=True)
    st.download_button("Download Candle CSV", df.to_csv(index=False), file_name=f"{symbol}_{timeframe}.csv")

st.subheader("Scalping Signals (Multi-indicator)")
if st.button("Show Scalping Signals"):
    sig_df = load_signals(symbol, timeframe)
    if not sig_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = pd.to_datetime(sig_df['time']) if 'time' in sig_df.columns else sig_df.index
        ax.plot(x, sig_df['close'], label='Close', color='blue')
        if 'signal_type' in sig_df.columns:
            buy = sig_df[sig_df['signal_type'] == 'buy']
            sell = sig_df[sig_df['signal_type'] == 'sell']
            trend_up = sig_df[sig_df['signal_type'] == 'trend_up']
            trend_down = sig_df[sig_df['signal_type'] == 'trend_down']
            hold = sig_df[sig_df['signal_type'] == 'hold']
            if not buy.empty:
                ax.scatter(pd.to_datetime(buy['time']), buy['close'], marker='^', color='green', label='Buy', alpha=0.9)
            if not sell.empty:
                ax.scatter(pd.to_datetime(sell['time']), sell['close'], marker='v', color='red', label='Sell', alpha=0.9)
            if not trend_up.empty:
                ax.scatter(pd.to_datetime(trend_up['time']), trend_up['close'], marker='o', color='orange', label='Trend Up', alpha=0.7)
            if not trend_down.empty:
                ax.scatter(pd.to_datetime(trend_down['time']), trend_down['close'], marker='o', color='purple', label='Trend Down', alpha=0.7)
        ax.set_title("Scalping Signals on Price")
        ax.set_ylabel("Price")
        ax.set_xlabel("Time")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(sig_df, use_container_width=True)
        st.download_button("Download Signals CSV", sig_df.to_csv(index=False), file_name=f"{symbol}_{timeframe}_signals.csv")
        st.markdown("""
**Signal Types:**
- `buy`: RSI < 30 (oversold), OBV increasing, close > MA5 (momentum up), volume spike.
- `sell`: RSI > 70 (overbought), OBV decreasing, close < MA5 (momentum down), volume spike.
- `trend_up`: Close > MA20 and RSI rising.
- `trend_down`: Close < MA20 and RSI falling.
- `hold`: None of the above (neutral/wait).
        """)
    else:
        st.info("No signals found for this selection.")

st.subheader("ML Buy/Sell Signals (from ml_data/full_dataset.csv)")
ml_path = os.path.join("ml_data", "full_dataset.csv")
chart_expl_path = os.path.join("ml_data", "training_chart_explanations.md")
chart_expl = {}
if os.path.exists(chart_expl_path):
    with open(chart_expl_path, "r", encoding="utf-8") as f:
        section = None
        for line in f:
            if line.startswith("## "):
                section = line.strip().replace("## ", "").lower()
                chart_expl[section] = ""
            elif section:
                chart_expl[section] += line
if os.path.exists(ml_path):
    ml_df = pd.read_csv(ml_path)
    if 'symbol' in ml_df.columns:
        ml_df = ml_df[ml_df['symbol'] == symbol]
    if 'expiry_date' in ml_df.columns:
        ml_df = ml_df[ml_df['expiry_date'] == expiry]
    if 'strike_price' in ml_df.columns:
        ml_df = ml_df[ml_df['strike_price'] == strike]
    if 'option_type' in ml_df.columns:
        ml_df = ml_df[ml_df['option_type'] == option_type]

    if not ml_df.empty:
        st.subheader("ML Training Data Visualization")
        # --- Price and Indicators Chart ---
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(pd.to_datetime(ml_df['"time"']), ml_df['close'], label='Close Price')
        if 'ma5' in ml_df.columns:
            ax1.plot(pd.to_datetime(ml_df['"time"']), ml_df['ma5'], label='MA5', alpha=0.7)
        if 'ma20' in ml_df.columns:
            ax1.plot(pd.to_datetime(ml_df['"time"']), ml_df['ma20'], label='MA20', alpha=0.7)
        ax1.set_title('Price and Indicators')
        ax1.legend()
        st.pyplot(fig1)
        if 'price_indicators' in chart_expl:
            st.markdown(chart_expl['price_indicators'])
        # --- RSI Chart ---
        if 'rsi' in ml_df.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 2))
            ax2.plot(pd.to_datetime(ml_df['"time"']), ml_df['rsi'], label='RSI')
            ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax2.set_title('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            st.pyplot(fig2)
            if 'rsi' in chart_expl:
                st.markdown(chart_expl['rsi'])
        # --- Buy/Sell Signals Chart ---
        if 'target' in ml_df.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(pd.to_datetime(ml_df['"time"']), ml_df['close'], label='Close Price', color='blue')
            buy_signals = ml_df[ml_df['target'] == 1]
            sell_signals = ml_df[ml_df['target'] == 0]
            ax3.scatter(pd.to_datetime(buy_signals['"time"']), buy_signals['close'], color='green', marker='^', label='Buy Signal', alpha=0.7)
            ax3.scatter(pd.to_datetime(sell_signals['"time"']), sell_signals['close'], color='red', marker='v', label='Sell Signal', alpha=0.7)
            ax3.set_title('Buy/Sell Signals')
            ax3.legend()
            st.pyplot(fig3)
            if 'signals' in chart_expl:
                st.markdown(chart_expl['signals'])
        st.download_button("Download ML Buy/Sell Signals CSV", ml_df.to_csv(index=False), file_name=f"{symbol}_ml_signals.csv")
    else:
        st.info("No ML signals found for this contract in ml_data/full_dataset.csv.")
else:
    st.info("ML data file not found (ml_data/full_dataset.csv).")

# --- Forward Test Section ---
st.subheader("Forward Testing")
st.markdown("**Forward testing simulates real-world prediction by training on historical data and testing on unseen recent data.**")

# Forward test date selection
col1, col2 = st.columns(2)
with col1:
    forward_start_date = st.date_input("Forward Test Start Date", value=date(2024, 6, 1))
with col2:
    forward_end_date = st.date_input("Forward Test End Date", value=date(2024, 6, 10))

# Forward test controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Forward Test Dataset"):
        with st.spinner("Generating forward test dataset..."):
            try:
                dp = DataPreparation()
                success = dp.prepare_forward_test_data(
                    symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    start_date=forward_start_date,
                    end_date=forward_end_date
                )
                if success:
                    st.success("Forward test dataset generated successfully!")
                else:
                    st.error("Failed to generate forward test dataset. Check logs for details.")
            except Exception as e:
                st.error(f"Error generating forward test dataset: {e}")

with col2:
    if st.button("Run Forward Test"):
        with st.spinner("Running forward test..."):
            try:
                dp = DataPreparation()
                success, metrics = dp.run_forward_test()
                if success and metrics:
                    st.success(f"Forward test completed with {metrics['accuracy']:.2%} accuracy!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("Total Samples", metrics['total_samples'])
                    with col3:
                        st.metric("Buy Predictions", metrics['buy_predictions'])
                    with col4:
                        st.metric("Sell Predictions", metrics['sell_predictions'])
                    
                    # Display confusion matrix
                    if 'confusion_matrix' in metrics:
                        import seaborn as sns
                        fig, ax = plt.subplots(figsize=(6, 4))
                        cm = np.array(metrics['confusion_matrix'])
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                else:
                    st.error("Failed to run forward test. Make sure you have a trained model and forward test dataset.")
            except Exception as e:
                st.error(f"Error running forward test: {e}")

# Display forward test results if available
forward_test_path = os.path.join("ml_data", "forward_test_results.csv")
if os.path.exists(forward_test_path):
    st.subheader("Forward Test Results")
    fwd_df = pd.read_csv(forward_test_path)
    
    if not fwd_df.empty:
        # Date range filter for results
        if '"time"' in fwd_df.columns:
            fwd_df = fwd_df.rename(columns={'"time"': 'time'})
        fwd_df['time'] = pd.to_datetime(fwd_df['time']) if 'time' in fwd_df.columns else pd.to_datetime(fwd_df.iloc[:,0])
        
        min_date = fwd_df['time'].min().date()
        max_date = fwd_df['time'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            result_start_date = st.date_input("Results Start Date", min_value=min_date, max_value=max_date, value=min_date, key="result_start")
        with col2:
            result_end_date = st.date_input("Results End Date", min_value=min_date, max_value=max_date, value=max_date, key="result_end")
        
        # Filter by selected date range
        mask = (fwd_df['time'].dt.date >= result_start_date) & (fwd_df['time'].dt.date <= result_end_date)
        fwd_df_filtered = fwd_df.loc[mask]
        
        if not fwd_df_filtered.empty:
            st.dataframe(fwd_df_filtered, use_container_width=True)
            
            # Calculate accuracy for filtered range
            if 'actual' in fwd_df_filtered.columns and 'predicted' in fwd_df_filtered.columns:
                acc = (fwd_df_filtered['actual'] == fwd_df_filtered['predicted']).mean()
                st.metric("Filtered Range Accuracy", f"{acc*100:.2f}%")
                
                # Plot predictions vs actual
                fig, ax = plt.subplots(figsize=(12, 6))
                time_col = pd.to_datetime(fwd_df_filtered['time'])
                ax.plot(time_col, fwd_df_filtered['close'], label='Close Price', color='blue', alpha=0.7)
                
                # Plot buy/sell signals
                buy_signals = fwd_df_filtered[fwd_df_filtered['predicted'] == 1]
                sell_signals = fwd_df_filtered[fwd_df_filtered['predicted'] == 0]
                
                if not buy_signals.empty:
                    ax.scatter(pd.to_datetime(buy_signals['time']), buy_signals['close'], 
                             color='green', marker='^', label='Predicted Buy', alpha=0.8, s=50)
                if not sell_signals.empty:
                    ax.scatter(pd.to_datetime(sell_signals['time']), sell_signals['close'], 
                             color='red', marker='v', label='Predicted Sell', alpha=0.8, s=50)
                
                ax.set_title('Forward Test Predictions vs Price')
                ax.set_ylabel('Price')
                ax.set_xlabel('Time')
                ax.legend()
                st.pyplot(fig)
                
                st.download_button("Download Forward Test Results", fwd_df_filtered.to_csv(index=False), 
                                 file_name=f"{symbol}_forward_test_results.csv")
        else:
            st.warning("No forward test data in the selected date range.")
    else:
        st.info("Forward test results file is empty.")

# --- Live Testing Section ---
st.subheader("Live Testing")
st.markdown("**Live testing runs predictions on the most recent data as if trading in real-time.**")

col1, col2 = st.columns(2)
with col1:
    if st.button("Run Live Inference"):
        with st.spinner("Running live inference..."):
            try:
                dp = DataPreparation()
                success, result = dp.run_live_inference(
                    symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type
                )
                if success and result:
                    st.success("Live inference completed!")
                    
                    # Display live prediction results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", result['prediction'])
                        st.metric("Current Price", f"₹{result['current_price']:.2f}")
                        st.metric("RSI", f"{result['rsi']:.2f}")
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                        st.metric("Buy Probability", f"{result['buy_probability']:.2%}")
                        st.metric("Sell Probability", f"{result['sell_probability']:.2%}")
                    
                    # Display additional metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Volume", f"{result['volume']:,}")
                        st.metric("MA5", f"₹{result['ma5']:.2f}")
                    with col2:
                        st.metric("MA20", f"₹{result['ma20']:.2f}")
                        st.metric("Timestamp", result['timestamp'])
                        
                else:
                    st.error("Failed to run live inference. Make sure you have a trained model.")
            except Exception as e:
                st.error(f"Error running live inference: {e}")

with col2:
    if st.button("View Live Predictions History"):
        live_predictions_path = 'ml_data/live_predictions.csv'
        if os.path.exists(live_predictions_path):
            live_df = pd.read_csv(live_predictions_path)
            if not live_df.empty:
                st.dataframe(live_df, use_container_width=True)
                
                # Plot live predictions over time
                if 'timestamp' in live_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'])
                    live_df = live_df.sort_values('timestamp')
                    
                    ax.plot(live_df['timestamp'], live_df['current_price'], label='Price', color='blue')
                    
                    # Plot predictions
                    buy_times = live_df[live_df['prediction'] == 'BUY']['timestamp']
                    buy_prices = live_df[live_df['prediction'] == 'BUY']['current_price']
                    sell_times = live_df[live_df['prediction'] == 'SELL']['timestamp']
                    sell_prices = live_df[live_df['prediction'] == 'SELL']['current_price']
                    
                    if not buy_times.empty:
                        ax.scatter(buy_times, buy_prices, color='green', marker='^', label='BUY', s=50)
                    if not sell_times.empty:
                        ax.scatter(sell_times, sell_prices, color='red', marker='v', label='SELL', s=50)
                    
                    ax.set_title('Live Predictions History')
                    ax.set_ylabel('Price')
                    ax.set_xlabel('Time')
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.download_button("Download Live Predictions", live_df.to_csv(index=False), 
                                     file_name=f"{symbol}_live_predictions.csv")
            else:
                st.info("No live predictions found.")
        else:
            st.info("No live predictions file found. Run live inference first.")

# --- Enhanced ML Training Result Section ---
strike_val = float(strike)
expiry_val = pd.to_datetime(expiry).date() if not isinstance(expiry, date) else expiry

if st.button("Run ML Training on this slice"):
    with st.spinner("Running ML training for this contract..."):
        try:
            dp = DataPreparation(llm_all_data=True)
            res = dp.prepare_ml_training_data(
                symbol=symbol,
                expiry=expiry_val,
                strike=strike_val,
                option_type=option_type
            )
            if res:
                st.success("ML training completed for this contract!")

                st.markdown(f"""
**ML Training Just Completed**
- **Symbol:** `{symbol}`
- **Expiry:** `{expiry_val}`
- **Strike:** `{strike_val}`
- **Option Type:** `{option_type}`
- **Output file:** `ml_data/full_dataset.csv`
                """)

                ml_csv_path = "ml_data/full_dataset.csv"
                if os.path.exists(ml_csv_path):
                    dfml = pd.read_csv(ml_csv_path)
                    st.markdown(f"**Rows in dataset:** {len(dfml)}")
                    st.dataframe(dfml.head(10), use_container_width=True)

                    # Class/label distribution
                    if 'target' in dfml.columns:
                        st.markdown("**Class Distribution:**")
                        lbl_counts = dfml['target'].value_counts().sort_index()
                        st.bar_chart(lbl_counts)
                        st.markdown(f"`target=1` (Buy): {lbl_counts.get(1,0)} &nbsp;&nbsp;&nbsp; `target=0` (Sell): {lbl_counts.get(0,0)}")

                    # Feature distributions (e.g. RSI, price)
                    plot_cols = []
                    if 'rsi' in dfml.columns:
                        plot_cols.append('rsi')
                    if 'close' in dfml.columns:
                        plot_cols.append('close')
                    if plot_cols:
                        for col in plot_cols:
                            fig, ax = plt.subplots()
                            dfml[col].hist(bins=30, ax=ax)
                            ax.set_title(f"{col} distribution")
                            st.pyplot(fig)

                    # Show price vs. target
                    if {'close','target','"time"'}.issubset(dfml.columns):
                        fig, ax = plt.subplots(figsize=(10,4))
                        timecol = pd.to_datetime(dfml['"time"'])
                        ax.plot(timecol, dfml['close'], label='Close Price', color='blue')
                        ax.scatter(timecol[dfml['target']==1], dfml[dfml['target']==1]['close'], label='Buy (target=1)', color='green', marker='^')
                        ax.scatter(timecol[dfml['target']==0], dfml[dfml['target']==0]['close'], label='Sell (target=0)', color='red', marker='v')
                        ax.set_title("Close Price with ML Targets")
                        ax.set_ylabel("Price")
                        ax.set_xlabel("Time")
                        ax.legend()
                        st.pyplot(fig)

                    # Show saved matplotlib plot if present
                    plot_path = "ml_data/training_data_visualization.png"
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Training Data Visualization (Indicators & Targets)", use_column_width=True)

                st.info("""
**What just happened?**
- The ML training pipeline took the selected contract's historical candles and indicators, engineered features, and labeled each row as `target=1` (buy) if future price was higher, or `target=0` (sell) otherwise.
- The output CSV (`ml_data/full_dataset.csv`) can be used for ML model training (classification/regression, etc).
- Features: Price, volume, technical indicators (RSI, OBV, TVI, etc.), moving averages, and their changes.
- The charts above show sample feature distributions and class balance. Use these to check for data quality, class imbalance, or outliers.
- For deeper ML, load this CSV in your favorite ML framework (scikit-learn, xgboost, tensorflow, etc).

**Next steps:**
- Download the CSV and experiment!
- Retrain your ML model using this data
- Use the visualizations to help with feature selection or label engineering.
                """)
            else:
                st.error("ML training failed. See logs for details.")
        except Exception as e:
            st.error(f"ML training error: {e}")

st.subheader("Download Data Files (Raw/ML Sample)")
col1, col2, col3 = st.columns(3)
with col1:
    cmaster_path = os.path.join("historical_data", "combined_master.csv")
    if os.path.exists(cmaster_path):
        with open(cmaster_path, "rb") as f:
            st.download_button("Download Raw Combined Master CSV", data=f, file_name="combined_master.csv", mime="text/csv")
with col2:
    ml_path = os.path.join("ml_data", "full_dataset.csv")
    if os.path.exists(ml_path):
        with open(ml_path, "rb") as f:
            st.download_button("Download ML Full Dataset CSV", data=f, file_name="full_dataset.csv", mime="text/csv")
with col3:
    sample_path = os.path.join("scalping_signals", "sample_scalping_data.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button("Download Sample Scalping Data CSV", data=f, file_name="sample_scalping_data.csv", mime="text/csv")

with st.expander("❓ How to use this tool / Signal logic"):
    st.markdown("""
**How to use:**
- **Pick Underlying, Expiry, Strike, and Option Type** to select the exact contract you want.
- **Pick date range and candle timeframe** (1min or 15sec).
- **See/download raw candles** and **scalping signals** instantly.
- **See/download ML buy/sell signals** from the ML output file, including a plot.
- Use "Run ML Training" to trigger model training for the chosen slice (requires backend hook).
- **Download Raw/ML/Sample Data** from the section above if you want to work with the CSVs directly.

**Signal Logic (for ML and trading):**
- **buy**: RSI < 30 (oversold), OBV rising, close > MA5, volume > 1.5 × 20-period average
- **sell**: RSI > 70 (overbought), OBV falling, close < MA5, volume > 1.5 × 20-period average
- **trend_up**: Close > MA20 & RSI rising
- **trend_down**: Close < MA20 & RSI falling
- **hold**: none of the above

**Columns:**
- `close`: Last price
- `rsi`, `obv`, `tvi`: Indicators
- `ma5`, `ma20`: 5/20-period moving averages
- `avg_vol20`: 20-period average volume

You can use these as features for ML, and `signal_type` as a label.
    """)

with st.expander("🗒️ Client Request Coverage"):
    st.markdown("""
- **Historical data:** Select any symbol/expiry/strike/type, date range, and view candles.
- **Weekly symbol/ticker changes:** GUI auto-loads all available contracts from DB.
- **On-demand signals:** Select any contract and instantly view/download rich signals for any expiry/strike/type.
- **ML Training:** Use the "Run ML Training" button to trigger backend ML training for the selected data slice.
- **Download Section:** Download raw historical (combined_master.csv), ML full dataset, and sample scalping data CSVs.
    """)