"""
Data Preparation for Scalping ML/LLM Application

This script prepares the collected and processed NSE BankNifty and options data
for use in scalping strategies and ML/LLM applications. It creates feature sets,
generates training data, and provides interfaces for real-time signal generation.
"""

import logging
import time
import psycopg2
from psycopg2 import pool
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from dotenv import load_dotenv
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataPreparation")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}
for k, v in DB_CONFIG.items():
    if not v:
        logger.error(f"Missing {k} in .env file.")
        sys.exit(1)

# --- Global connection pool ---
db_pool = None
def init_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
        if db_pool:
            logger.info('Database connection pool created')
        else:
            logger.error('Failed to create database connection pool')
init_db_pool()

class DataPreparation:
    def __init__(self, llm_all_data=False):
        self.db_conn = None
        self.db_cursor = None
        self.llm_all_data = llm_all_data

        # Initialize database connection
        self.connect_to_db()

        # Create output directories
        os.makedirs("ml_data", exist_ok=True)
        os.makedirs("llm_data", exist_ok=True)
        os.makedirs("scalping_signals", exist_ok=True)

    def connect_to_db(self):
        """Establish connection to PostgreSQL/TimescaleDB"""
        try:
            self.db_conn = db_pool.getconn()
            self.db_conn.autocommit = False
            self.db_cursor = self.db_conn.cursor()
            logger.info("Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            return False

    def get_first_available_instrument(self):
        self.db_cursor.execute("SELECT instrument_id, symbol FROM instruments LIMIT 1")
        result = self.db_cursor.fetchone()
        if result:
            return result[0], result[1]
        return None, None

    def get_best_instrument_with_data(self):
        self.db_cursor.execute('''
            SELECT i.instrument_id, i.symbol
            FROM instruments i
            WHERE i.symbol LIKE '%NIFTY%'
            AND EXISTS (SELECT 1 FROM candle_data_1min c WHERE c.instrument_id = i.instrument_id)
            AND EXISTS (SELECT 1 FROM technical_indicators t WHERE t.instrument_id = i.instrument_id AND t.timeframe = '1min')
            ORDER BY CASE WHEN i.symbol LIKE 'NIFTY%' THEN 0 WHEN i.symbol LIKE 'BANKNIFTY%' THEN 1 ELSE 2 END
            LIMIT 1
        ''')
        result = self.db_cursor.fetchone()
        if result:
            return result[0], result[1]
        self.db_cursor.execute('''
            SELECT i.instrument_id, i.symbol
            FROM instruments i
            WHERE EXISTS (SELECT 1 FROM candle_data_1min c WHERE c.instrument_id = i.instrument_id)
            AND EXISTS (SELECT 1 FROM technical_indicators t WHERE t.instrument_id = i.instrument_id AND t.timeframe = '1min')
            LIMIT 1
        ''')
        result = self.db_cursor.fetchone()
        if result:
            return result[0], result[1]
        return None, None

    def prepare_scalping_data(self):
        try:
            self.db_cursor.execute("""
                CREATE OR REPLACE VIEW scalping_data AS
                SELECT 
                    cd."time",
                    i.symbol,
                    i.instrument_type,
                    i.option_type,
                    i.strike_price,
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.open_interest,
                    ti.tvi,
                    ti.obv,
                    ti.rsi,
                    ti.pvi,
                    ti.pvt
                FROM candle_data_1min cd
                JOIN instruments i ON cd.instrument_id = i.instrument_id
                LEFT JOIN technical_indicators ti ON 
                    cd."time" = ti."time" AND 
                    cd.instrument_id = ti.instrument_id AND 
                    ti.timeframe = '1min'
                ORDER BY cd."time" DESC;
            """)
            self.db_cursor.execute("""
                CREATE OR REPLACE FUNCTION get_latest_scalping_data(
                    p_limit INTEGER DEFAULT 100
                )
                RETURNS TABLE (
                    "time" TIMESTAMP WITH TIME ZONE,
                    symbol VARCHAR,
                    instrument_type VARCHAR,
                    option_type VARCHAR,
                    strike_price NUMERIC,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    open_interest BIGINT,
                    tvi NUMERIC,
                    obv NUMERIC,
                    rsi NUMERIC,
                    pvi NUMERIC,
                    pvt NUMERIC
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT * FROM scalping_data
                    ORDER BY "time" DESC
                    LIMIT p_limit;
                END;
                $$ LANGUAGE plpgsql;
            """)
            self.db_cursor.execute("""
                CREATE OR REPLACE FUNCTION get_instrument_scalping_data(
                    p_symbol VARCHAR,
                    p_limit INTEGER DEFAULT 100
                )
                RETURNS TABLE (
                    "time" TIMESTAMP WITH TIME ZONE,
                    symbol VARCHAR,
                    instrument_type VARCHAR,
                    option_type VARCHAR,
                    strike_price NUMERIC,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    open_interest BIGINT,
                    tvi NUMERIC,
                    obv NUMERIC,
                    rsi NUMERIC,
                    pvi NUMERIC,
                    pvt NUMERIC
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT * FROM scalping_data
                    WHERE symbol = p_symbol
                    ORDER BY "time" DESC
                    LIMIT p_limit;
                END;
                $$ LANGUAGE plpgsql;
            """)
            self.db_conn.commit()
            logger.info("Scalping data preparation completed")
            self.db_cursor.execute("SELECT * FROM scalping_data")
            rows = self.db_cursor.fetchall()
            if rows:
                columns = [
                    '"time"', 'symbol', 'instrument_type', 'option_type', 'strike_price',
                    'open', 'high', 'low', 'close', 'volume', 'open_interest',
                    'tvi', 'obv', 'rsi', 'pvi', 'pvt'
                ]
                df = pd.DataFrame(rows, columns=columns)
                # Convert datetime columns to string for CSV export
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)
                df.to_csv('scalping_signals/sample_scalping_data.csv', index=False)
                logger.info(f"Saved sample scalping data with {len(df)} rows")
            else:
                logger.warning("No scalping data available for sample")
            return True
        except Exception as e:
            logger.error(f"Error preparing scalping data: {str(e)}")
            self.db_conn.rollback()
            return False

    def create_scalping_signal_generator(self):
        try:
            self.db_cursor.execute("DROP FUNCTION IF EXISTS generate_scalping_signals_flexible(VARCHAR, VARCHAR, INTEGER) CASCADE;")
            self.db_cursor.execute("""
                CREATE OR REPLACE FUNCTION generate_scalping_signals_flexible(
                    p_timeframe VARCHAR DEFAULT '1min',
                    p_symbol VARCHAR DEFAULT NULL,
                    p_limit INTEGER DEFAULT 100
                )
                RETURNS TABLE (
                    "time" TIMESTAMP WITH TIME ZONE,
                    symbol VARCHAR,
                    signal_type VARCHAR,
                    close NUMERIC,
                    rsi NUMERIC,
                    obv NUMERIC,
                    tvi NUMERIC,
                    volume BIGINT,
                    ma5 NUMERIC,
                    ma20 NUMERIC,
                    avg_vol20 NUMERIC
                ) AS $$
                DECLARE
                    table_name TEXT;
                    sql TEXT;
                BEGIN
                    table_name := 'candle_data_' || lower(p_timeframe);

                    sql := '
                        SELECT
                            cd."time",
                            i.symbol,
                            CASE
                                WHEN ti.rsi < 30
                                  AND ti.obv > LAG(ti.obv) OVER (ORDER BY cd."time")
                                  AND cd.close > AVG(cd.close) OVER (ORDER BY cd."time" ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
                                  AND cd.volume > 1.5 * AVG(cd.volume) OVER (ORDER BY cd."time" ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                                  THEN ''buy''::varchar
                                WHEN ti.rsi > 70
                                  AND ti.obv < LAG(ti.obv) OVER (ORDER BY cd."time")
                                  AND cd.close < AVG(cd.close) OVER (ORDER BY cd."time" ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
                                  AND cd.volume > 1.5 * AVG(cd.volume) OVER (ORDER BY cd."time" ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                                  THEN ''sell''::varchar
                                WHEN cd.close > AVG(cd.close) OVER (ORDER BY cd."time" ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                                  AND ti.rsi > LAG(ti.rsi) OVER (ORDER BY cd."time")
                                  THEN ''trend_up''::varchar
                                WHEN cd.close < AVG(cd.close) OVER (ORDER BY cd."time" ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                                  AND ti.rsi < LAG(ti.rsi) OVER (ORDER BY cd."time")
                                  THEN ''trend_down''::varchar
                                ELSE ''hold''::varchar
                            END AS signal_type,
                            cd.close,
                            ti.rsi,
                            ti.obv,
                            ti.tvi,
                            cd.volume,
                            AVG(cd.close) OVER (ORDER BY cd."time" ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ma5,
                            AVG(cd.close) OVER (ORDER BY cd."time" ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                            AVG(cd.volume) OVER (ORDER BY cd."time" ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol20
                        FROM ' || table_name || ' cd
                        JOIN instruments i ON cd.instrument_id = i.instrument_id
                        LEFT JOIN technical_indicators ti ON
                            cd."time" = ti."time" AND
                            cd.instrument_id = ti.instrument_id AND
                            ti.timeframe = $1
                    ';

                    IF p_symbol IS NOT NULL THEN
                        sql := sql || ' WHERE i.symbol = $2 ';
                        sql := sql || ' ORDER BY cd."time" DESC LIMIT $3 ';
                        RETURN QUERY EXECUTE sql USING p_timeframe, p_symbol, p_limit;
                    ELSE
                        sql := sql || ' ORDER BY cd."time" DESC LIMIT $2 ';
                        RETURN QUERY EXECUTE sql USING p_timeframe, p_limit;
                    END IF;
                END;
                $$ LANGUAGE plpgsql;
            """)
            self.db_conn.commit()
            logger.info("Flexible scalping signal generator (generate_scalping_signals_flexible) created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating flexible scalping signal generator: {str(e)}")
            self.db_conn.rollback()
            return False

    def create_feature_engineering_functions(self):
        try:
            self.db_cursor.execute("""
                CREATE OR REPLACE FUNCTION calculate_moving_average(
                    p_instrument_id INTEGER,
                    p_timeframe VARCHAR,
                    p_window INTEGER
                )
                RETURNS TABLE (
                    "time" TIMESTAMP WITH TIME ZONE,
                    ma NUMERIC
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        "time",
                        AVG(close) OVER (
                            PARTITION BY instrument_id 
                            ORDER BY "time" 
                            ROWS BETWEEN p_window-1 PRECEDING AND CURRENT ROW
                        ) AS ma
                    FROM candle_data_1min
                    WHERE instrument_id = p_instrument_id
                    ORDER BY "time" DESC;
                END;
                $$ LANGUAGE plpgsql;
            """)
            self.db_conn.commit()
            logger.info("Feature engineering functions created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating feature engineering functions: {str(e)}")
            self.db_conn.rollback()
            return False

    def prepare_ml_training_data(self, symbol=None, expiry=None, strike=None, option_type=None):
        """Prepare data for machine learning model training for a specific contract if provided"""
        try:
            instrument_id = None
            symbol_used = None

            # Robust conversion
            if expiry is not None and not isinstance(expiry, date):
                try:
                    expiry = pd.to_datetime(expiry).date()
                except Exception:
                    expiry = None
            if strike is not None:
                strike = float(strike)

            if symbol:
                params = [symbol]
                query = "SELECT instrument_id, symbol FROM instruments WHERE symbol=%s"
                if expiry is not None:
                    query += " AND expiry_date=%s"
                    params.append(expiry)
                if strike is not None:
                    query += " AND strike_price=%s"
                    params.append(strike)
                if option_type is not None:
                    query += " AND option_type=%s"
                    params.append(option_type)
                self.db_cursor.execute(query, tuple(params))
                result = self.db_cursor.fetchone()
                if result:
                    instrument_id, symbol_used = result
                else:
                    logger.error(f"No instrument found for symbol={symbol}, expiry={expiry}, strike={strike}, option_type={option_type}")
                    return False
            else:
                instrument_id, symbol_used = self.get_best_instrument_with_data()
            if not instrument_id:
                logger.error("No instrument with both candles and indicators found in database.")
                return False
            logger.info(f"Using symbol: {symbol_used}")

            self.db_cursor.execute("SELECT COUNT(*) FROM technical_indicators WHERE instrument_id = %s AND timeframe='1min'", (instrument_id,))
            indicator_count = self.db_cursor.fetchone()[0]
            self.db_cursor.execute("SELECT COUNT(*) FROM candle_data_1min WHERE instrument_id = %s", (instrument_id,))
            candle_count = self.db_cursor.fetchone()[0]
            logger.info(f"Found {indicator_count} indicators and {candle_count} 1min candles for {symbol_used}")
            # --- Auto-trigger indicator calculation if missing ---
            if indicator_count == 0 and candle_count > 0:
                logger.warning(f"No indicators found for instrument_id={instrument_id} ({symbol_used}), running indicator calculation...")
                try:
                    from indicator_calculator import IndicatorCalculator
                    calc = IndicatorCalculator(mode="batch")
                    calc.start()
                    # Re-check indicator count
                    self.db_cursor.execute("SELECT COUNT(*) FROM technical_indicators WHERE instrument_id = %s AND timeframe='1min'", (instrument_id,))
                    indicator_count = self.db_cursor.fetchone()[0]
                    logger.info(f"After recalculation: {indicator_count} indicators for {symbol_used}")
                    if indicator_count == 0:
                        logger.error(f"Indicator calculation did not produce any indicators for instrument_id={instrument_id} ({symbol_used})")
                        return False
                except Exception as e:
                    logger.error(f"Error auto-triggering indicator calculation: {str(e)}")
                    return False

            self.db_cursor.execute("""
                SELECT 
                    cd."time",
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.open_interest,
                    ti.tvi,
                    ti.obv,
                    ti.rsi,
                    ti.pvi,
                    ti.pvt,
                    LEAD(cd.close, 4) OVER (ORDER BY cd."time") AS future_price
                FROM candle_data_1min cd
                JOIN technical_indicators ti ON 
                    cd."time" = ti."time" AND 
                    cd.instrument_id = ti.instrument_id AND 
                    ti.timeframe = '1min'
                WHERE 
                    cd.instrument_id = %s
                ORDER BY cd."time" ASC
            """, (instrument_id,))

            rows = self.db_cursor.fetchall()
            if not rows:
                logger.warning(f"No data available for ML training for instrument_id={instrument_id} ({symbol_used})")
                return False

            columns = [
                '"time"', 'open', 'high', 'low', 'close', 'volume', 'open_interest',
                'tvi', 'obv', 'rsi', 'pvi', 'pvt', 'future_price'
            ]
            df = pd.DataFrame(rows, columns=columns)
            df = df.dropna(subset=['future_price'])
            if df.empty:
                logger.warning(f"All joined data dropped due to missing future_price for instrument_id={instrument_id} ({symbol_used})")
                return False

            df['target'] = (df['future_price'] > df['close']).astype(int)
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['oi_change'] = df['open_interest'].pct_change()
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['rsi_change'] = df['rsi'].diff()
            df = df.dropna()
            if df.empty:
                logger.warning(f"All data dropped after feature engineering for instrument_id={instrument_id} ({symbol_used})")
                return False

            # Convert datetime columns to string for CSV export
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
            df.to_csv('ml_data/full_dataset.csv', index=False)
            features = [
                'open', 'high', 'low', 'close', 'volume', 'open_interest',
                'tvi', 'obv', 'rsi', 'pvi', 'pvt',
                'price_change', 'volume_change', 'oi_change',
                'ma5', 'ma20', 'rsi_change'
            ]
            X = df[features]
            y = df['target']

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()
            X = X.clip(lower=-1e10, upper=1e10)
            y = y.loc[X.index]
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            # --- Chronological Split for Training and Forward Testing ---
            # Use the first 80% of data for training, and the last 20% for forward testing.
            # This simulates a real-world scenario where we train on past data and test on future data.
            split_idx = int(0.8 * len(df))
            train_df = df.iloc[:split_idx]
            forward_df = df.iloc[split_idx:]

            # Prepare training data
            X_train = train_df[features]
            y_train = train_df['target']
            X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
            y_train = y_train.loc[X_train.index]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            with open('ml_data/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            # Prepare forward test data
            X_forward = forward_df[features]
            y_forward = forward_df['target']
            X_forward = X_forward.replace([np.inf, -np.inf], np.nan).dropna()
            y_forward = y_forward.loc[X_forward.index]
            X_forward_scaled = scaler.transform(X_forward)
            
            np.save('ml_data/X_train.npy', X_train_scaled)
            np.save('ml_data/y_train.npy', y_train)
            np.save('ml_data/X_test.npy', X_forward_scaled)
            np.save('ml_data/y_test.npy', y_forward)

            with open('ml_data/feature_names.txt', 'w') as f:
                f.write('\n'.join(features))

            # --- ML Model Training and Persistence ---
            try:
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train_scaled, y_train)
                with open('ml_data/model.pkl', 'wb') as f:
                    pickle.dump(clf, f)
                logger.info('RandomForestClassifier model trained and saved to ml_data/model.pkl')
                # --- Forward Test Evaluation ---
                if len(X_forward_scaled) > 0:
                    y_pred = clf.predict(X_forward_scaled)
                    from sklearn.metrics import accuracy_score
                    forward_acc = accuracy_score(y_forward, y_pred)
                    logger.info(f'Forward test accuracy: {forward_acc:.2%}')
                    # Save forward test results
                    results_df = forward_df.loc[X_forward.index].copy()
                    results_df['predicted'] = y_pred
                    results_df['actual'] = y_forward.values
                    # Convert datetime columns to string for CSV export
                    for col in results_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(results_df[col]):
                            results_df[col] = results_df[col].astype(str)
                    results_df.to_csv('ml_data/forward_test_results.csv', index=False)
                else:
                    logger.warning('No data available for forward test split.')
            except Exception as e:
                logger.error(f'Error training or saving RandomForestClassifier: {str(e)}')

            logger.info(f"ML training data prepared successfully with {len(df)} samples for symbol {symbol_used}")

            plt.figure(figsize=(12, 8))
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(df['"time"'], df['close'], label='Close Price')
            ax1.plot(df['"time"'], df['ma5'], label='MA5', alpha=0.7)
            ax1.plot(df['"time"'], df['ma20'], label='MA20', alpha=0.7)
            ax1.set_title(f'{symbol_used} Price and Indicators')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(df['"time"'], df['rsi'], label='RSI')
            ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            buy_signals = df[df['target'] == 1]
            sell_signals = df[df['target'] == 0]
            ax3.scatter(buy_signals['"time"'], buy_signals['close'], color='green', label='Buy Signal', marker='^', alpha=0.7)
            ax3.scatter(sell_signals['"time"'], sell_signals['close'], color='red', label='Sell Signal', marker='v', alpha=0.7)
            ax3.set_ylabel('Signals')
            ax3.legend()
            plt.tight_layout()
            plt.savefig('ml_data/training_data_visualization.png')
            plt.close()
            # --- Chart Explanations ---
            chart_explanations = {
                'price_indicators': (
                    """
                    **Price and Indicators Chart**  
                    This chart displays the contract's closing price along with moving averages (MA5 and MA20).
                    - **Close Price**: The actual closing price for each time interval.
                    - **MA5/MA20**: Short-term and long-term moving averages, which help identify trends and momentum.

                    **How to interpret:**
                    - When the close price is above MA20 and MA5, it often signals an uptrend.
                    - Crossovers between MA5 and MA20 can indicate trend reversals.
                    """
                ),
                'rsi': (
                    """
                    **RSI (Relative Strength Index) Chart**  
                    This chart shows the RSI value over time.
                    - **RSI**: Measures momentum and identifies overbought (>70) or oversold (<30) conditions.

                    **How to interpret:**
                    - RSI > 70: Asset may be overbought (potential for price to fall).
                    - RSI < 30: Asset may be oversold (potential for price to rise).
                    - RSI between 30 and 70: Neutral/sideways market.
                    """
                ),
                'signals': (
                    """
                    **Buy/Sell Signals Chart**  
                    This chart marks buy (green ^) and sell (red v) signals on the price chart.
                    - **Buy Signal**: Model predicts price will rise in the next interval.
                    - **Sell Signal**: Model predicts price will fall.

                    **How to interpret:**
                    - Clusters of buy signals may indicate strong upward momentum.
                    - Clusters of sell signals may indicate downward momentum.
                    - Use in conjunction with price and RSI for confirmation.
                    """
                )
            }
            # Optionally, save explanations to a markdown file for GUI use
            with open('ml_data/training_chart_explanations.md', 'w') as f:
                for key, text in chart_explanations.items():
                    f.write(f"## {key.capitalize()}\n{text}\n\n")
            return True
        except Exception as e:
            logger.error(f"Error preparing ML training data: {str(e)}")
            return False

    def prepare_llm_data(self):
        try:
            if self.llm_all_data:
                latest_prices_where = ""
                latest_indicators_where = "WHERE ti.timeframe = '1min'"
                logger.info("Preparing LLM data using ALL available historical data (no time filter).")
            else:
                latest_prices_where = "WHERE cd.\"time\" >= NOW() - INTERVAL '30 minutes'"
                latest_indicators_where = "WHERE ti.timeframe = '1min' AND ti.\"time\" >= NOW() - INTERVAL '30 minutes'"
                logger.info("Preparing LLM data using ONLY the last 30 minutes of data (default behavior).")
            query = f"""
                CREATE OR REPLACE VIEW market_summary AS
                WITH latest_prices AS (
                    SELECT 
                        i.symbol,
                        i.instrument_type,
                        i.option_type,
                        i.strike_price,
                        cd.close AS latest_price,
                        cd."time" AS latest_time,
                        LAG(cd.close) OVER (PARTITION BY i.instrument_id ORDER BY cd."time" DESC) AS prev_price,
                        cd.volume,
                        cd.open_interest
                    FROM instruments i
                    JOIN candle_data_1min cd ON i.instrument_id = cd.instrument_id
                    {latest_prices_where}
                    ORDER BY i.symbol, cd."time" DESC
                ),
                latest_indicators AS (
                    SELECT 
                        i.symbol,
                        ti.rsi,
                        ti.tvi,
                        ti.obv,
                        ti."time"
                    FROM instruments i
                    JOIN technical_indicators ti ON i.instrument_id = ti.instrument_id
                    {latest_indicators_where}
                    ORDER BY i.symbol, ti."time" DESC
                )
                SELECT 
                    lp.symbol,
                    lp.instrument_type,
                    lp.option_type,
                    lp.strike_price,
                    lp.latest_price,
                    lp.latest_time,
                    ROUND((lp.latest_price - lp.prev_price) / NULLIF(lp.prev_price, 0) * 100, 2) AS price_change_pct,
                    lp.volume,
                    lp.open_interest,
                    li.rsi,
                    li.tvi,
                    li.obv,
                    CASE
                        WHEN li.rsi > 70 THEN 'Overbought'
                        WHEN li.rsi < 30 THEN 'Oversold'
                        ELSE 'Neutral'
                    END AS rsi_signal
                FROM latest_prices lp
                LEFT JOIN latest_indicators li ON lp.symbol = li.symbol AND lp.latest_time = li."time"
                WHERE lp.prev_price IS NOT NULL
                ORDER BY lp.symbol;
            """
            self.db_cursor.execute(query)
            self.db_conn.commit()

            self.db_cursor.execute("SELECT * FROM market_summary")
            rows = self.db_cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in self.db_cursor.description]
                market_data = []
                for row in rows:
                    market_data.append(dict(zip(columns, row)))
                df = pd.DataFrame(market_data)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                # Convert datetime columns to string for CSV export
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)
                df.to_csv('llm_data/market_data.csv', index=False)
                df.to_json('llm_data/market_data.json', orient='records', date_format='iso')
                logger.info(f"LLM data prepared successfully with {len(df)} records")
            else:
                logger.warning("No market data available for LLM preparation")
            return True
        except Exception as e:
            logger.error(f"Error preparing LLM data: {str(e)}")
            self.db_conn.rollback()
            return False

    def prepare_forward_test_data(self, symbol=None, expiry=None, strike=None, option_type=None, 
                                 start_date=None, end_date=None, output_path="ml_data/forward_test_dataset.csv"):
        """Prepare forward test dataset for a specific date range"""
        try:
            instrument_id = None
            symbol_used = None

            # Robust conversion
            if expiry is not None and not isinstance(expiry, date):
                try:
                    expiry = pd.to_datetime(expiry).date()
                except Exception:
                    expiry = None
            if strike is not None:
                strike = float(strike)
            if start_date is not None and not isinstance(start_date, date):
                start_date = pd.to_datetime(start_date).date()
            if end_date is not None and not isinstance(end_date, date):
                end_date = pd.to_datetime(end_date).date()

            if symbol:
                params = [symbol]
                query = "SELECT instrument_id, symbol FROM instruments WHERE symbol=%s"
                if expiry is not None:
                    query += " AND expiry_date=%s"
                    params.append(expiry)
                if strike is not None:
                    query += " AND strike_price=%s"
                    params.append(strike)
                if option_type is not None:
                    query += " AND option_type=%s"
                    params.append(option_type)
                self.db_cursor.execute(query, tuple(params))
                result = self.db_cursor.fetchone()
                if result:
                    instrument_id, symbol_used = result
                else:
                    logger.error(f"No instrument found for symbol={symbol}, expiry={expiry}, strike={strike}, option_type={option_type}")
                    return False
            else:
                instrument_id, symbol_used = self.get_best_instrument_with_data()
            
            if not instrument_id:
                logger.error("No instrument with both candles and indicators found in database.")
                return False
            
            logger.info(f"Preparing forward test data for symbol: {symbol_used}")

            # Build query with date range filter
            query = """
                SELECT 
                    cd."time",
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.open_interest,
                    ti.tvi,
                    ti.obv,
                    ti.rsi,
                    ti.pvi,
                    ti.pvt,
                    LEAD(cd.close, 4) OVER (ORDER BY cd."time") AS future_price
                FROM candle_data_1min cd
                JOIN technical_indicators ti ON 
                    cd."time" = ti."time" AND 
                    cd.instrument_id = ti.instrument_id AND 
                    ti.timeframe = '1min'
                WHERE 
                    cd.instrument_id = %s
            """
            params = [instrument_id]
            
            if start_date:
                query += " AND cd.\"time\" >= %s"
                params.append(start_date)
            if end_date:
                query += " AND cd.\"time\" <= %s"
                params.append(end_date)
            
            query += " ORDER BY cd.\"time\" ASC"
            
            self.db_cursor.execute(query, tuple(params))
            rows = self.db_cursor.fetchall()
            
            if not rows:
                logger.warning(f"No data available for forward test for instrument_id={instrument_id} ({symbol_used})")
                return False

            columns = [
                '"time"', 'open', 'high', 'low', 'close', 'volume', 'open_interest',
                'tvi', 'obv', 'rsi', 'pvi', 'pvt', 'future_price'
            ]
            df = pd.DataFrame(rows, columns=columns)
            df = df.dropna(subset=['future_price'])
            
            if df.empty:
                logger.warning(f"All forward test data dropped due to missing future_price for instrument_id={instrument_id} ({symbol_used})")
                return False

            # Create target and features (same as training data)
            df['target'] = (df['future_price'] > df['close']).astype(int)
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['oi_change'] = df['open_interest'].pct_change()
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['rsi_change'] = df['rsi'].diff()
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"All forward test data dropped after feature engineering for instrument_id={instrument_id} ({symbol_used})")
                return False

            # Save forward test dataset
            df.to_csv(output_path, index=False)
            logger.info(f"Forward test dataset saved to {output_path} with {len(df)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing forward test data: {str(e)}")
            return False

    def run_forward_test(self, forward_test_path="ml_data/forward_test_dataset.csv", 
                        model_path="ml_data/model.pkl", scaler_path="ml_data/scaler.pkl"):
        """Run forward test using trained model on forward test dataset"""
        try:
            # Check if required files exist
            if not os.path.exists(forward_test_path):
                logger.error(f"Forward test dataset not found: {forward_test_path}")
                return False
            if not os.path.exists(model_path):
                logger.error(f"Trained model not found: {model_path}")
                return False
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler not found: {scaler_path}")
                return False

            # Load data and model
            df = pd.read_csv(forward_test_path)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # Prepare features
            features = [
                'open', 'high', 'low', 'close', 'volume', 'open_interest',
                'tvi', 'obv', 'rsi', 'pvi', 'pvt',
                'price_change', 'volume_change', 'oi_change',
                'ma5', 'ma20', 'rsi_change'
            ]
            
            X = df[features]
            y = df['target']

            # Clean and scale features
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()
            X = X.clip(lower=-1e10, upper=1e10)
            y = y.loc[X.index]

            X_scaled = scaler.transform(X)

            # Make predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            
            accuracy = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)

            # Save results
            results_df = df.loc[X.index].copy()
            results_df['predicted'] = y_pred
            results_df['actual'] = y.values
            results_df['prediction_confidence'] = np.max(y_pred_proba, axis=1)
            results_df['predicted_probability_buy'] = y_pred_proba[:, 1]
            results_df['predicted_probability_sell'] = y_pred_proba[:, 0]
            # Convert datetime columns to string for CSV export
            for col in results_df.columns:
                if pd.api.types.is_datetime64_any_dtype(results_df[col]):
                    results_df[col] = results_df[col].astype(str)
            results_df.to_csv('ml_data/forward_test_results.csv', index=False)

            # Save metrics
            metrics = {
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'total_samples': len(y),
                'buy_predictions': int(sum(y_pred == 1)),
                'sell_predictions': int(sum(y_pred == 0)),
                'actual_buys': int(sum(y == 1)),
                'actual_sells': int(sum(y == 0))
            }
            
            import json
            with open('ml_data/forward_test_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Forward test completed with accuracy: {accuracy:.2%}")
            logger.info(f"Results saved to ml_data/forward_test_results.csv")
            logger.info(f"Metrics saved to ml_data/forward_test_metrics.json")
            
            return True, metrics
            
        except Exception as e:
            logger.error(f"Error running forward test: {str(e)}")
            return False, None

    def run_live_inference(self, symbol=None, expiry=None, strike=None, option_type=None, 
                          model_path="ml_data/model.pkl", scaler_path="ml_data/scaler.pkl"):
        """Run live inference on the most recent data"""
        try:
            # Load model and scaler
            if not os.path.exists(model_path):
                logger.error(f"Trained model not found: {model_path}")
                return False
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler not found: {scaler_path}")
                return False

            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # Get instrument ID
            instrument_id = None
            if symbol:
                params = [symbol]
                query = "SELECT instrument_id, symbol FROM instruments WHERE symbol=%s"
                if expiry is not None:
                    query += " AND expiry_date=%s"
                    params.append(expiry)
                if strike is not None:
                    query += " AND strike_price=%s"
                    params.append(strike)
                if option_type is not None:
                    query += " AND option_type=%s"
                    params.append(option_type)
                self.db_cursor.execute(query, tuple(params))
                result = self.db_cursor.fetchone()
                if result:
                    instrument_id, symbol_used = result
                else:
                    logger.error(f"No instrument found for live inference")
                    return False
            else:
                instrument_id, symbol_used = self.get_best_instrument_with_data()

            # Get latest data (last 50 candles to ensure we have enough for features)
            self.db_cursor.execute("""
                SELECT 
                    cd."time",
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.open_interest,
                    ti.tvi,
                    ti.obv,
                    ti.rsi,
                    ti.pvi,
                    ti.pvt
                FROM candle_data_1min cd
                JOIN technical_indicators ti ON 
                    cd."time" = ti."time" AND 
                    cd.instrument_id = ti.instrument_id AND 
                    ti.timeframe = '1min'
                WHERE 
                    cd.instrument_id = %s
                ORDER BY cd."time" DESC
                LIMIT 50
            """, (instrument_id,))

            rows = self.db_cursor.fetchall()
            if not rows:
                logger.error("No recent data available for live inference")
                return False

            columns = [
                '"time"', 'open', 'high', 'low', 'close', 'volume', 'open_interest',
                'tvi', 'obv', 'rsi', 'pvi', 'pvt'
            ]
            df = pd.DataFrame(rows, columns=columns)
            df = df.sort_values('"time"').reset_index(drop=True)  # Sort chronologically

            # Create features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['oi_change'] = df['open_interest'].pct_change()
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['rsi_change'] = df['rsi'].diff()
            df = df.dropna()

            if df.empty:
                logger.error("No data available after feature engineering for live inference")
                return False

            # Use only the latest row for prediction
            latest_row = df.iloc[-1:]
            features = [
                'open', 'high', 'low', 'close', 'volume', 'open_interest',
                'tvi', 'obv', 'rsi', 'pvi', 'pvt',
                'price_change', 'volume_change', 'oi_change',
                'ma5', 'ma20', 'rsi_change'
            ]
            
            X = latest_row[features]
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()
            
            if X.empty:
                logger.error("Latest data has invalid features for prediction")
                return False

            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            prediction_proba = model.predict_proba(X_scaled)[0]

            # Convert timestamp to str if it's a pandas Timestamp
            ts_value = latest_row['"time"'].iloc[0]
            if isinstance(ts_value, pd.Timestamp):
                ts_value = str(ts_value)

            # Create result
            result = {
                'timestamp': ts_value,
                'symbol': symbol_used,
                'current_price': latest_row['close'].iloc[0],
                'prediction': 'BUY' if prediction == 1 else 'SELL',
                'confidence': float(np.max(prediction_proba)),
                'buy_probability': float(prediction_proba[1]),
                'sell_probability': float(prediction_proba[0]),
                'rsi': float(latest_row['rsi'].iloc[0]),
                'volume': int(latest_row['volume'].iloc[0]),
                'ma5': float(latest_row['ma5'].iloc[0]),
                'ma20': float(latest_row['ma20'].iloc[0])
            }

            # Save live prediction
            live_predictions_path = 'ml_data/live_predictions.csv'
            result_df = pd.DataFrame([result])
            
            if os.path.exists(live_predictions_path):
                existing_df = pd.read_csv(live_predictions_path)
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
            
            result_df.to_csv(live_predictions_path, index=False)
            
            logger.info(f"Live inference completed: {result['prediction']} with {result['confidence']:.2%} confidence")
            return True, result
            
        except Exception as e:
            logger.error(f"Error running live inference: {str(e)}")
            return False, None

    def run(self):
        try:
            logger.info("Starting data preparation for ML/LLM applications")
            if self.prepare_scalping_data():
                logger.info("Scalping data preparation completed successfully")
            else:
                logger.error("Scalping data preparation failed")
            if self.create_feature_engineering_functions():
                logger.info("Feature engineering functions created successfully")
            else:
                logger.error("Feature engineering functions creation failed")
            if self.create_scalping_signal_generator():
                logger.info("Scalping signal generator created successfully")
            else:
                logger.error("Scalping signal generator creation failed")
            if self.prepare_ml_training_data():
                logger.info("ML training data preparation completed successfully")
            else:
                logger.error("ML training data preparation failed")
            if self.prepare_llm_data():
                logger.info("LLM data preparation completed successfully")
            else:
                logger.error("LLM data preparation failed")
            logger.info("Data preparation completed")
            return True
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            return False
        finally:
            if self.db_conn:
                self.db_cursor.close()
                db_pool.putconn(self.db_conn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation for Scalping ML/LLM Application")
    parser.add_argument('--offline', action='store_true', help='Run in offline mode with historical data file')
    parser.add_argument('--input', type=str, help='Path to historical data file (CSV/JSON)')
    parser.add_argument('--llm-all-data', action='store_true', help='Use ALL available data for LLM/market summary instead of just last 30 minutes')
    parser.add_argument('--symbol', type=str, help='Instrument symbol for ML training')
    parser.add_argument('--expiry', type=str, help='Expiry date for ML training (YYYY-MM-DD)')
    parser.add_argument('--strike', type=float, help='Strike price for ML training')
    parser.add_argument('--option_type', type=str, help='Option type (CE/PE) for ML training')
    args = parser.parse_args()

    expiry_val = None
    if args.expiry:
        try:
            expiry_val = pd.to_datetime(args.expiry).date()
        except Exception:
            expiry_val = None
    strike_val = float(args.strike) if args.strike is not None else None

    if args.offline and args.input:
        logger.info(f"Running in offline mode with file: {args.input}")
        try:
            if args.input.endswith('.csv'):
                df = pd.read_csv(args.input)
                logger.info(f"Loaded {len(df)} rows from CSV")
            else:
                logger.warning("Only CSV input supported for offline mode in this script")
        except Exception as e:
            logger.error(f"Error loading offline input: {str(e)}")
        dp = DataPreparation(llm_all_data=args.llm_all_data)
        dp.run()
    else:
        dp = DataPreparation(llm_all_data=args.llm_all_data)
        dp.prepare_ml_training_data(
            symbol=args.symbol,
            expiry=expiry_val,
            strike=strike_val,
            option_type=args.option_type
        )