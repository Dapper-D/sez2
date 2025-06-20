# NSE Options Trading Analysis and Signal Generation

This project is a comprehensive data pipeline and analysis tool for NSE (National Stock Exchange) options data. It automates the process of fetching, storing, processing, and analyzing options data to generate trading signals. The application features a real-time data collector, a historical data downloader, a robust data processing pipeline, a machine learning model for signal generation, and an interactive web-based GUI for data exploration and analysis.

## Features

*   **Real-time Data Collection**: Ingests live tick data from an NSE data vendor via a WebSocket connection.
*   **Historical Data Downloader**: Fetches historical options data for specified instruments and date ranges.
*   **TimescaleDB Integration**: Utilizes TimescaleDB for efficient storage and querying of time-series data (candlesticks, technical indicators, etc.).
*   **Automated Data Pipeline**: A fully automated pipeline that handles:
    *   Database schema creation and management.
    *   Data collection and storage.
    *   Calculation of various technical indicators (OBV, RSI, TVI, PVI, PVT).
    *   Data preparation for machine learning models.
    *   Pipeline validation and monitoring.
*   **Machine Learning Model**: Includes a machine learning model to generate buy/sell signals based on historical data.
*   **Interactive GUI**: A Streamlit-based web application for:
    *   Exploring options contracts.
    *   Visualizing candlestick data and technical indicators.
    *   Analyzing scalping and ML-based trading signals.
    *   Performing forward testing and live inference.
*   **Modular and Extensible**: The project is designed with a modular architecture, making it easy to extend and customize.

## System Architecture

The application is orchestrated by `app.py`, which initializes and runs all the components in the correct order.

### Step-by-Step Workflow

1. **Database Schema Setup (`schema_design.py`)**
   - Connects to PostgreSQL/TimescaleDB.
   - Creates tables for instruments, candlesticks (1min, 15sec), technical indicators, and trading signals.
   - Converts tables to hypertables and sets retention policies.

2. **Historical Data Download (`historical_downloader.py`)**
   - Reads `Nifty_Input.csv` for instrument/date config.
   - Downloads historical options data and saves as CSVs in `historical_data/`.
   - Imports CSVs into the database.

3. **Real-time Data Collection (`data_collector.py`)**
   - Authenticates with NSE data vendor, downloads tickers, connects to WebSocket.
   - Subscribes to up to 70 tickers (NSE/BSE).
   - Aggregates tick data into 1min/15sec candles, stores in DB.

4. **Indicator Calculation (`indicator_calculator.py`)**
   - Loads 1min candles for each instrument.
   - Calculates indicators (OBV, RSI, TVI, PVI, PVT).
   - Stores results in `technical_indicators` table.

5. **Data Preparation (`data_preparation.py`)**
   - Prepares data for ML/LLM, including feature engineering and scaling.
   - Generates training/testing datasets, supports forward/live testing.
   - Trains RandomForestClassifier, saves model and results.

6. **Pipeline Validation (`pipeline_validator.py`)**
   - Validates schema, data flow, indicator completeness, and retention policies.
   - Generates validation reports.

7. **GUI (`app_gui.py`)**
   - Streamlit app for data exploration, charting, signal analysis, and forward/live test visualization.

## File Structure

```
.
├── app.py                     # Main application orchestrator
├── app_gui.py                 # Streamlit GUI
├── schema_design.py           # Database schema and setup
├── data_collector.py          # Real-time data collection
├── historical_downloader.py   # Historical data downloader
├── indicator_calculator.py    # Technical indicator calculation
├── data_preparation.py        # Data preparation for ML and analysis
├── pipeline_validator.py      # Data pipeline validation
├── test_forward_live.py       # Tests for forward and live testing
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (credentials, DB config)
├── Nifty_Input.csv            # Input for historical data downloader
├── historical_data/           # Directory for historical data CSVs
├── ml_data/                   # Directory for ML datasets, models, and results
├── ...
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   PostgreSQL with TimescaleDB extension
*   Credentials for an NSE data vendor (for real-time and historical data)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the database:**
    *   Install PostgreSQL and the TimescaleDB extension.
    *   Create a new database for the application.

5.  **Configure the environment variables:**
    *   Create a file named `.env` in the root of the project.
    *   Add the following variables to the `.env` file, replacing the placeholder values with your actual credentials and database connection details:
        ```
        DB_NAME=your_db_name
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_HOST=localhost
        DB_PORT=5432

        NSE_LOGIN_ID=your_nse_login_id
        NSE_PRODUCT=your_nse_product
        NSE_API_KEY=your_nse_api_key
        NSE_AUTH_ENDPOINT=https://your_nse_auth_endpoint
        NSE_TICKERS_ENDPOINT=https://your_nse_tickers_endpoint
        NSE_WEBSOCKET_ENDPOINT=wss://your_nse_websocket_endpoint
        ```

## Usage

1.  **Configure the historical data download:**
    *   Edit the `Nifty_Input.csv` file to specify the instruments and date ranges for which you want to download historical data.

2.  **Run the application:**
    ```bash
    python app.py
    ```
    This will start the entire data pipeline, including the historical data download, real-time data collection, and the Streamlit GUI.

3.  **Access the GUI:**
    *   Open your web browser and navigate to the URL provided in the console (usually `http://localhost:8501`).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## 🚦 System Flow

**ALL steps are managed by `app.py` in this exact order:**

1. **Schema Initialization** (`schema_design.py`)
   - Verifies and creates all required tables and views
   - Creates PostgreSQL/TimescaleDB schema with tables for instruments, candle data, and technical indicators

2. **Data Collection** (`data_collector.py`)
   - Downloads tickers and connects to vendor websocket/API
   - Processes tick data into 1-minute and 15-second candles
   - Stores data in TimescaleDB with 7-day retention policy

3. **Indicator Calculation** (`indicator_calculator.py`)
   - Computes technical indicators (OBV, RSI, TVI, PVI, PVT)
   - Inserts results into the `technical_indicators` table

4. **Pipeline Validation** (`pipeline_validator.py`)
   - Ensures all tables and views exist
   - Validates data presence and correctness
   - Checks indicator completeness and data retention

5. **Data Preparation** (`data_preparation.py`)
   - Prepares and exports data for ML/trading
   - Supports parameterized ML dataset generation

## Project Progress So Far
- Set up a data pipeline for Indian market data with a PostgreSQL backend
- Built a Streamlit web interface for exploring options data, viewing candlestick charts, and analyzing trading signals
- Implemented ML-based signal generation and visualization
- Added forward testing and (simulated) live testing features
- Successfully implemented real-time data collection for NSE and BSE stocks
- Added ML model training and prediction capabilities with RandomForestClassifier
- Implemented comprehensive data preparation pipeline for both ML and LLM training
- Added database connection pooling for improved performance
- Implemented proper datetime handling for Streamlit display
- Added feature name tracking in ML pipeline

## Latest Improvements and Fixes
- **Forward Test Date Range Selection:** Users can now select custom start and end dates for forward test analysis in the GUI, enabling more flexible and targeted evaluation of model performance.
- **Contextual Chart Explanations:** All ML training and forward test charts in the GUI now include clear, user-friendly explanations to help users interpret the analytics.
- **Bug Fix - Forward Test Results Display:** Fixed a KeyError in the GUI by ensuring the correct column names ('predicted', 'actual') are used and handling quoted CSV headers.
- **Auto-Trigger Indicator Calculation:** The pipeline now automatically runs indicator calculation if indicators are missing before ML training, preventing data errors and improving robustness.
- **Dependency Update:** Added `seaborn` for confusion matrix plotting in the GUI.
- **Improved Error Handling:** Enhanced error handling and user feedback throughout the GUI, especially for missing data or invalid date selections.
- **Data Serialization:** Fixed PyArrow serialization issues with datetime columns in Streamlit
- **Database Optimization:** Implemented connection pooling to reduce frequent reconnections
- **ML Pipeline:** Added proper feature name handling to eliminate warnings
- **Error Handling:** Enhanced error handling and logging throughout the application
- **Performance:** Optimized database connections and data processing
- **UI Improvements:** Added better data visualization and error messages
- **Code Structure:** Improved code organization and maintainability
- **Datetime Export Robustness:** All data exports to CSV (including live predictions, forward test results, scalping data, and LLM data) now automatically convert pandas Timestamp columns to string. This prevents errors when loading these files in Streamlit or other tools that do not accept pandas Timestamp objects.

## Current Status
- Real-time data collection is operational for both NSE and BSE stocks
- ML model training and prediction pipeline is functional
- Forward testing shows current accuracy of 47.06%
- Streamlit UI is successfully launching and serving the application
- Data preparation pipeline is handling both ML and LLM data effectively
- Model persistence is working (saved as ml_data/model.pkl)
- Forward test datasets are being saved successfully
- Database connections are stable and optimized with connection pooling
- Datetime handling in Streamlit display is fixed
- Feature name warnings in ML pipeline are resolved

## 🗂️ File Structure
```
.
├── app.py                     # Main orchestrator, run this!
├── app_gui.py                 # Interactive Streamlit GUI
├── schema_design.py           # Schema + historical data loader
├── data_collector.py          # Real-time data collection
├── indicator_calculator.py    # Technical indicator computation
├── pipeline_validator.py      # Data validation
├── data_preparation.py        # ML/LLM data preparation
├── requirements.txt           # Python dependencies
├── .env                       # Configuration
├── historical_data/           # Historical candle CSVs
├── ml_data/                   # ML datasets and models
├── llm_data/                  # LLM/market summary data
├── scalping_signals/          # Trading signals
└── validation_reports/        # Validation reports
```

## 🔧 Requirements
- **Windows 10/11**
- **Python 3.9+**
- **PostgreSQL 13+** (with optional TimescaleDB)
- **Vendor API credentials** (NSE/BSE/MCX websocket & REST)
- **Python packages:**
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - (see `requirements.txt` for full list)

## ⚙️ Setup

### 1. Python & Virtual Environment
```powershell
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database
- **Install PostgreSQL** and (optionally) the TimescaleDB extension.
- Create a new database.
- Update your `.env` file with the database credentials (see `.env.example`).

### 3. API Credentials
- Add your data vendor API keys and endpoints to the `.env` file.

## Troubleshooting

### WebSocket Connection Error: `AttributeError: module 'websocket' has no attribute 'WebSocketApp'`

- **Symptom**: The `data_collector.py` script fails with an `AttributeError`, preventing real-time data collection. This is visible in the `data_collector.log`.
- **Cause**: This error is caused by a dependency conflict. The `socketclusterclient` library used in this project requires an outdated version of the `websocket-client` library (`<=0.48.0`). If a newer version of `websocket-client` or the conflicting `websocket` library is installed, it will cause this `AttributeError`.
- **Fix**: The environment must be configured with the specific required versions. The `requirements.txt` file has been updated to lock these dependencies. To fix a broken environment, run the following:
  ```powershell
  pip uninstall websocket websocket-client -y
  pip install -r requirements.txt
  ```
This ensures the correct, compatible versions (`socketclusterclient==1.3.6` and `websocket-client==0.48.0`) are installed.

## ▶️ Starting the Pipeline
Always run:
```powershell
python app.py
```

This will:
- Initialize and validate the schema
- Import historical data
- Start all components in the correct order

## Known Issues and Limitations

- **WebSocket Connection Error:** If you see `module 'websocket' has no attribute 'WebSocketApp'`, ensure the `websocket-client` package is installed and imported correctly in `data_collector.py`.
- **TimescaleDB Retention Policy:** If you see errors about `_timescaledb_config.bgw_policy_drop_chunks` not existing, your TimescaleDB extension or background jobs may not be set up correctly. Re-run the TimescaleDB extension setup and check your database configuration.
- **Data Gaps:** Many "No data" messages for certain instruments/strikes may occur due to market inactivity or data vendor limitations. This is normal, but you may want to improve error handling or filter out instruments with no data.
- **Empty Logs:** If `indicator_calculator.log` or `pipeline_validator.log` are empty, check that logging is enabled and these scripts are being executed as part of the pipeline.
- **Small Forward Test Sample Size:** Forward test accuracy may be reported on small samples. For more robust evaluation, increase the sample size or adjust your data selection.
- **Documentation:** For troubleshooting common errors (websocket, TimescaleDB, data gaps), see the Troubleshooting section below.

## Recent Bug Fixes
1. **PyArrow Serialization Error**
   - Fixed datetime column serialization issues in Streamlit
   - Added proper type conversion for datetime columns
   - Implemented pre-processing step for DataFrames before display
2. **Feature Names Warning**
   - Added feature name tracking in ML pipeline
   - Implemented proper feature name handling in model training
   - Added explicit DataFrame conversion with feature names
3. **Database Connection Issues**
   - Implemented connection pooling
   - Added proper connection management
   - Optimized database access patterns
4. **Forward Test Results Display**
   - Fixed KeyError by using correct column names and handling quoted CSV headers
   - Added robust error handling for missing or malformed data
5. **Indicator Calculation Auto-Trigger**
   - Now automatically runs indicator calculation if missing before ML training
6. **GUI Chart Explanations**
   - Added contextual markdown explanations for all ML and forward test charts
7. **Forward Test Date Range Selection**
   - Users can now select custom start/end dates for forward test analysis in the GUI

## Next Steps
1. **ML Model Improvements**
   - Improve model accuracy beyond current 47.06%
   - Implement additional ML models and ensemble methods
   - Add more sophisticated feature engineering techniques
   - Implement automated model retraining pipeline

2. **Performance Optimization**
   - Add caching for frequently accessed data
   - Implement batch processing for large datasets
   - Add progress indicators for long-running operations

3. **Monitoring and Logging**
   - Add more detailed logging
   - Implement performance metrics collection
   - Add system health monitoring dashboard

4. **UI Enhancements**
   - Add more analytics and batch contract analysis
   - Improve data visualization
   - Add user preferences and settings

5. **Documentation**
   - Add API documentation
   - Create user guide
   - Add deployment instructions

## Troubleshooting Guide
1. **Database Connection Issues**
   - Verify database credentials in .env file
   - Check network connectivity
   - Ensure PostgreSQL service is running
   - Check connection pool status

2. **Data Display Issues**
   - Check datetime column formats
   - Verify data types in DataFrames
   - Ensure proper data preprocessing

3. **ML Model Issues**
   - Verify feature names consistency
   - Check data preprocessing steps
   - Validate model input/output

4. **Performance Issues**
   - Monitor database connection pool
   - Check system resources
   - Verify data caching

## 📝 Important Notes
- **schema_design.py runs FIRST and is critical**
- **Historical data import is always included**
- **All datetimes are stored/processed as UTC and timezone-aware**
- **Start everything using `app.py`**
- **GUI (`app_gui.py`):**
  - Explore contracts, download candles, visualize signals
  - Trigger ML training for any contract slice
  - View ML training summaries and visualizations

## 🧑‍💻 Running as a Service
- Use Windows Task Scheduler or NSSM to run `app.py` as a background service
- Always activate your venv in the task's startup command

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

For any issues, please check the error messages in the Streamlit UI or refer to this README for troubleshooting steps.

## Troubleshooting

### Pandas Timestamp Export Errors
If you previously encountered errors such as:

```
Error running live inference: '2021-09-09 08:29:00-07:00' is of type <class 'pandas._libs.tslibs.timestamps.Timestamp'>, which is not an accepted type. value only accepts: int, float, str, or None. Please convert the value to an accepted type.
```

**This issue is now resolved.** All datetime columns in exported CSVs are automatically converted to string, ensuring compatibility with Streamlit and other consumers. 