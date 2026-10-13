# 📈 Apple Stock Price Prediction Using Time Series Forecasting

## 📌 Project Overview

This project analyzes historical Apple Inc. (AAPL) stock prices and applies time series forecasting techniques to predict future closing prices.

The application compares four forecasting models — ARIMA, SARIMAX, Prophet, and LSTM — and evaluates their performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

An interactive web application is developed using Streamlit to visualize historical stock trends, evaluate model predictions, and generate future price forecasts.

## 🎯 Objectives

- Analyze historical Apple stock closing prices and trading volume.
- Perform exploratory data analysis (EDA) and visualize stock price trends.
- Calculate 20-day and 50-day moving averages.
- Understand stationarity and time series behavior.
- Build and evaluate ARIMA, SARIMAX, Prophet, and LSTM forecasting models.
- Compare model performance using RMSE and MAE.
- Develop an interactive Streamlit dashboard for stock analysis and forecasting.

## 📂 Project Structure

```text
apple-stock-prediction/
│
├── app.py
├── AAPL.csv
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project uses historical Apple stock market data from 2012 to 2019.

**Dataset features may include:**

- `Date` — Trading date
- `Open` — Opening stock price
- `High` — Highest price during the trading session
- `Low` — Lowest price during the trading session
- `Close` — Closing stock price
- `Adj Close` — Adjusted closing price, if available
- `Volume` — Number of shares traded

The closing price is used as the primary target variable for forecasting.

## 🔍 Exploratory Data Analysis

The project includes:

- Historical closing price visualization
- Trading volume analysis
- 20-day and 50-day moving averages
- Descriptive statistical analysis
- Missing-value handling
- Time series decomposition
- Augmented Dickey-Fuller (ADF) stationarity testing
- Autocorrelation (ACF) and partial autocorrelation (PACF) analysis

## 🤖 Forecasting Models

### 1. ARIMA

AutoRegressive Integrated Moving Average is a statistical model used for time series forecasting. The project uses an ARIMA configuration to model changes in stock prices.

### 2. SARIMAX

Seasonal AutoRegressive Integrated Moving Average with eXogenous factors extends ARIMA with seasonal components. The model configuration is evaluated on the held-out test period.

### 3. Prophet

Prophet is a time series forecasting library that models trends and seasonal patterns. It is used to generate stock price estimates for future dates.

### 4. LSTM

Long Short-Term Memory is a recurrent neural network architecture designed to learn sequential patterns. The model uses historical price sequences to predict subsequent closing prices.

## 📏 Model Evaluation

Model performance is evaluated using two metrics:

**Root Mean Squared Error (RMSE)**

Measures the square root of the average squared prediction error. Lower values indicate better performance, with larger errors receiving more weight.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted prices. Lower values indicate better performance.

The application displays the evaluation metrics and plots actual versus predicted closing prices for the selected model.

## 🖥️ Streamlit Application Features

- Interactive Apple stock analytics dashboard
- Historical closing price visualization
- Moving-average analysis
- Trading volume visualization, when available
- Selection of forecasting models
- Adjustable test-data percentage
- Configurable forecast horizon
- Model evaluation using RMSE and MAE
- Actual-versus-predicted comparison charts
- Future stock price forecasts
- CSV downloads for historical data and forecasts

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Statistical Forecasting:** Statsmodels
- **Time Series Forecasting:** ARIMA, SARIMAX, Prophet
- **Deep Learning:** TensorFlow, Keras, LSTM
- **Machine Learning Utilities:** Scikit-learn
- **Web Application:** Streamlit
- **Development Environment:** Google Colab, VS Code
- **Version Control:** Git, GitHub

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd apple-stock-prediction
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

`http://localhost:8501`

## 📈 Results and Insights

The application evaluates the selected model on a chronological holdout test set and displays RMSE, MAE, and actual-versus-predicted price charts.

The best-performing model should be identified only after evaluating all candidate models on the same test period using a consistent evaluation methodology.

*Note: Actual performance metrics will depend on the dataset, model configurations, training settings, and test period.*

## 🚀 Future Enhancements

- Integrate a live stock market data source.
- Add interactive Plotly visualizations.
- Compare all four models automatically on the same test period.
- Add prediction intervals and forecast uncertainty analysis.
- Improve LSTM performance through hyperparameter tuning.
- Include additional market indicators and technical features.
- Deploy the application using Streamlit Community Cloud.

## ⚠️ Disclaimer

This project is intended for educational and portfolio purposes only. Stock price forecasts are estimates based on historical data and do not guarantee future market performance. This application should not be considered financial or investment advice.

## 👨‍💻 Author

**Harshal Patil**

Data Science | Machine Learning | Time Series Forecasting | Deep Learning

---

⭐ If you find this project useful, consider starring the repository on GitHub.
