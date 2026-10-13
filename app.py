
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prophet and TensorFlow are used only by their respective models.
# If either library fails to import, the app will show a useful error.
st.set_page_config(
    page_title="Apple Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    [data-testid="stMetric"] {
        background-color: #1C2333;
        padding: 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    data = pd.read_csv("AAPL.csv")

    if "Date" not in data.columns or "Close" not in data.columns:
        raise ValueError(
            "AAPL.csv must contain Date and Close columns."
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"])
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset="Date")
    data = data.set_index("Date")

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load AAPL.csv: {error}")
    st.stop()

if len(df) < 100:
    st.error("Please provide at least 100 valid daily observations.")
    st.stop()

# ---------------- HEADER ----------------

st.title("📈 Apple Stock Price Prediction")
st.caption(
    "Historical analysis | Time-series forecasting | Model evaluation"
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("Forecast Settings")

model_name = st.sidebar.selectbox(
    "Select forecasting model",
    ["ARIMA", "SARIMAX", "Prophet", "LSTM"]
)

forecast_days = st.sidebar.slider(
    "Forecast horizon (business days)",
    min_value=5,
    max_value=60,
    value=30,
    step=5
)

test_percentage = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=30,
    value=20,
    step=5
)

# ---------------- DATA PREPARATION ----------------

close = df["Close"].astype(float)
train_size = int(len(close) * (1 - test_percentage / 100))

train = close.iloc[:train_size]
test = close.iloc[train_size:]

# ---------------- OVERVIEW ----------------

st.subheader("Stock Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Close", f"${close.iloc[-1]:,.2f}")
c2.metric("Highest Close", f"${close.max():,.2f}")
c3.metric("Lowest Close", f"${close.min():,.2f}")
c4.metric("Trading Days", f"{len(close):,}")

tab1, tab2, tab3 = st.tabs(
    ["📊 Stock Analysis", "🧠 Model Evaluation", "🔮 Forecast"]
)

with tab1:
    st.subheader("Historical Closing Price")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close.index, close, label="Closing Price", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Moving Averages")

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close.index, close, label="Close", alpha=0.7)
    ax.plot(ma20.index, ma20, label="20-day MA")
    ax.plot(ma50.index, ma50, label="50-day MA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if "Volume" in df.columns:
        st.subheader("Trading Volume")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Volume"], linewidth=0.8)
        ax.set_ylabel("Shares traded")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("View historical data"):
        st.dataframe(df.tail(20), use_container_width=True)

        st.download_button(
            "Download historical data",
            data=df.to_csv().encode("utf-8"),
            file_name="AAPL_historical_data.csv",
            mime="text/csv"
        )

# ---------------- MODEL TRAINING ----------------

def create_sequences(values, sequence_length=60):
    X, y = [], []

    for i in range(sequence_length, len(values)):
        X.append(values[i-sequence_length:i])
        y.append(values[i])

    return np.array(X), np.array(y)


def evaluate_model(model_name, train, test, forecast_days):
    """Train on the chronological training period and evaluate on test data."""

    if model_name == "ARIMA":
        model = ARIMA(train, order=(0, 1, 0)).fit()

        test_pred = np.asarray(
            model.forecast(steps=len(test))
        )

        future_pred = np.asarray(
            model.forecast(steps=forecast_days)
        )

    elif model_name == "SARIMAX":
        model = SARIMAX(
            train,
            order=(0, 1, 0),
            seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=100)

        test_pred = np.asarray(
            model.forecast(steps=len(test))
        )

        future_pred = np.asarray(
            model.forecast(steps=forecast_days)
        )

    elif model_name == "Prophet":
        from prophet import Prophet

        prophet_train = pd.DataFrame({
            "ds": train.index,
            "y": train.values
        })

        model = Prophet()
        model.fit(prophet_train)

        future_test = pd.DataFrame({
            "ds": test.index
        })

        test_pred = model.predict(future_test)["yhat"].to_numpy()

        future_dates = pd.bdate_range(
            start=train.index[-1] + pd.offsets.BDay(1),
            periods=forecast_days
        )

        future_frame = pd.DataFrame({"ds": future_dates})
        future_pred = model.predict(future_frame)["yhat"].to_numpy()

    elif model_name == "LSTM":
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, LSTM

        tf.keras.utils.set_random_seed(42)

        if len(train) <= 60:
            raise ValueError(
                "LSTM requires more than 60 training observations."
            )

        # Fit the scaler only on training data.
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(
            train.to_numpy().reshape(-1, 1)
        )

        X_train, y_train = create_sequences(train_scaled, 60)

        # Use the last 60 training values as context for test predictions.
        combined = np.concatenate([
            train_scaled,
            scaler.transform(test.to_numpy().reshape(-1, 1))
        ])

        X_test, y_test_scaled = create_sequences(
            combined, 60
        )

        X_test = X_test[len(X_train):]
        y_test_scaled = y_test_scaled[len(X_train):]

        tf.keras.backend.clear_session()

        model = Sequential([
            Input(shape=(60, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=0
        )

        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(
            test_pred_scaled
        ).ravel()

        # Recursively forecast future values.
        sequence = train_scaled[-60:].copy()
        future_scaled = []

        for _ in range(forecast_days):
            next_value = model.predict(
                sequence.reshape(1, 60, 1),
                verbose=0
            )[0, 0]

            future_scaled.append(next_value)
            sequence = np.vstack([
                sequence[1:],
                [[next_value]]
            ])

        future_pred = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).ravel()

    else:
        raise ValueError("Unknown forecasting model.")

    actual = test.to_numpy()

    rmse = np.sqrt(mean_squared_error(actual, test_pred))
    mae = mean_absolute_error(actual, test_pred)

    test_results = pd.DataFrame({
        "Actual": actual,
        "Predicted": test_pred
    }, index=test.index)

    future_dates = pd.bdate_range(
        start=train.index[-1] + pd.offsets.BDay(1),
        periods=forecast_days
    )

    future_results = pd.DataFrame({
        "Forecast": future_pred
    }, index=future_dates)

    return rmse, mae, test_results, future_results


# ---------------- EVALUATION AND FORECAST ----------------

with tab2:
    st.subheader("Model Performance")

    st.write(
        f"Selected model: **{model_name}**. "
        f"Training observations: {len(train):,}; "
        f"test observations: {len(test):,}."
    )

    st.info(
        "Models are trained on the earlier portion of the dataset. "
        "Test predictions are evaluated against later observations."
    )

    run_model = st.button(
        "Train Model & Evaluate",
        type="primary",
        key="evaluate_model"
    )

    if run_model:
        with st.spinner(
            f"Training {model_name} and evaluating predictions..."
        ):
            try:
                rmse, mae, test_results, future_results = evaluate_model(
                    model_name, train, test, forecast_days
                )

                st.session_state["stock_results"] = {
                    "model": model_name,
                    "rmse": rmse,
                    "mae": mae,
                    "test": test_results,
                    "future": future_results
                }

            except Exception as error:
                st.error(
                    f"Model training failed: {error}"
                )

    results = st.session_state.get("stock_results")

    if results and results["model"] == model_name:
        m1, m2 = st.columns(2)
        m1.metric("RMSE", f"${results['rmse']:,.3f}")
        m2.metric("MAE", f"${results['mae']:,.3f}")

        st.caption(
            "Lower RMSE and MAE generally indicate better test-set accuracy."
        )

        st.subheader("Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            results["test"].index,
            results["test"]["Actual"],
            label="Actual"
        )
        ax.plot(
            results["test"].index,
            results["test"]["Predicted"],
            label="Predicted"
        )
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            results["test"].tail(30),
            use_container_width=True
        )

with tab3:
    st.subheader("Future Stock Price Forecast")

    st.write(
        f"Forecast horizon: **{forecast_days} business days**. "
        "Train and evaluate the selected model first."
    )

    results = st.session_state.get("stock_results")

    if results and results["model"] == model_name:
        future = results["future"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            close.tail(120).index,
            close.tail(120),
            label="Historical Close"
        )
        ax.plot(
            future.index,
            future["Forecast"],
            label="Forecast",
            linestyle="--"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(future, use_container_width=True)

        st.download_button(
            "Download forecast CSV",
            data=future.to_csv().encode("utf-8"),
            file_name=f"AAPL_{model_name}_forecast.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Go to Model Evaluation and click "
            "'Train Model & Evaluate' to generate a forecast."
        )

st.divider()
st.caption(
    "Educational project using historical stock prices. "
    "Forecasts are estimates, not investment advice."
)
