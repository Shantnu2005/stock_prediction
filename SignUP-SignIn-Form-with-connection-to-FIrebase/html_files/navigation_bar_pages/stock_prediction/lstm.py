import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
df = pd.read_csv('ULTRACEMCO.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Feature engineering
df['Range'] = df['High'] - df['Low']
df['MA5'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
df['Volume'] = np.log1p(df['Volume'])

# Feature selection
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Range', 'MA5']
target = 'Close'
data = df[features].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences with multi-step targets
def create_sequences(data, window_size=30, prediction_steps=7):
    X, y = [], []
    for i in range(window_size, len(data) - prediction_steps + 1):
        X.append(data[i - window_size:i])
        y.append(data[i:i + prediction_steps, features.index(target)])
    return np.array(X), np.array(y)

window_size = 30
prediction_steps = 7
X, y = create_sequences(scaled_data, window_size, prediction_steps)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, len(features))),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(prediction_steps)
])

# Compile model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Train model
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

# Inverse transformation
def inverse_transform(predictions, scaler, feature_index):
    n_samples, n_steps = predictions.shape
    flat_predictions = predictions.flatten()
    dummy = np.zeros((len(flat_predictions), scaled_data.shape[1]))
    dummy[:, feature_index] = flat_predictions
    dummy = scaler.inverse_transform(dummy)
    return dummy[:, feature_index].reshape(n_samples, n_steps)

# Generate predictions
test_predictions = model.predict(X_test)
actual_prices = inverse_transform(y_test, scaler, features.index(target))
predicted_prices = inverse_transform(test_predictions, scaler, features.index(target))

# Main graph visualization
plt.figure(figsize=(16, 8))

# Training data
train_dates = df['Date'][window_size:split+window_size]
plt.plot(train_dates, 
         df['Close'][window_size:split+window_size], 
         label='Actual Training Prices', 
         color='blue',
         linewidth=2)

# Training predictions
train_pred = inverse_transform(model.predict(X_train), scaler, features.index(target))
plt.plot(train_dates, 
         train_pred[:,0],  # First prediction step
         label='Predicted Training Prices', 
         color='cyan',
         linestyle='--',
         linewidth=1.5)

# Testing data
test_dates = df['Date'][split+window_size:split+window_size+len(X_test)+prediction_steps-1]
plt.plot(test_dates,
         df['Close'][split+window_size:split+window_size+len(X_test)+prediction_steps-1],
         label='Actual Testing Prices',
         color='green',
         linewidth=2)

# Testing predictions (plot each 7-day sequence)
for i in range(len(X_test)):
    start_idx = split + window_size + i
    end_idx = start_idx + prediction_steps
    pred_dates = df['Date'][start_idx:end_idx]
    plt.plot(pred_dates, 
             predicted_prices[i], 
             color='red', 
             linestyle='--',
             alpha=0.6,
             linewidth=1.5)

# Last week markers
last_week_dates = df['Date'][-7:]
plt.scatter(last_week_dates, df['Close'][-7:],
           label='Actual (Daily)',
           color='black',
           marker='o',
           zorder=5)

plt.scatter(test_dates[-7:], 
            predicted_prices[-1][-7:],
            label='Predicted (Daily)',
            color='purple',
            marker='x',
            s=100,
            zorder=6)

plt.title('HDFC Stock Price Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (INR)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7-day market-aware prediction
def get_market_dates(last_date, days=7):
    dates = []
    current = last_date
    while len(dates) < days:
        current += pd.DateOffset(days=1)
        if current.weekday() < 5:  # Monday-Friday
            dates.append(current)
    return dates[:days]

last_sequence = scaled_data[-window_size:]
next_pred = inverse_transform(model.predict(np.expand_dims(last_sequence, 0)), scaler, features.index(target))[0]
future_dates = get_market_dates(df['Date'].iloc[-1])

# Plot 7-day forecast
plt.figure(figsize=(12, 6))
plt.plot(future_dates, next_pred,
         marker='o',
         linestyle='-',
         color='orange',
         label='7-Day Forecast')

# Add annotations
for date, price in zip(future_dates, next_pred):
    plt.annotate(f'₹{price:.2f}', (date, price),
                 xytext=(5,5), textcoords='offset points',
                 fontsize=9, 
                 arrowprops=dict(arrowstyle='-', alpha=0.3))

plt.title('Next 7 Market Days Prediction', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (INR)', fontsize=12)
plt.grid(True, alpha=0.2)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print predictions
print("Next 7 Market Days Forecast:")
for date, price in zip(future_dates, next_pred):
    print(f"{date.strftime('%Y-%m-%d')}: ₹{price:.2f}")

# Calculate metrics
mae = np.mean(np.abs(actual_prices.flatten()[:len(test_dates)] - predicted_prices.flatten()[:len(test_dates)]))
rmse = np.sqrt(np.mean((actual_prices.flatten()[:len(test_dates)] - predicted_prices.flatten()[:len(test_dates)])**2))
print(f"\nMAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")