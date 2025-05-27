import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Parameters
# Parameters
symbol = "ICICIBANK.NS"
period = "4y"

# Fetch data
stck = yf.Ticker(symbol)
df = stck.history(period=period, interval="1d")
df.dropna(inplace=True)

# Calculate percentage change features
df['Pct_Change_Close'] = df['Close'].pct_change().round(6)
df['Pct_Change_Volume'] = df['Volume'].pct_change().round(6)
df['High_Low_Range'] = ((df['High'] - df['Low']) / df['Close']).round(6)
df['Open_Close_Change'] = ((df['Close'] - df['Open']) / df['Open']).round(6)
df['Target'] = df['Pct_Change_Close'].shift(-1).round(6)

# Drop NaNs and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Features and target
X = df[['Pct_Change_Close', 'Pct_Change_Volume', 'High_Low_Range', 'Open_Close_Change']]
y = df['Target']

# Train-test split (first 3 years train, last year test)
split_index = int(len(df) * 3 / 4)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit KNN model with default parameters
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Save the model and scaler
#joblib.dump(knn, 'knn_model.pkl')
#joblib.dump(scaler, 'scaler.pkl')

# Predict
y_pred = knn.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.6f}")

# Top 3 nearest target values for the last test instance
last_point = X_test_scaled[-1].reshape(1, -1)
distances, indices = knn.kneighbors(last_point, n_neighbors=3)
top_targets = y_train.iloc[indices[0]].values
print("Top 3 likely % change in price for next day:", top_targets)

# Predict custom input
def predict_custom(pct_change_price, pct_change_volume, high_low_range, open_close_change):
    input_scaled = scaler.transform([[pct_change_price, pct_change_volume, high_low_range, open_close_change]])
    prediction = knn.predict(input_scaled)[0]
    print(f"Predicted % change in price for next day: {prediction:.6f}")
    return prediction

# Example usage:
# predict_custom(0.01, -0.02, 0.015, 0.005)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
plt.title(f'{symbol} - Next Day Price Change Prediction (KNN)')
plt.xlabel('Date')
plt.ylabel('% Change')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
