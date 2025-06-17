import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Parameters
symbol = 'ICICIBANK.NS'
period = '4y'

# Fetch data
stck = yf.Ticker(symbol)
df = stck.history(period=period, interval="1d")
df.dropna(inplace=True)
##df = yf.download(symbol, period=period, interval='1d')
#df.dropna(inplace=True)

# Calculate percentage change features
df['Pct_Change_Close'] = df['Close'].pct_change()
df['Pct_Change_Volume'] = df['Volume'].pct_change()
df['Target'] = df['Pct_Change_Close'].shift(-1)

# Drop NaNs from beginning and end
df.dropna(inplace=True)

# Features and target
X = df[['Pct_Change_Close', 'Pct_Change_Volume']]
y = df['Target']

# Train-test split (first 3 years train, last year test)
split_index = int(len(df) * 3 / 4)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.6f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
plt.title(f'{symbol} - Next Day Price Change Prediction')
plt.xlabel('Date')
plt.ylabel('% Change')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
