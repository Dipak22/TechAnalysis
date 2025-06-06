import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from datetime import datetime, timedelta
from my_stocks import my_stocks
import warnings
warnings.filterwarnings("ignore")

# 1. Download Latest 3 Years of Data
def download_recent_data(tickers, years=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

# 2. ARIMA-GARCH Forecast for Next Week with Dates
def arima_garch_forecast_with_dates(series, forecast_horizon=5):
    # Log returns (stationary data)
    returns = np.log(series).diff().dropna()
    
    # Fit ARIMA(1,1,1)
    model_arima = ARIMA(returns, order=(1,1,1))
    results_arima = model_arima.fit()
    residuals = results_arima.resid
    
    # Fit GARCH(1,1)
    model_garch = arch_model(residuals, vol='Garch', p=1, q=1)
    results_garch = model_garch.fit(disp='off')
    
    # Forecast returns
    arima_forecast = results_arima.forecast(steps=forecast_horizon)
    garch_forecast = results_garch.forecast(horizon=forecast_horizon)
    volatility = np.sqrt(garch_forecast.variance.values[-1, :])
    
    # Generate next week's dates (Monday to Friday)
    last_date = series.index[-1]
    next_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_horizon,
        freq='B'  # Business days only
    )
    
    # Convert returns to price forecasts and round to 2 decimals
    forecast_prices = np.round(series.iloc[-1] * np.exp(np.cumsum(arima_forecast)), 2)
    lower_bounds = np.round(forecast_prices * np.exp(-1.96 * volatility), 2)
    upper_bounds = np.round(forecast_prices * np.exp(1.96 * volatility), 2)
    
    # Create DataFrame with dates
    forecast_df = pd.DataFrame({
        'Date': next_dates,
        'Forecast': forecast_prices,
        'Lower_95': lower_bounds,
        'Upper_95': upper_bounds
    })
    
    return forecast_df

# 3. Predict Next Week for All Stocks and Save to CSV
def predict_and_save_next_week(data, filename='next_week_forecasts.csv'):
    all_forecasts = []
    
    for ticker in data.columns:
        series = data[ticker].dropna()
        try:
            forecast_df = arima_garch_forecast_with_dates(series)
            forecast_df['Ticker'] = ticker  # Add stock symbol column
            all_forecasts.append(forecast_df)
        except Exception as e:
            print(f"Failed for {ticker}: {str(e)}")
    
    # Combine all forecasts and save
    if all_forecasts:
        final_df = pd.concat(all_forecasts)
        final_df = final_df[['Ticker', 'Date', 'Forecast', 'Lower_95', 'Upper_95']]
        final_df.to_csv(filename, index=False)
        print(f"Forecasts saved to {filename}")
        return final_df
    else:
        print("No forecasts generated")
        return None

# Main Execution
if __name__ == "__main__":
    # Get data and run predictions
    tickers = my_stocks
    stock_data = download_recent_data(tickers)

    # Output file name
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"next_week_forecasts_{current_date}.txt"
    
    # Get and save forecasts
    forecasts = predict_and_save_next_week(data=stock_data, filename=OUTPUT_FILE)
    
    # Print sample output
    if forecasts is not None:
        print("\nSample Forecasts:")
        print(forecasts.head())