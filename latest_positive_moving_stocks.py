import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
from sector_mapping import sector_stocks
from my_stocks import my_stocks, PENNY_STOCKS, NEW_STOCKS,SHORT_TERM_STOCKS
import time
import logging

# Configure logging
logging.basicConfig(filename='latest_moving__errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flatten all stocks from all sectors
all_stocks = [stock for stocks in sector_stocks.values() for stock in stocks]
#all_stocks = [ "NEWGEN.NS","MAPMYINDIA.NS","INTELLECT.NS","INSPIRISYS.NS"]
all_stocks.extend(my_stocks)
all_stocks.extend(PENNY_STOCKS) 
all_stocks.extend(NEW_STOCKS)
all_stocks.extend(SHORT_TERM_STOCKS)

buy_signals = []
sell_signals = []

for symbol in all_stocks:
    success = False
    for attempt in range(3):  # Try up to 3 times
        #time.sleep(1)  # Sleep for 1 second between requests
        if success:
            break
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=15)  # Extra data for indicators
            stck = yf.Ticker(symbol)
            df = stck.history(start=start_date, end=end_date, interval='1d')
            df.dropna(inplace=True)
            #df = yf.download(symbol, period="6mo", interval="1d")
            if df.empty:
                continue
            df.dropna(inplace=True)

             # Check % positive days in last 10 days
            df['Pct_Change'] = df['Close'].pct_change()
            last_10 = df['Pct_Change'].tail(10)
            pos_days = (last_10 > 0).sum()
            ratio = pos_days / 10.0

            if ratio> 0.7:
                buy_signals.append(symbol)
            success = True
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt == 2:
                logging.error(f"Failed to process {symbol} after 3 attempts.")
                logging.error(f"Unhandled error processing {symbol}: {e}")

# Save to file
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"latest_signals_{current_date}.txt"

# Save to file
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"latest positive signal Report - {current_date}\n")
    f.write("=" * 40 + "\n\n")

    f.write("Buy Signals :\n")
    f.write("-" * 40 + "\n")
    f.write("\n".join(buy_signals) + "\n\n")

    f.write("Sell Signals :\n")
    f.write("-" * 40 + "\n")
    f.write("\n".join(sell_signals) + "\n")

print(f"Signals saved to {filename}")


