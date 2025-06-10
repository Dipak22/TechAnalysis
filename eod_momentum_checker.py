import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sector_mapping import sector_stocks

def get_previous_trading_day():
    """Get the previous trading day (Monday-Friday) for Indian market"""
    today = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    # If today is Monday, go back to Friday
    if today.weekday() == 0:  # Monday
        previous_day = today - timedelta(days=3)
    elif today.weekday() == 6:  # Sunday
        previous_day = today - timedelta(days=2)
    else:
        previous_day = today - timedelta(days=1)
    
    return previous_day.strftime('%Y-%m-%d')

def get_last_40_minutes_data(ticker, date):
    """Get the last 40 minutes of trading data for a given Indian stock ticker"""
    try:
        # For Indian stocks, we need to add .NS (NSE) or .BO (BSE) suffix
        if not ticker.endswith(('.NS', '.BO')):
            ticker += '.NS'  # Default to NSE
        
        stock = yf.Ticker(ticker)
        
        # Convert date to IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        start_date = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=ist)
        end_date = (start_date + timedelta(days=1))
        
        # Get 5-minute interval data
        data = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='5m'
        )
        
        if not data.empty:
            # Indian market hours: 9:15 AM to 3:30 PM IST
            # Get last 40 minutes (2:50 PM to 3:30 PM)
            last_40_min = data.between_time('14:50', '15:30')
            
            if len(last_40_min) >= 2:  # Need at least 2 data points
                start_price = last_40_min.iloc[0]['Close']
                end_price = last_40_min.iloc[-1]['Close']
                percent_change = ((end_price - start_price) / start_price) * 100
                return percent_change
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
    return None

def write_to_file(file, content):
    """Helper function to write content to file and print to console"""
    try:
        print(content)
        file.write(content + "\n")
    except UnicodeEncodeError:
        # Replace Unicode characters with ASCII equivalents
        safe_content = content.replace('≥', '>=').replace('≤', '<=')
        print(safe_content)
        file.write(safe_content + "\n")

def load_percentage_changes(file_path):
    """Load percentage changes from a file and clean column names"""
    try:
        # Try reading with different encodings if needed
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
        # Clean column names by removing whitespace and newline characters
        df.columns = df.columns.str.strip().str.replace('\n', '').str.lower()
        
        # Check if required columns exist (using cleaned names)
        required_columns = {'symbol', '%chng'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"File is missing required columns: {missing}")
            
        # Select and rename columns for consistency
        result = df[['symbol', '%chng']].copy()
        result.columns = ['ticker', '%chng']  # Standardize column names
        
        # Clean ticker names by removing whitespace and newlines
        result['ticker'] = result['ticker'].str.strip().str.replace('\n', '')
        
        # Convert %chng to numeric, handling any non-numeric values
        result['%chng'] = pd.to_numeric(result['%chng'], errors='coerce')
        
        # Drop rows with invalid %chng values
        result = result.dropna(subset=['%chng'])
        #print(result)
        return result
        
    except Exception as e:
        print(f"Error loading percentage changes file: {str(e)}")
        return None

def analyze_indian_stocks(stock_list, percentage_file=None, positive_threshold=None, negative_threshold=None, output_file='stock_analysis.txt'):
    """Analyze Indian stocks for price changes in last 40 minutes of trading"""
    previous_day = get_previous_trading_day()
    
    # Load percentage changes if file provided
    percentage_data = None
    if percentage_file:
        percentage_data = load_percentage_changes(percentage_file)

    # Open file with UTF-8 encoding to handle special characters
    with open(output_file, 'w', encoding='utf-8') as f:
        write_to_file(f, f"Indian Stock Market Analysis - Last 40 Minutes Momentum")
        write_to_file(f, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_to_file(f, f"Trading Day Analyzed: {previous_day}\n")
        write_to_file(f, "Note: For Indian stocks, script automatically adds .NS for NSE")
        write_to_file(f, "You can also use .BO suffix for BSE stocks\n")
        
        results = []
        matched_results = []  # For stocks where both indicators agree
        
        for ticker in stock_list:
            change = get_last_40_minutes_data(ticker, previous_day)
            if change is not None:
                result = {
                    'Ticker': ticker,
                    'EOD % Change': round(change, 2),
                    'EOD Trend': 'Positive' if change >= 0 else 'Negative'
                }
                
                # Check if we have percentage data for this stock
                if percentage_data is not None:
                    #print(percentage_data.head())  # Debugging line to check data
                    # Remove .NS/.BO suffix for matching if present
                    base_ticker = ticker.replace('.NS', '').replace('.BO', '')
                    match = percentage_data[percentage_data['ticker'].str.replace('.NS', '').str.replace('.BO', '').str.lower() == base_ticker.lower()]
                    
                    if not match.empty:
                        file_pct_change = match.iloc[0]['%chng']
                        result['File %Chng'] = round(float(file_pct_change), 2)
                        result['File Trend'] = 'Positive' if file_pct_change >= 0 else 'Negative'
                        
                        # Check if both indicators agree
                        if (change >= 0 and file_pct_change >= 0) or (change < 0 and file_pct_change < 0):
                            matched_results.append(result.copy())
                
                results.append(result)
        
        if not results:
            write_to_file(f, "No data available for the specified stocks.")
            return
        
        df = pd.DataFrame(results).sort_values('EOD % Change', ascending=False)
        
        # Write all results
        write_to_file(f, "All Stock Changes (Last 40 Minutes):")
        write_to_file(f, df.to_string(index=False))
        write_to_file(f, "\n")
        
        # Write matched results if available
        if percentage_data is not None and matched_results:
            matched_df = pd.DataFrame(matched_results).sort_values('EOD % Change', ascending=False)
            write_to_file(f, "Stocks with Matching Trends (EOD and File %Chng):")
            write_to_file(f, matched_df.to_string(index=False))
            write_to_file(f, "\n")
        
        # Filter and highlight stocks based on thresholds
        if positive_threshold is not None:
            positive_stocks = df[(df['EOD % Change'] >= positive_threshold) & (df['EOD Trend'] == 'Positive')]
            if not positive_stocks.empty:
                write_to_file(f, f"STRONG POSITIVE MOMENTUM (>= {positive_threshold}%):")
                write_to_file(f, positive_stocks.to_string(index=False))
            else:
                write_to_file(f, f"No stocks with positive change >= {positive_threshold}%")
            write_to_file(f, "\n")
        
        if negative_threshold is not None:
            negative_stocks = df[(df['EOD % Change'] <= negative_threshold) & (df['EOD Trend'] == 'Negative')]
            if not negative_stocks.empty:
                write_to_file(f, f"STRONG NEGATIVE MOMENTUM (<= {negative_threshold}%):")
                write_to_file(f, negative_stocks.to_string(index=False))
            else:
                write_to_file(f, f"No stocks with negative change <= {negative_threshold}%")
            write_to_file(f, "\n")
        
        # Add summary statistics
        write_to_file(f, "Summary Statistics:")
        write_to_file(f, f"Total stocks analyzed: {len(results)}")
        write_to_file(f, f"Stocks with positive momentum: {len(df[df['EOD Trend'] == 'Positive'])}")
        write_to_file(f, f"Stocks with negative momentum: {len(df[df['EOD Trend'] == 'Negative'])}")
        
        if percentage_data is not None:
            write_to_file(f, f"Stocks with matching trends: {len(matched_results)}")
        
        write_to_file(f, f"\nAnalysis complete. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # List of Indian stock symbols (without .NS/.BO suffix, or with if you prefer)
    indian_stocks = [stock for stocks in sector_stocks.values() for stock in stocks]
    #indian_stocks =["LT.NS", "RELIANCE.NS","ICICIBANK.NS"]
    # Path to file containing %chng data
    PERCENTAGE_FILE = "../MW-Pre-Open-Market-10-Jun-2025.csv"  # Update this path
    
    # Set your thresholds (positive % and negative %)
    POSITIVE_THRESHOLD = 0.5  # Highlight stocks with >= 0.5% increase
    NEGATIVE_THRESHOLD = -0.5  # Highlight stocks with <= -0.5% decrease
    
    # Output file name
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"end_of_day_momentum_analysis_{current_date}.txt"
    
    analyze_indian_stocks(
        stock_list=indian_stocks,
        percentage_file=PERCENTAGE_FILE,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_threshold=NEGATIVE_THRESHOLD,
        output_file=OUTPUT_FILE
    )