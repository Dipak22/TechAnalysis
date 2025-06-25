import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sector_mapping import sector_stocks
from my_stocks import my_stocks, PENNY_STOCKS, NEW_STOCKS, SHORT_TERM_STOCKS

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

def get_first_hour_data(ticker, date):
    """Get the first 1-hour trading data for a given Indian stock ticker"""
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
            # Get first 1 hour (9:15 AM to 10:15 AM)
            first_hour = data.between_time('09:15', '10:15')
            
            if len(first_hour) >= 2:  # Need at least 2 data points
                start_price = first_hour.iloc[0]['Close']
                end_price = first_hour.iloc[-1]['Close']
                percent_change = ((end_price - start_price) / start_price) * 100
                return percent_change
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
    return None

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
        return result
        
    except Exception as e:
        print(f"Error loading percentage changes file: {str(e)}")
        return None

def generate_html_report(results, matched_results, previous_day, current_day, positive_threshold, negative_threshold, percentage_data):
    """Generate HTML report with styled output"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Momentum Analysis - {current_day}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .positive {{
                color: green;
                font-weight: bold;
            }}
            .negative {{
                color: red;
                font-weight: bold;
            }}
            .strong-signal {{
                background-color: #f0f0f0;
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid #333;
            }}
            .bullish {{
                background-color: #e8f5e9;
                font-weight: bold;
                font-style: italic;
            }}
            .bearish {{
                background-color: #ffebee;
                font-weight: bold;
                font-style: italic;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .summary {{
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
            }}
            .date-info {{
                background-color: #fff8e1;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 15px;
            }}
        </style>
    </head>
    <body>
        <h1>Indian Stock Market Momentum Analysis</h1>
        <div class="date-info">
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>First Hour Data Date:</strong> {current_day}</p>
            <p><strong>EOD Data Date:</strong> {previous_day}</p>
        </div>
        <p><em>Note: For Indian stocks, script automatically adds .NS for NSE. You can also use .BO suffix for BSE stocks</em></p>
    """

    # All Results Section
    html_content += """
        <h2>All Stock Momentum Indicators</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>EOD % Change</th>
                <th>EOD Trend</th>
    """
    
    if percentage_data is not None:
        html_content += """
                <th>File %Chng</th>
                <th>File Trend</th>
        """
    
    html_content += """
                <th>First Hour %Chg</th>
                <th>First Hour Trend</th>
            </tr>
    """
    
    for result in results:
        html_content += f"""
            <tr>
                <td>{result['Ticker']}</td>
                <td>{result['EOD % Change']}</td>
                <td class="{result['EOD Trend'].lower()}">{result['EOD Trend']}</td>
        """
        if percentage_data is not None and 'File %Chng' in result:
            html_content += f"""
                <td>{result['File %Chng']}</td>
                <td class="{result['File Trend'].lower()}">{result['File Trend']}</td>
            """
        html_content += f"""
                <td>{result['First Hour %Chg']}</td>
                <td class="{result['First Hour Trend'].lower()}">{result['First Hour Trend']}</td>
            </tr>
        """
    
    html_content += """
        </table>
    """

    # Strong Signals Section (where all indicators agree)
    if percentage_data is not None and matched_results:
        html_content += """
        <h2>Strong Signals - All Indicators Agree</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>EOD % Change</th>
                <th>File %Chng</th>
                <th>First Hour %Chg</th>
                <th>Consensus Trend</th>
            </tr>
        """
        
        for result in matched_results:
            trend_class = "bullish" if result['EOD Trend'] == 'Positive' else "bearish"
            trend_text = "BULLISH" if result['EOD Trend'] == 'Positive' else "BEARISH"
            
            html_content += f"""
            <tr class="{trend_class}">
                <td>{result['Ticker']}</td>
                <td>{result['EOD % Change']}</td>
                <td>{result['File %Chng']}</td>
                <td>{result['First Hour %Chg']}</td>
                <td>{trend_text}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """

    # Threshold-based Highlights
    if positive_threshold is not None:
        positive_stocks = [r for r in results if r['EOD % Change'] >= positive_threshold and r['EOD Trend'] == 'Positive']
        if positive_stocks:
            html_content += f"""
            <div class="strong-signal">
                <h3>Strong Positive Momentum (≥ {positive_threshold}%)</h3>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>EOD % Change</th>
                        <th>First Hour %Chg</th>
                    </tr>
            """
            for stock in positive_stocks:
                html_content += f"""
                    <tr>
                        <td>{stock['Ticker']}</td>
                        <td class="positive">{stock['EOD % Change']}</td>
                        <td class="{stock['First Hour Trend'].lower()}">{stock['First Hour %Chg']}</td>
                    </tr>
                """
            html_content += """
                </table>
            </div>
            """
    
    if negative_threshold is not None:
        negative_stocks = [r for r in results if r['EOD % Change'] <= negative_threshold and r['EOD Trend'] == 'Negative']
        if negative_stocks:
            html_content += f"""
            <div class="strong-signal">
                <h3>Strong Negative Momentum (≤ {negative_threshold}%)</h3>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>EOD % Change</th>
                        <th>First Hour %Chg</th>
                    </tr>
            """
            for stock in negative_stocks:
                html_content += f"""
                    <tr>
                        <td>{stock['Ticker']}</td>
                        <td class="negative">{stock['EOD % Change']}</td>
                        <td class="{stock['First Hour Trend'].lower()}">{stock['First Hour %Chg']}</td>
                    </tr>
                """
            html_content += """
                </table>
            </div>
            """

    # Summary Statistics
    html_content += """
    <div class="summary">
        <h2>Summary Statistics</h2>
        <ul>
    """
    
    html_content += f"""
            <li>Total stocks analyzed: {len(results)}</li>
            <li>Stocks with positive EOD momentum: {len([r for r in results if r['EOD Trend'] == 'Positive'])}</li>
            <li>Stocks with negative EOD momentum: {len([r for r in results if r['EOD Trend'] == 'Negative'])}</li>
            <li>Stocks with positive first hour momentum: {len([r for r in results if r['First Hour Trend'] == 'Positive'])}</li>
            <li>Stocks with negative first hour momentum: {len([r for r in results if r['First Hour Trend'] == 'Negative'])}</li>
    """
    
    if percentage_data is not None:
        html_content += f"""
            <li>Stocks with all matching trends: {len(matched_results)}</li>
        """
    
    html_content += """
        </ul>
    </div>
    """

    html_content += """
    </body>
    </html>
    """
    
    return html_content

def analyze_indian_stocks(stock_list, percentage_file=None, positive_threshold=None, negative_threshold=None, output_file='stock_analysis.html'):
    """Analyze Indian stocks with:
    - First hour % change for today
    - Last 40 min change for previous day
    """
    previous_day = get_previous_trading_day()
    #previous_day = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
    current_day = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
    
    # Load percentage changes if file provided
    percentage_data = None
    if percentage_file:
        percentage_data = load_percentage_changes(percentage_file)
    print(f"Loaded {len(percentage_data)} percentage changes from file." if percentage_data is not None else "No percentage data loaded.")

    results = []
    matched_results = []  # For stocks where all indicators agree
    
    for ticker in stock_list:
        # Get EOD data from previous day
        eod_change = get_last_40_minutes_data(ticker, previous_day)
        
        # Get first hour data from current day
        first_hour_change = get_first_hour_data(ticker, current_day)
        
        if eod_change is not None and first_hour_change is not None:
            result = {
                'Ticker': ticker,
                'First Hour %Chg': round(first_hour_change, 2),
                'First Hour Trend': 'Positive' if first_hour_change >= 0 else 'Negative',
                'EOD % Change': round(eod_change, 2),
                'EOD Trend': 'Positive' if eod_change >= 0 else 'Negative'
            }
            
            # Check if we have percentage data for this stock
            if percentage_data is not None:
                # Remove .NS/.BO suffix for matching if present
                base_ticker = ticker.replace('.NS', '').replace('.BO', '')
                match = percentage_data[percentage_data['ticker'].str.replace('.NS', '').str.replace('.BO', '').str.lower() == base_ticker.lower()]
                
                if not match.empty:
                    file_pct_change = match.iloc[0]['%chng']
                    result['File %Chng'] = round(float(file_pct_change), 2)
                    result['File Trend'] = 'Positive' if file_pct_change >= 0 else 'Negative'
                    
                    # Check if all three indicators agree
                    if ((first_hour_change >= 0 and eod_change >= 0 and file_pct_change >= 0) or 
                        (first_hour_change < 0 and eod_change < 0 and file_pct_change < 0)):
                        matched_results.append(result.copy())
            
            results.append(result)
    
    if not results:
        print("No data available for the specified stocks.")
        return
    
    # Generate HTML report
    html_report = generate_html_report(
        results=results,
        matched_results=matched_results,
        previous_day=previous_day,
        current_day=current_day,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_threshold=NEGATIVE_THRESHOLD,
        percentage_data=percentage_data
    )
    
    # Save to HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"Analysis complete. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # List of Indian stock symbols (without .NS/.BO suffix, or with if you prefer)
    indian_stocks = [stock for stocks in sector_stocks.values() for stock in stocks]
    indian_stocks.extend(my_stocks)
    indian_stocks.extend(PENNY_STOCKS)  
    indian_stocks.extend(NEW_STOCKS)
    indian_stocks.extend(SHORT_TERM_STOCKS)
    
    # Path to file containing %chng data
    PERCENTAGE_FILE = "MW-Pre-Open-Market-24-Jun-2025.csv"  # Update this path
    
    # Set your thresholds (positive % and negative %)
    POSITIVE_THRESHOLD = 0.5  # Highlight stocks with >= 0.5% increase
    NEGATIVE_THRESHOLD = -0.5  # Highlight stocks with <= -0.5% decrease
    
    # Output file name
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"institutional_moves_{current_date}.html"
    
    analyze_indian_stocks(
        stock_list=indian_stocks,
        percentage_file=PERCENTAGE_FILE,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_threshold=NEGATIVE_THRESHOLD,
        output_file=OUTPUT_FILE
    )