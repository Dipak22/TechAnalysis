from operator import itemgetter
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
from pathlib import Path
from MLTechnicalScorer import MLTechnicalScorer
from my_stocks import my_stocks,PENNY_STOCKS
from sector_mapping import sector_stocks
from daily_report import calculate_signals

class StrategyBacktester:
    def __init__(self, capital=1_000_000, top_n=5):
        self.scorer = MLTechnicalScorer(
            short_period=14,
            medium_period=26,
            long_period=50,
            vix_file='hist_india_vix_-18-06-2024-to-17-06-2025.csv'  # Your CSV with all VIX fields
        )
        
        self.capital = capital
        self.top_n = top_n
        self.portfolio = {'cash': capital, 'holdings': {}, 'history': []}
        self.price_cache = {}  # Cache to store recent prices
        self.trade_log = []
        self.weekly_log = []
        self.output_dir = Path("backtest_results")
        self.output_dir.mkdir(exist_ok=True)
        self.current_portfolio_value = capital
        
    def run_backtest(self, start_date, end_date, stock_universe):
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 0:  # Run weekly on Mondays
                #self.log_weekly_status(current_date, "PRE-REBALANCE")
                self.run_weekly_rebalance(current_date, stock_universe)
                self.log_weekly_status(current_date, "POST-REBALANCE")
            current_date += timedelta(days=1)
        
        self.generate_report()

    def log_weekly_status(self, date, phase):
        """Detailed logging of portfolio status"""
        log_entry = {
            'date': date,
            'phase': phase,
            'cash': self.portfolio['cash'],
            'total_value': self.portfolio['cash'],
            'positions': []
        }
        
        # Calculate current positions
        for ticker, position in self.portfolio['holdings'].items():
            current_price = position['last_price']
            #current_price = self.get_current_price(ticker, date)
            #if current_price is None:
            #    current_price = position['last_price']
            
            position_value = position['shares'] * current_price
            log_entry['total_value'] += position_value
            
            position_return = (current_price - position['entry_price']) / position['entry_price'] * 100
            days_held = (date - position['entry_date']).days

            log_entry['positions'].append({
                'ticker': ticker,
                'shares': position['shares'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'position_value': position_value,
                'return_pct': position_return,
                'days_held': days_held,
                'signal': position['signal']  # Last known signal
            })

        self.current_portfolio_value = log_entry['total_value']
        
        self.weekly_log.append(log_entry)
        #self.print_weekly_log(log_entry)

    def print_weekly_log(self, log_entry):
         
        """Formatted console output for weekly status"""
        print(f"\n{'='*50}")
        print(f"DATE: {log_entry['date'].strftime('%Y-%m-%d')} | PHASE: {log_entry['phase']}")
        print(f"CASH: ${log_entry['cash']:,.2f} | TOTAL VALUE: ${log_entry['total_value']:,.2f}")
        print(f"{'-'*50}")
        print("CURRENT POSITIONS:")
        for pos in log_entry['positions']:
            print(f"{pos['ticker']}: {pos['shares']:,.2f} shares | "
                  f"Entry: ${pos['entry_price']:.2f} | "
                  f"Current: ${pos['current_price']:.2f} | "
                  f"Value: ${pos['position_value']:,.2f} | "
                  f"Return: {pos['return_pct']:+.2f}% | "
                  f"Days: {pos['days_held']} | "
                  f"Signal: {pos['signal']}")
        print(f"{'='*50}\n")

    def get_current_signal(self, ticker, date):
        """Get current signal for a ticker"""
        try:
            signal_data = calculate_signals(ticker, date)
            return signal_data['Signal'] if signal_data else "NO SIGNAL"
        except:
            return "ERROR"
    
    def run_weekly_rebalance(self, current_date, stock_universe):
        # Get signals for all stocks
        signals = []
        for ticker in stock_universe:
            try:
                signal = calculate_signals(ticker, current_date)
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
        
        # Sort by score and pick top N
        signals.sort(key=lambda x: float(x['Score']), reverse=True)
        top_stocks = signals[:self.top_n]
        top_tickers = [s['Ticker'] for s in top_stocks]
        
        # Track changes
        opened_positions = []
        closed_positions = []

        # Close positions not in top stocks
        for ticker in list(self.portfolio['holdings'].keys()):
            if ticker not in top_tickers:
                closing_price = float(list(map(itemgetter('Price'), filter(lambda x: x['Ticker'] == ticker, signals)))[0])
                print(f"Closing position for {ticker} on {current_date.strftime('%Y-%m-%d')} at ${closing_price:.2f}")
                #print(names)
                closed_positions.append(self.close_position(ticker, current_date,closing_price))
        
        # Calculate position sizes based on scores (weighted allocation)
        total_score = sum(float(stock['Score']) for stock in top_stocks)
        if total_score > 0:
    
            # Calculate weighted position sizes
            position_sizes = [
                (float(stock['Score']) / total_score) * (self.current_portfolio_value + self.portfolio['cash'])
                for stock in top_stocks
            ]
        else:
            position_sizes = [0] * len(top_stocks)

        # Open/rebalance positions
        for stock, position_size in zip(top_stocks, position_sizes):
            ticker = stock['Ticker']
            current_price = float(stock['Price'])
            
            # Skip if no allocation or price is zero
            if position_size <= 0 or current_price <= 0:
                continue
                
            # Calculate target shares
            target_shares = position_size / current_price
            
            # Handle existing position
            if ticker in self.portfolio['holdings']:
                current_position = self.portfolio['holdings'][ticker]
                current_shares = current_position['shares']
                return_pct = (current_price - current_position['entry_price']) / current_position['entry_price'] * 100
                if stock['Signal'].startswith(('SELL', 'STRONG SELL')) or return_pct <= 0:
                    closed_positions.append(self.close_position(ticker, current_date,current_price))
                else:
                    # Calculate difference
                    shares_diff = target_shares - current_shares
                
                    if shares_diff > 0:  # Need to buy more
                        cost = shares_diff * current_price
                        if cost > self.portfolio['cash']:
                            continue  # Not enough cash
                        opened_positions.append(self.open_position(
                            ticker, shares_diff, current_price, current_date, stock['Signal']
                        ))
                    elif shares_diff < 0:  # Need to sell some
                        closed_shares = -shares_diff
                        closed_positions.append(self.close_partial_position(
                            ticker, closed_shares, current_price, current_date
                        ))

            else:  # New position
                cost = target_shares * current_price
                if cost > self.portfolio['cash']:
                    continue  # Not enough cash
                opened_positions.append(self.open_position(
                    ticker, target_shares, current_price, current_date, stock['Signal']
                ))
    
        # Log transaction details
        self.log_transactions(current_date, opened_positions, closed_positions)
        
        # Record portfolio snapshot
        self.record_portfolio(current_date)
        
        
    def close_partial_position(self, ticker, shares_to_close, current_price, date):
        """Close a portion of a position"""
        if ticker not in self.portfolio['holdings']:
            return None
            
        position = self.portfolio['holdings'][ticker]
        
        # Can't close more shares than we have
        shares_to_close = min(shares_to_close, position['shares'])
        
        proceeds = shares_to_close * current_price
        self.portfolio['cash'] += proceeds
        self.current_portfolio_value -=proceeds
        
        return_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
        days_held = (date - position['entry_date']).days

        # Update remaining position
        position['shares'] -= shares_to_close
        if position['shares'] <= 0:
            del self.portfolio['holdings'][ticker]
        
        return {
            'ticker': ticker,
            'shares': shares_to_close,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'return_pct': return_pct,
            'days_held': days_held
        }


    def log_transactions(self, date, opened, closed):
        """Log detailed transaction information"""
        transaction_log = {
            'date': date,
            'opened': [],
            'closed': []
        }
        
        for pos in opened:
            transaction_log['opened'].append({
                'ticker': pos['ticker'],
                'shares': pos['shares'],
                'price': pos['price'],
                'amount': pos['amount']
            })
        
        for pos in closed:
            transaction_log['closed'].append({
                'ticker': pos['ticker'],
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'exit_price': pos['exit_price'],
                'return_pct': pos['return_pct'],
                'days_held': pos['days_held']
            })

        self.trade_log.append(transaction_log)
        #self.print_transaction_log(transaction_log)

    def print_transaction_log(self, log):
        """Formatted console output for transactions"""
        print(f"\n{'#'*50}")
        print(f"TRANSACTIONS ON {log['date'].strftime('%Y-%m-%d')}")
        
        if log['opened']:
            print("\nOPENED POSITIONS:")
            for pos in log['opened']:
                print(f"+ {pos['ticker']}: {pos['shares']:,.2f} shares @ ${pos['price']:.2f} "
                      f"(Amount: ${pos['amount']:,.2f})")
        
        if log['closed']:
            print("\nCLOSED POSITIONS:")
            for pos in log['closed']:
                print(f"- {pos['ticker']}: {pos['shares']:,.2f} shares | "
                      f"Entry: ${pos['entry_price']:.2f} | "
                      f"Exit: ${pos['exit_price']:.2f} | "
                      f"Return: {pos['return_pct']:+.2f}% | "
                      f"Held: {pos['days_held']} days")
        
        print(f"{'#'*50}\n")
    
    def get_current_price(self, ticker, date):
        """Safe method to get current price with fallback"""
        try:
            # Check cache first
            if ticker in self.price_cache and self.price_cache[ticker]['date'] == date:
                return self.price_cache[ticker]['price']
            
            # Get fresh data
            price_data = yf.Ticker(ticker).history(
                start=date - timedelta(days=5),
                end=date + timedelta(days=1))
            
            if not price_data.empty:
                price = price_data['Close'].iloc[-1]
                self.price_cache[ticker] = {'date': date, 'price': price}
                return price
            return None
        except:
            return None
    
    def open_position(self, ticker, shares, price, date, signal):
        cost = shares * price
        total_shares = 0
        avg_price = 0
        if cost > self.portfolio['cash']:
            shares = self.portfolio['cash'] / price
            cost = shares * price
        if ticker in self.portfolio['holdings']:
            existing_shares = self.portfolio['holdings'][ticker]['shares']
            total_shares = shares + existing_shares
            entry_price = self.portfolio['holdings'][ticker]['entry_price']
            #print(ticker, entry_price, existing_shares, shares, price, total_shares)
            avg_price = (entry_price * existing_shares + price * shares) / total_shares
            date = self.portfolio['holdings'][ticker]['entry_date']  # Keep original entry date
        else:
            total_shares = shares
            avg_price = price

        self.portfolio['holdings'][ticker] = {
            'shares': total_shares,
            'entry_price': avg_price,
            'entry_date': date,
            'last_price': price,
            'signal': signal  # Initialize last known price
        }
        self.portfolio['cash'] -= cost

        return {
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'amount': cost
        }
    
    def close_position(self, ticker, date, current_price=None):
        if ticker in self.portfolio['holdings']:
            position = self.portfolio['holdings'][ticker]
            current_price = current_price
            
            # Use last known price if current unavailable
            if current_price is None:
                current_price = position['last_price']
            
            proceeds = position['shares'] * current_price
            self.portfolio['cash'] += proceeds
            self.current_portfolio_value -= proceeds
            return_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            days_held = (date - position['entry_date']).days
        
            closed_position = {
                'ticker': ticker,
                'shares': position['shares'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'return_pct': return_pct,
                'days_held': days_held
            }
            
            del self.portfolio['holdings'][ticker]
            return closed_position
    
    def record_portfolio(self, date):
        total_value = self.portfolio['cash']
        for ticker, position in self.portfolio['holdings'].items():
            current_price = self.get_current_price(ticker, date)
            
            if current_price is None:
                current_price = position['last_price']
            else:
                # Update last known price
                self.portfolio['holdings'][ticker]['last_price'] = current_price
            
            total_value += position['shares'] * current_price
        
        self.portfolio['history'].append({
            'date': date,
            'value': total_value,
            'holdings': {k: v['shares'] for k,v in self.portfolio['holdings'].items()}
        })
    
    def generate_report(self):
        # Save JSON logs
        with open(self.output_dir / "weekly_log.json", "w") as f:
            json.dump(self.weekly_log, f, indent=2, default=str)
            
        with open(self.output_dir / "trade_log.json", "w") as f:
            json.dump(self.trade_log, f, indent=2, default=str)
        
        # Generate HTML report
        self.generate_html_report()
        
        # Generate performance plot (as before)
        self.generate_performance_plot()

    def generate_html_report(self):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .log-entry {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .phase {{ font-weight: bold; color: #3498db; }}
                .position {{ margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .transaction {{ margin: 15px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
        <h1>Strategy Backtest Report</h1>
            <h2>Weekly Portfolio Snapshots</h2>
            {self.generate_weekly_log_html()}
            
            <h2>Transaction History</h2>
            {self.generate_transaction_log_html()}
            
            <h2>Performance Summary</h2>
            <img src="performance_plot.png" width="800">
        </body>
        </html>
        """
        
        with open(self.output_dir / "report_entire_stocks_1.html", "w") as f:
            f.write(html_content)

    def generate_weekly_log_html(self):
        html_entries = []
        for log in self.weekly_log:
            positions_html = []
            for pos in log['positions']:
                return_class = "positive" if pos['return_pct'] >= 0 else "negative"
                positions_html.append(f"""
                <div class="position">
                    <strong>{pos['ticker']}</strong>: {pos['shares']:,.2f} shares<br>
                    Entry: ${pos['entry_price']:.2f} | Current: ${pos['current_price']:.2f}<br>
                    Value: ${pos['position_value']:,.2f} | 
                    <span class="{return_class}">Return: {pos['return_pct']:+.2f}%</span><br>
                    Days Held: {pos['days_held']} | Signal: {pos['signal']}
                </div>
                """)
            
            html_entries.append(f"""
            <div class="log-entry">
                <h3>Date: {log['date']} | <span class="phase">Phase: {log['phase']}</span></h3>
                <p><strong>Cash:</strong> ${log['cash']:,.2f} | 
                <strong>Total Value:</strong> ${log['total_value']:,.2f}</p>
                <h4>Positions:</h4>
                {"".join(positions_html) if positions_html else "<p>No positions</p>"}
            </div>
            """)

        return "\n".join(html_entries)

    def generate_transaction_log_html(self):
        html_entries = []
        for log in self.trade_log:
            opened_html = []
            for pos in log['opened']:
                opened_html.append(f"""
                <tr>
                    <td>{pos['ticker']}</td>
                    <td>{pos['shares']:,.2f}</td>
                    <td>${pos['price']:.2f}</td>
                    <td>${pos['amount']:,.2f}</td>
                </tr>
                """)
                
            closed_html = []
            for pos in log['closed']:
                return_class = "positive" if pos['return_pct'] >= 0 else "negative"
                closed_html.append(f"""
                <tr>
                    <td>{pos['ticker']}</td>
                    <td>{pos['shares']:,.2f}</td>
                    <td>${pos['entry_price']:.2f}</td>
                    <td>${pos['exit_price']:.2f}</td>
                    <td class="{return_class}">{pos['return_pct']:+.2f}%</td>
                    <td>{pos['days_held']}</td>
                </tr>
                """)
            html_entries.append(f"""
            <div class="transaction">
                <h3>Transactions on {log['date']}</h3>
                
                <h4>Opened Positions</h4>
                {"<p>No positions opened</p>" if not opened_html else f"""
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Price</th>
                        <th>Amount</th>
                    </tr>
                    {"".join(opened_html)}
                </table>
                """}
                 <h4>Closed Positions</h4>
                {"<p>No positions closed</p>" if not closed_html else f"""
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Return</th>
                        <th>Days Held</th>
                    </tr>
                    {"".join(closed_html)}
                </table>
                """}
            </div>
            """)
            
        return "\n".join(html_entries)

    def generate_performance_plot(self):
        # Create performance dataframe
        df = pd.DataFrame(self.portfolio['history'])
        df.set_index('date', inplace=True)
        
        # Calculate metrics
        initial_value = self.capital
        final_value = df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        max_drawdown = (df['value'].cummax() - df['value']).max() / df['value'].cummax().max() * 100
        
        # Plot performance
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['value'], label='Portfolio Value')
        plt.title(f'Strategy Performance\nTotal Return: {total_return:.2f}% | Max Drawdown: {max_drawdown:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid()
        plt.savefig(self.output_dir / "performance_plot.png")
        plt.close()
            

# Example usage
if __name__ == "__main__":
    # Configuration
    start_date = datetime(2025, 5, 15)
    end_date = datetime(2025, 6, 18)
    stock_universe = [stock for stocks in sector_stocks.values() for stock in stocks]
    #stock_universe = my_stocks  # Use your own stock universe
    #stock_universe = PENNY_STOCKS
    # Run backtest
    backtester = StrategyBacktester(capital=1_000_000, top_n=7)
    backtester.run_backtest(start_date, end_date, stock_universe)