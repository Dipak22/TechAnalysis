import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from MLTechnicalScorer import MLTechnicalScorer
from my_stocks import my_stocks
from sector_mapping import sector_stocks
from daily_report import FixedPSARIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator

def calculate_signals(ticker, current_date, short_period=14, medium_period=26, long_period=50):
    """Calculate momentum, volume, and trend signals across three timeframes with PSAR"""
    try:
        # Download data - extended for long-term indicators
        end_date = current_date
        start_date = end_date - timedelta(days=long_period*3)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < long_period:
            return None

        # Initialize indicators dictionary
        indicators = {}

        # Short-term indicators (14 days)
        indicators['short'] = {
            'RSI': RSIIndicator(close=df['Close'], window=short_period).rsi(),
            'ROC': ROCIndicator(close=df['Close'], window=short_period).roc(),
            'Stoch_%K': StochasticOscillator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=short_period,
                smooth_window=3
            ).stoch(),
            'SMA': SMAIndicator(close=df['Close'], window=short_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=short_period).ema_indicator(),
            'ATR': AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=short_period
            ).average_true_range(),
            'PSAR': FixedPSARIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                step=0.02,  # Standard acceleration
                max_step=0.2
            ).psar()
        }

        # Medium-term indicators (26 days)
        indicators['medium'] = {
            'RSI': RSIIndicator(close=df['Close'], window=medium_period).rsi(),
            'ROC': ROCIndicator(close=df['Close'], window=medium_period).roc(),
            'SMA': SMAIndicator(close=df['Close'], window=medium_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=medium_period).ema_indicator(),
            'MACD': MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd(),
            'MACD_diff': MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff(),
            'ADX': ADXIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=medium_period
            ).adx(),
            'PSAR': FixedPSARIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                step=0.015,  # Slightly slower
                max_step=0.15
            ).psar()
        }
        
        # Long-term indicators (50 days)
        indicators['long'] = {
            'SMA': SMAIndicator(close=df['Close'], window=long_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=long_period).ema_indicator(),
            'BB': BollingerBands(close=df['Close'], window=long_period, window_dev=2),
            'VWAP': VolumeWeightedAveragePrice(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume'],
                window=long_period
            ).volume_weighted_average_price(),
            'PSAR': FixedPSARIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                step=0.01,  # Slow acceleration for long-term
                max_step=0.1
            ).psar()
        }

        # Volume indicators
        volume_indicators = {
            'OBV': OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume(),
            'ADI': AccDistIndexIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume']
            ).acc_dist_index(),
            'Volume_MA': df['Volume'].rolling(20).mean(),
            'Volume_Spike': df['Volume'] > (df['Volume'].rolling(20).mean() * 2)
        }
        
        # Get latest values
        latest = {
            'price': df['Close'].iloc[-1],
            'short': {k: v.iloc[-1] for k, v in indicators['short'].items()},
            'medium': {k: v.iloc[-1] if not isinstance(v, dict) else {sk: sv.iloc[-1] for sk, sv in v.items()} 
                      for k, v in indicators['medium'].items()},
            'long': {k: v.iloc[-1] if not hasattr(v, 'bollinger_hband') else {
                'upper': v.bollinger_hband().iloc[-1],
                'middle': v.bollinger_mavg().iloc[-1],
                'lower': v.bollinger_lband().iloc[-1],
                'percent': np.divide(
                                (df['Close'].iloc[-1] - v.bollinger_lband().iloc[-1]),
                                (v.bollinger_hband().iloc[-1] - v.bollinger_lband().iloc[-1]),
                                out=np.zeros(1),
    where=(v.bollinger_hband().iloc[-1] - v.bollinger_lband().iloc[-1]) != 0
)[0]
            } for k, v in indicators['long'].items()},
            'volume': {k: v.iloc[-1] for k, v in volume_indicators.items()}
        }

        # Previous values for trend analysis
        prev = {
            'short': {k: v.iloc[-2] for k, v in indicators['short'].items()},
            'medium': {k: v.iloc[-2] if not isinstance(v, dict) else {sk: sv.iloc[-2] for sk, sv in v.items()} 
                     for k, v in indicators['medium'].items()},
        }
        
        # Price changes
        price_changes = {
            'short': (latest['price'] - df['Close'].iloc[-short_period]) / df['Close'].iloc[-short_period] * 100,
            'medium': (latest['price'] - df['Close'].iloc[-medium_period]) / df['Close'].iloc[-medium_period] * 100,
            'long': (latest['price'] - df['Close'].iloc[-long_period]) / df['Close'].iloc[-long_period] * 100
        }

        # Trend analysis with PSAR
        trends = {
            # Moving average trends
            'short_term_up': latest['price'] > latest['short']['SMA'] > latest['short']['EMA'],
            'medium_term_up': latest['price'] > latest['medium']['SMA'] > latest['medium']['EMA'],
            'long_term_up': latest['price'] > latest['long']['SMA'] > latest['long']['EMA'],
            'golden_cross': latest['short']['SMA'] > latest['medium']['SMA'] and prev['short']['SMA'] <= prev['medium']['SMA'],
            'death_cross': latest['short']['SMA'] < latest['medium']['SMA'] and prev['short']['SMA'] >= prev['medium']['SMA'],
            'macd_bullish': latest['medium']['MACD_diff'] > 0 and prev['medium']['MACD_diff'] <= 0,
            'macd_bearish': latest['medium']['MACD_diff'] < 0 and prev['medium']['MACD_diff'] >= 0,
            'adx_strength': latest['medium']['ADX'] > 25,
            
            # PSAR trends
            'sar_short_bullish': latest['price'] > latest['short']['PSAR'],
            'sar_medium_bullish': latest['price'] > latest['medium']['PSAR'],
            'sar_long_bullish': latest['price'] > latest['long']['PSAR'],
            'sar_short_bearish': latest['price'] < latest['short']['PSAR'],
            'sar_medium_bearish': latest['price'] < latest['medium']['PSAR'],
            'sar_long_bearish': latest['price'] < latest['long']['PSAR'],
        }
        
        # Momentum signals
        momentum = {
            'rsi_short': latest['short']['RSI'],
            'rsi_medium': latest['medium']['RSI'],
            'stoch_overbought': latest['short']['Stoch_%K'] > 80,
            'stoch_oversold': latest['short']['Stoch_%K'] < 20,
            'roc_short': latest['short']['ROC'],
            'roc_medium': latest['medium']['ROC'],
            'bb_position': latest['long']['BB']['percent'],
            'atr': latest['short']['ATR']
        }

        # Volume signals
        volume = {
            'obv_trend': '↑' if latest['volume']['OBV'] > volume_indicators['OBV'].iloc[-2] else '↓',
            'adi_trend': '↑' if latest['volume']['ADI'] > volume_indicators['ADI'].iloc[-2] else '↓',
            'volume_spike': latest['volume']['Volume_Spike'],
            'vwap_relation': 'above' if latest['price'] > latest['long']['VWAP'] else 'below'
        }
        
        # Generate signal reasons
        signal_reasons = []
        
        # Trend reasons
        if trends['golden_cross']:
            signal_reasons.append("Golden Cross (Short > Medium MA)")
        if trends['death_cross']:
            signal_reasons.append("Death Cross (Short < Medium MA)")
        if trends['sar_short_bullish'] and trends['sar_medium_bullish'] and trends['sar_long_bullish']:
            signal_reasons.append("PSAR Bullish (All Timeframes)")
        elif trends['sar_short_bullish'] or trends['sar_medium_bullish']:
            signal_reasons.append("PSAR Bullish (Partial)")
        if trends['adx_strength']:
            signal_reasons.append("Strong Trend (ADX > 25)")

        # Momentum reasons
        if momentum['rsi_short'] > 70:
            signal_reasons.append(f"Short-term RSI Overbought ({momentum['rsi_short']:.1f})")
        if momentum['rsi_short'] < 30:
            signal_reasons.append(f"Short-term RSI Oversold ({momentum['rsi_short']:.1f})")
        if momentum['stoch_overbought']:
            signal_reasons.append("Stochastic Overbought")
        if momentum['stoch_oversold']:
            signal_reasons.append("Stochastic Oversold")
        if momentum['bb_position'] > 0.8:
            signal_reasons.append(f"BB Upper Band ({momentum['bb_position']:.2%})")
        if momentum['bb_position'] < 0.2:
            signal_reasons.append(f"BB Lower Band ({momentum['bb_position']:.2%})")
            
        # Volume reasons
        if volume['volume_spike']:
            signal_reasons.append("Volume Spike (2x MA)")
        if volume['obv_trend'] == '↑' and volume['volume_spike']:
            signal_reasons.append("OBV Uptick with Volume")
        if volume['vwap_relation'] == 'above':
            signal_reasons.append("Price Above VWAP")
            
        # Price change reasons
        if price_changes['short'] > 10:
            signal_reasons.append(f"Short-term Up {price_changes['short']:.1f}%")
        if price_changes['short'] < -5:
            signal_reasons.append(f"Short-term Down {abs(price_changes['short']):.1f}%")
        
        # Generate composite score (0-100)
        score = 50  # Neutral starting point

        # Trend factors (30%)
        score += 5 if trends['short_term_up'] else -10
        score += 5 if trends['medium_term_up'] else -5
        score += 5 if trends['long_term_up'] else -3
        score += 5 if trends['golden_cross'] else (-10 if trends['death_cross'] else 0)
        score += 5 if trends['macd_bullish'] else (-5 if trends['macd_bearish'] else 0)
        score += 5 if trends['adx_strength'] else 0
        
        # PSAR factors (15%)
        score += 5 if trends['sar_short_bullish'] else -5
        score += 5 if trends['sar_medium_bullish'] else -5
        score += 5 if trends['sar_long_bullish'] else -5
        
        # Momentum factors (25%)
        score += 5 * ((momentum['rsi_short'] - 50) / 10)  # Normalized RSI contribution
        score += 5 if momentum['stoch_oversold'] else (-5 if momentum['stoch_overbought'] else 0)
        score += 5 * (momentum['roc_short'] / 5)  # Normalized ROC contribution
        score += 5 * (momentum['roc_medium'] / 3)  # Normalized ROC contribution
        score += 5 * (momentum['bb_position'] - 0.5)  # BB position contribution
        
        # Volume factors (20%)
        score += 10 if volume['obv_trend'] == '↑' else -10
        score += 5 if volume['adi_trend'] == '↑' else -3
        score += 5 if volume['volume_spike'] and volume['obv_trend'] == '↑' else (
                 -5 if volume['volume_spike'] and volume['obv_trend'] == '↓' else 0)
        score += 5 if volume['vwap_relation'] == 'above' else -3

        # Price action factors (10%)
        score += 5 * (price_changes['short'] / 5)  # Normalized short-term change
        score += 3 * (price_changes['medium'] / 3)  # Normalized medium-term change
        score += 2 * (price_changes['long'] / 2)  # Normalized long-term change
        
        # Cap score between 0 and 100
        #score = max(0, min(100, score))
        
        # Generate trading signal
        signal = "HOLD"
        signal_strength = ""
        
        # STRONG BUY: All PSAR timeframes bullish + high score
        if (score > 80 and 
            trends['sar_short_bullish'] and 
            trends['sar_medium_bullish'] and 
            trends['sar_long_bullish'] and 
            volume['obv_trend'] == '↑'):
            signal = "STRONG BUY"
            signal_strength = "(Multi-Timeframe PSAR Confirmed)"
        
        # BUY: Partial PSAR confirmation
        elif (score > 65 and 
              (trends['sar_short_bullish'] or trends['sar_medium_bullish']) and 
              volume['obv_trend'] == '↑'):
            signal = "BUY"
            signal_strength = "(PSAR Bullish)"

        # STRONG SELL: All PSAR timeframes bearish + low score
        elif (score < 20 and 
              trends['sar_short_bearish'] and 
              trends['sar_medium_bearish'] and 
              trends['sar_long_bearish'] and 
              volume['obv_trend'] == '↓'):
            signal = "STRONG SELL"
            signal_strength = "(Multi-Timeframe PSAR Confirmed)"
            
        # SELL: Partial PSAR confirmation
        elif (score < 35 and 
              (trends['sar_short_bearish'] or trends['sar_medium_bearish']) and 
              volume['obv_trend'] == '↓'):
            signal = "SELL"
            signal_strength = "(PSAR Bearish)"
        
        # Add reasons if not already included
        if signal != "HOLD" and signal_reasons:
            signal += f" {signal_strength} [{', '.join(signal_reasons)}]"
        elif signal == "HOLD" and signal_reasons:
            signal = f"HOLD (Conflicting signals: {', '.join(signal_reasons)})"

        # Prepare result dictionary
        result = {
            'Ticker': ticker,
            'Price': f"{latest['price']:.2f}",
            f'Change_{short_period}D': f"{price_changes['short']:.1f}%",
            f'Change_{medium_period}D': f"{price_changes['medium']:.1f}%",
            f'Change_{long_period}D': f"{price_changes['long']:.1f}%",
            f'RSI_{short_period}': f"{momentum['rsi_short']:.1f}",
            f'RSI_{medium_period}': f"{momentum['rsi_medium']:.1f}",
            f'Stoch_%K_{short_period}': f"{latest['short']['Stoch_%K']:.1f}",
            f'MACD_diff_{medium_period}': f"{latest['medium']['MACD_diff']:.3f}",
            'BB_%': f"{momentum['bb_position']:.2%}",
            f'SMA_{short_period}/{medium_period}/{long_period}': f"{latest['short']['SMA']:.1f}/{latest['medium']['SMA']:.1f}/{latest['long']['SMA']:.1f}",
            f'PSAR_{short_period}': 'Bullish' if trends['sar_short_bullish'] else 'Bearish',
            f'PSAR_{medium_period}': 'Bullish' if trends['sar_medium_bullish'] else 'Bearish',
            f'PSAR_{long_period}': 'Bullish' if trends['sar_long_bullish'] else 'Bearish',
            'Volume': f"{df['Volume'].iloc[-1]/1e6:.1f}M",
            'Volume_Spike': 'Yes' if volume['volume_spike'] else 'No',
            'OBV_Trend': volume['obv_trend'],
            'VWAP_Relation': volume['vwap_relation'],
            'Score': f"{score:.1f}",
            'Signal': signal
        }
        
        return result
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

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
        
    def run_backtest(self, start_date, end_date, stock_universe):
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 0:  # Run weekly on Mondays
                self.log_weekly_status(current_date, "PRE-REBALANCE")
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
            current_price = self.get_current_price(ticker, date)
            if current_price is None:
                current_price = position['last_price']
            
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
                'signal': self.get_current_signal(ticker, date)
            })
        
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
            signal_data = self.scorer.calculate_score_and_signals(ticker, date)
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
                closed_positions.append(self.close_position(ticker, current_date))
        
        # Calculate position size
        position_size = self.capital / len(top_stocks) if top_stocks else 0
        
        
        # Open new positions
        for stock in top_stocks:
            ticker = stock['Ticker']
            current_price = float(stock['Price'])
            
            if ticker not in self.portfolio['holdings']:
                if stock['Signal'].startswith(('BUY', 'STRONG BUY')):
                    shares = position_size / current_price
                    opened_positions.append(self.open_position(ticker, shares, current_price, current_date))
            
            # Handle existing positions
            else:
                if stock['Signal'].startswith(('SELL', 'STRONG SELL')):
                    self.close_position(ticker, current_date)

        # Log transaction details
        self.log_transactions(current_date, opened_positions, closed_positions)
        
        # Record portfolio snapshot
        self.record_portfolio(current_date)

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
    
    def open_position(self, ticker, shares, price, date):
        cost = shares * price
        if cost > self.portfolio['cash']:
            shares = self.portfolio['cash'] / price
            cost = shares * price
        
        self.portfolio['holdings'][ticker] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'last_price': price  # Initialize last known price
        }
        self.portfolio['cash'] -= cost

        return {
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'amount': cost
        }
    
    def close_position(self, ticker, date):
        if ticker in self.portfolio['holdings']:
            position = self.portfolio['holdings'][ticker]
            current_price = self.get_current_price(ticker, date)
            
            # Use last known price if current unavailable
            if current_price is None:
                current_price = position['last_price']
            
            proceeds = position['shares'] * current_price
            self.portfolio['cash'] += proceeds
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
        
        with open(self.output_dir / "report_all_stocks.html", "w") as f:
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
    start_date = datetime(2024, 7, 1)
    end_date = datetime(2025, 6, 14)
    stock_universe = [stock for stocks in sector_stocks.values() for stock in stocks]
    # Run backtest
    backtester = StrategyBacktester(capital=1_000_000, top_n=5)
    backtester.run_backtest(start_date, end_date, stock_universe)