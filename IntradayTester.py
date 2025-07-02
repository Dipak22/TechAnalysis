import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator,MACD, macd_diff, macd_signal
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class IntradayMomentumStrategy(Strategy):
    # Strategy parameters
    short_period = 5
    medium_period = 10
    long_period = 20
    position_size = 0.1  # 10% of capital per trade
    stop_loss_pct = 0.01  # 1% stop loss
    take_profit_pct = 0.02  # 2% take profit
    
    def init(self):
        super().init()
        self.trade_log = []  # To store trade details
        self.current_trade = None
        
    def next(self):
        current_time = self.data.index[-1]
        
        # Only trade during market hours (9:15 AM to 3:30 PM IST)
        start_time = datetime.strptime('09:15:00', '%H:%M:%S').time()
        end_time = datetime.strptime('15:30:00', '%H:%M:%S').time()
        
        if current_time.time() < start_time or current_time.time() > end_time:
            return
            
        df_live = pd.DataFrame({
            'Open': self.data.Open[:],
            'High': self.data.High[:],
            'Low': self.data.Low[:],
            'Close': self.data.Close[:],
            'Volume': self.data.Volume[:]
        }, index=self.data.index)

        signal_data = calculate_signals(
            ticker="NONE",
            current_date=current_time,
            df=df_live
        )
        
        if not signal_data:
            return
            
        # Execute trades based on signals
        if signal_data['Signal_Value'] >= 4 and not self.position:  # Buy signals
            # Calculate number of shares (10% of equity / current price)
            price = self.data.Close[-1]
            equity = self.equity
            shares = int((equity * self.position_size) / price)
            
            if shares > 0:
                self.buy(size=shares, 
                        sl=price * (1 - self.stop_loss_pct),
                        tp=price * (1 + self.take_profit_pct))
                
                # Log trade entry
                self.current_trade = {
                    'entry_time': current_time,
                    'entry_price': price,
                    'shares': shares,
                    'signal': signal_data['Signal'],
                    'stop_loss': price * (1 - self.stop_loss_pct),
                    'take_profit': price * (1 + self.take_profit_pct)
                }
                
        elif signal_data['Signal_Value'] <= 3 and not self.position:  # Sell signals
            price = self.data.Close[-1]
            equity = self.equity
            shares = int((equity * self.position_size) / price)
            
            if shares > 0:
                self.sell(size=shares,
                         sl=price * (1 + self.stop_loss_pct),
                         tp=price * (1 - self.take_profit_pct))
                
                # Log trade entry
                self.current_trade = {
                    'entry_time': current_time,
                    'entry_price': price,
                    'shares': shares,
                    'signal': signal_data['Signal'],
                    'stop_loss': price * (1 + self.stop_loss_pct),
                    'take_profit': price * (1 - self.take_profit_pct)
                }
    
    # Called when a trade is closed
    def notify_trade(self, trade):
        if trade.is_long:
            direction = "BUY"
        else:
            direction = "SELL"
            
        if self.current_trade:
            self.current_trade.update({
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'pnl': trade.pl,
                'pnl_pct': trade.pl_pct,
                'duration': trade.exit_time - trade.entry_time
            })
            self.trade_log.append(self.current_trade)
            self.current_trade = None

def calculate_signals(ticker, current_date, short_period=5, medium_period=10, long_period=20, df=None):
    """Modified version of your signal calculator that works with backtesting data"""
    try:
        if df is None:
            return None
            
        # Use the last long_period*2 data points for calculation
        if len(df) < long_period * 2:
            return None
            
        # Get the most recent data
        latest_data = df.iloc[-long_period*2:]
        
        # Initialize indicators dictionary
        indicators = {}

        # Short-term indicators
        indicators['short'] = {
            'RSI': RSIIndicator(close=latest_data['Close'], window=short_period).rsi(),
            'ROC': ROCIndicator(close=latest_data['Close'], window=short_period).roc(),
            'Stoch_%K': StochasticOscillator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                window=short_period,
                smooth_window=3
            ).stoch(),
            'SMA': SMAIndicator(close=latest_data['Close'], window=short_period).sma_indicator(),
            'EMA': EMAIndicator(close=latest_data['Close'], window=short_period).ema_indicator(),
            'ATR': AverageTrueRange(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                window=short_period
            ).average_true_range(),
            'PSAR': PSARIndicator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                step=0.02,
                max_step=0.2
            ).psar(),
            'ADX': ADXIndicator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                window=short_period
            ).adx()
        }

        # Medium-term indicators
        adx_indicator = ADXIndicator(
            high=latest_data['High'],
            low=latest_data['Low'],
            close=latest_data['Close'],
            window=medium_period
        )
        macd_indicator = MACD(
            close=latest_data['Close'],
            window_slow=20,
            window_fast=6,
            window_sign=10
        )
        indicators['medium'] = {
            'RSI': RSIIndicator(close=latest_data['Close'], window=medium_period).rsi(),
            'ROC': ROCIndicator(close=latest_data['Close'], window=medium_period).roc(),
            'SMA': SMAIndicator(close=latest_data['Close'], window=medium_period).sma_indicator(),
            'EMA': EMAIndicator(close=latest_data['Close'], window=medium_period).ema_indicator(),
            'MACD': macd_indicator.macd(),
            'MACD_hist': macd_indicator.macd_diff(),
            'MACD_signal': macd_indicator.macd_signal(),
            'ADX': adx_indicator.adx(),
            'DMP': adx_indicator.adx_pos(),
            'DMN': adx_indicator.adx_neg(),
            'PSAR': PSARIndicator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                step=0.015,
                max_step=0.15
            ).psar()
        }
        
        # Long-term indicators
        indicators['long'] = {
            'SMA': SMAIndicator(close=latest_data['Close'], window=long_period).sma_indicator(),
            'EMA': EMAIndicator(close=latest_data['Close'], window=long_period).ema_indicator(),
            'BB': BollingerBands(close=latest_data['Close'], window=long_period, window_dev=2),
            'VWAP': VolumeWeightedAveragePrice(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                volume=latest_data['Volume'],
                window=long_period
            ).volume_weighted_average_price(),
            'PSAR': PSARIndicator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                step=0.01,
                max_step=0.1
            ).psar(),
            'ADX': ADXIndicator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                window=long_period
            ).adx()
        }

        # Volume indicators
        volume_indicators = {
            'OBV': OnBalanceVolumeIndicator(
                close=latest_data['Close'],
                volume=latest_data['Volume']
            ).on_balance_volume(),
            'ADI': AccDistIndexIndicator(
                high=latest_data['High'],
                low=latest_data['Low'],
                close=latest_data['Close'],
                volume=latest_data['Volume']
            ).acc_dist_index(),
            'Volume_MA': latest_data['Volume'].rolling(20).mean(),
            'Volume_Spike': latest_data['Volume'] > (latest_data['Volume'].rolling(20).mean() * 2)
        }
        
        # Get latest values
        latest = {
            'price': latest_data['Close'].iloc[-1],
            'open': latest_data['Open'].iloc[-1],
            'high': latest_data['High'].iloc[-1],
            'low': latest_data['Low'].iloc[-1],
            'close': latest_data['Close'].iloc[-1],
            'prev_close': latest_data['Close'].iloc[-2],
            'prev_open': latest_data['Open'].iloc[-2],
            'prev_high': latest_data['High'].iloc[-2],
            'prev_low': latest_data['Low'].iloc[-2],
            'short': {k: v.iloc[-1] for k, v in indicators['short'].items()},
            'medium': {k: v.iloc[-1] if not isinstance(v, dict) else {sk: sv.iloc[-1] for sk, sv in v.items()} 
                      for k, v in indicators['medium'].items()},
            'long': {k: v.iloc[-1] if not hasattr(v, 'bollinger_hband') else {
                'upper': v.bollinger_hband().iloc[-1],
                'middle': v.bollinger_mavg().iloc[-1],
                'lower': v.bollinger_lband().iloc[-1],
                'percent': np.divide(
                    (latest_data['Close'].iloc[-1] - v.bollinger_lband().iloc[-1]),
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
            'short': (latest['price'] - latest_data['Close'].iloc[-short_period]) / latest_data['Close'].iloc[-short_period] * 100,
            'medium': (latest['price'] - latest_data['Close'].iloc[-medium_period]) / latest_data['Close'].iloc[-medium_period] * 100,
            'long': (latest['price'] - latest_data['Close'].iloc[-long_period]) / latest_data['Close'].iloc[-long_period] * 100
        }

        # Trend analysis with PSAR and candlestick patterns (same as before)
        trends = {
            'short_term_up': latest['price'] > latest['short']['SMA'] > latest['short']['EMA'],
            'medium_term_up': latest['price'] > latest['medium']['SMA'] > latest['medium']['EMA'],
            'long_term_up': latest['price'] > latest['long']['SMA'] > latest['long']['EMA'],
            'golden_cross': latest['short']['SMA'] > latest['medium']['SMA'] and prev['short']['SMA'] <= prev['medium']['SMA'],
            'death_cross': latest['short']['SMA'] < latest['medium']['SMA'] and prev['short']['SMA'] >= prev['medium']['SMA'],
            'macd_bullish': latest['medium']['MACD_hist'] > prev['medium']['MACD_hist'] and 
                           latest['medium']['MACD'] > latest['medium']['MACD_signal'] and 
                           latest['medium']['MACD'] > prev['medium']['MACD'] and 
                           latest['medium']['MACD'] > 0,
            'macd_bearish': latest['medium']['MACD_hist'] < prev['medium']['MACD_hist'] and 
                           latest['medium']['MACD'] < latest['medium']['MACD_signal'] and 
                           latest['medium']['MACD'] < prev['medium']['MACD'] and 
                           latest['medium']['MACD'] < 0,
            'adx_strength': latest['medium']['ADX'] > 25,
            'dmp_dominant': latest['medium']['DMP'] > latest['medium']['DMN'],
            'dmn_dominant': latest['medium']['DMN'] > latest['medium']['DMP'],
            'adx_bullish': latest['short']['ADX'] > latest['medium']['ADX'] and latest['medium']['ADX'] > latest['long']['ADX'],
            'adx_bearish': latest['short']['ADX'] < latest['medium']['ADX'] and latest['medium']['ADX'] < latest['long']['ADX'],
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
        
        # Generate composite score (0-100)
        score = 50  # Neutral starting point

        # Trend Strength (25% weight)
        trend_score = 0
        trend_score += 3 if trends['short_term_up'] else -3
        trend_score += 5 if trends['medium_term_up'] else -5
        trend_score += 7 if trends['long_term_up'] else -7
        trend_score += 10 if trends['golden_cross'] else (-10 if trends['death_cross'] else 0)
        trend_score += 7 if trends['macd_bullish'] else (-7 if trends['macd_bearish'] else 0)
        adx_component = min(50, latest['medium']['ADX']) / 2
        trend_score += adx_component if trends['adx_strength'] else 0
        trend_score += 5 if trends['dmp_dominant'] else (-5 if trends['dmn_dominant'] else 0)
        score += (trend_score / 67) * 30

        # PSAR Confirmation (20% weight)
        psar_score = 0
        psar_score += 5 if trends['sar_short_bullish'] else -5
        psar_score += 7 if trends['sar_medium_bullish'] else -7
        psar_score += 8 if trends['sar_long_bullish'] else -8
        score += (psar_score / 20) * 20

        # Momentum Strength (20% weight)
        momentum_score = 0
        if momentum['rsi_short'] > 70:
            momentum_score -= ((momentum['rsi_short'] - 70) / 30) ** 2 * 10
        elif momentum['rsi_short'] < 30:
            momentum_score += ((30 - momentum['rsi_short']) / 30) ** 2 * 10
        if momentum['stoch_overbought']:
            momentum_score -= ((latest['short']['Stoch_%K'] - 80) / 20) ** 2 * 5
        elif momentum['stoch_oversold']:
            momentum_score += ((20 - latest['short']['Stoch_%K']) / 20) ** 2 * 5
        momentum_score += (momentum['roc_short'] / 10) * 3
        momentum_score += (momentum['roc_medium'] / 7) * 2
        momentum_score += (momentum['bb_position'] - 0.5) * 10
        score += (momentum_score / 30) * 20

        # Volume Confirmation (15% weight)
        volume_score = 0
        volume_score += 10 if volume['obv_trend'] == '↑' else -10
        if volume['volume_spike']:
            volume_score += 5 if volume['obv_trend'] == '↑' else -5
        volume_score += 3 if volume['adi_trend'] == '↑' else -3
        volume_score += 2 if volume['vwap_relation'] == 'above' else -2
        score += (volume_score / 20) * 15

        # Final score adjustment
        score = max(0, min(100, score))

        # Generate trading signal
        signal = "HOLD"
        signal_strength = ""
        signal_value = 0
        
        if (trends['adx_bullish'] and
            trends['sar_short_bullish'] and 
            trends['sar_medium_bullish'] and 
            trends['sar_long_bullish'] and 
            volume['obv_trend'] == '↑' and
            trends['dmp_dominant'] and
            trends['macd_bullish']):
            signal = "STRONG BUY"
            signal_value = 6
        elif (trends['adx_bullish'] and
              (trends['sar_short_bullish'] or trends['sar_medium_bullish']) and 
              volume['obv_trend'] == '↑' and
              trends['dmp_dominant'] and
              trends['macd_bullish']):
            signal = "BUY"
            signal_value = 5
        elif (trends['adx_bearish'] and
              trends['sar_short_bearish'] and 
              trends['sar_medium_bearish'] and 
              trends['sar_long_bearish'] and 
              volume['obv_trend'] == '↓' and
              trends['dmn_dominant'] and
              trends['macd_bearish']):
            signal = "STRONG SELL"
            signal_value = 1
        elif (trends['adx_bearish'] and
              (trends['sar_short_bearish'] or trends['sar_medium_bearish']) and 
              volume['obv_trend'] == '↓' and
              trends['dmn_dominant'] and
              trends['macd_bearish']):
            signal = "SELL"
            signal_value = 2
        elif (trends['macd_bullish'] and
              (trends['sar_short_bullish'] or trends['sar_medium_bullish'])):
            signal = "WEAK BUY"
            signal_value = 4
        elif (trends['macd_bearish'] and
              (trends['sar_short_bearish'] or trends['sar_medium_bearish'])):
            signal = "WEAK SELL"
            signal_value = 3

        # Prepare result dictionary
        result = {
            'Price': f"{latest['price']:.2f}",
            'RSI_Short': f"{momentum['rsi_short']:.1f}",
            'MACD': 'Bullish' if trends['macd_bullish'] else ('Bearish' if trends['macd_bearish'] else 'Neutral'),
            'BB_%': f"{momentum['bb_position']:.2%}",
            'PSAR_Short': 'Bullish' if trends['sar_short_bullish'] else 'Bearish',
            'Volume_Spike': 'Yes' if volume['volume_spike'] else 'No',
            'Score': f"{score:.1f}",
            'Signal': signal,
            'Signal_Value': signal_value
        }
        
        return result
    except Exception as e:
        print(f"Error calculating signals: {str(e)}")
        return None

def print_trade_details(trade_log):
    print("\nTrade Details:")
    print("-" * 120)
    print(f"{'#':<3} | {'Direction':<8} | {'Entry Time':<20} | {'Entry Price':<12} | {'Shares':<8} | "
          f"{'Exit Time':<20} | {'Exit Price':<12} | {'PNL':<10} | {'PNL %':<8} | {'Duration':<15} | {'Signal'}")
    print("-" * 120)
    
    for i, trade in enumerate(trade_log, 1):
        direction = "BUY" if trade['entry_price'] < trade.get('exit_price', trade['entry_price']) else "SELL"
        
        print(f"{i:<3} | {direction:<8} | "
              f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'):<20} | "
              f"{trade['entry_price']:>12.2f} | "
              f"{trade['shares']:>8} | "
              f"{trade.get('exit_time', 'Open').strftime('%Y-%m-%d %H:%M:%S') if 'exit_time' in trade else 'Open':<20} | "
              f"{trade.get('exit_price', ''):>12.2f} | "
              f"{trade.get('pnl', 0):>10.2f} | "
              f"{trade.get('pnl_pct', 0):>7.2f}% | "
              f"{str(trade.get('duration', '')).split('.')[0] if 'duration' in trade else '':<15} | "
              f"{trade['signal']}")

def run_backtest(ticker, start_date, end_date, interval='5m'):
    # Download data
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}")
        return None

    
    # Run backtest
    bt = Backtest(data, IntradayMomentumStrategy, 
                  commission=.002, 
                  margin=1.0,
                  cash=100000,
                  trade_on_close=True)
    
    stats = bt.run()
    
    # Get trade details from the strategy instance
    trade_log = stats._strategy.trade_log
    
    # Print trade details
    print_trade_details(trade_log)
    
    # Print performance summary with updated keys
    print("\nPerformance Summary:")
    #print(f"Initial Capital: ₹{stats['_equity_initial']:,.2f}")
    #print(f"Final Capital: ₹{stats['_equity_final']:,.2f}")
    print(f"Total Return: {stats['Return [%]']:.2f}%")
    print(f"Total Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    #print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    #print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    
    return stats


# Example usage
if __name__ == "__main__":
    # Test parameters
    ticker = "RELIANCE.NS"  # NSE stock
    start_date = "2025-06-25"
    end_date = "2025-07-01"
    interval = "5m"  # 5-minute data
    
    # Run backtest
    results = run_backtest(ticker, start_date, end_date, interval)
    
    