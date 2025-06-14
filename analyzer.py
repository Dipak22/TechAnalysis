import yfinance as yf
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from datetime import datetime, timedelta
from models import StockAnalysis

class StockAnalyzer:
    def __init__(self, lookback_days=14):
        self.lookback_days = lookback_days

    def _download_data(self, ticker):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.lookback_days*3)
        stock = yf.Ticker(ticker)
        return stock.history(start=start_date, end=end_date, interval='1d')

    def _calculate_indicators(self, df):
        # Price Momentum
        df['ROC'] = ROCIndicator(close=df['Close'], window=self.lookback_days).roc()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Trend
        macd = MACD(close=df['Close'])
        df['MACD_diff'] = macd.macd_diff()
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Volume
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=self.lookback_days
        ).volume_weighted_average_price()
        
        df['OBV'] = OnBalanceVolumeIndicator(
            close=df['Close'],
            volume=df['Volume']
        ).on_balance_volume()
        
        return df

    def _generate_signal(self, latest, prev, price_change_pct):
        if latest['RSI'] > 70 and price_change_pct > 15:
            return "SELL (Overbought)"
        elif latest['RSI'] < 30 and price_change_pct < -10:
            return "BUY (Oversold)"
        elif (latest['Close'] > latest['SMA_20'] > latest['SMA_50'] and 
              latest['MACD_diff'] > 0 and 
              latest['Volume'] > (prev['Volume'] * 2)):
            return "BUY (Strong Trend)"
        elif (latest['Close'] < latest['SMA_20'] < latest['SMA_50'] and 
              latest['MACD_diff'] < 0):
            return "SELL (Downtrend)"
        return "HOLD"

    def analyze(self, ticker):
        try:
            df = self._download_data(ticker)
            if df.empty or len(df) < self.lookback_days:
                return None

            df = self._calculate_indicators(df)
            latest, prev = df.iloc[-1], df.iloc[-2]
            
            price_change_pct = (latest['Close'] - df['Close'].iloc[-self.lookback_days]) / df['Close'].iloc[-self.lookback_days] * 100
            
            sma_cross = ('Golden Cross' if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50'] 
                        else 'Death Cross' if latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50'] else None)
            
            return StockAnalysis(
                ticker=ticker,
                price=latest['Close'],
                period_change_pct=price_change_pct,
                rsi=latest['RSI'],
                sma_20=latest['SMA_20'],
                sma_50=latest['SMA_50'],
                sma_cross=sma_cross,
                volume=latest['Volume'],
                volume_spike=latest['Volume'] > (df['Volume'].rolling(20).mean().iloc[-1] * 2),
                obv_trend='↑' if latest['OBV'] > prev['OBV'] else '↓',
                signal=self._generate_signal(latest, prev, price_change_pct),
                momentum_score=min(100, max(0, (
                    0.3 * price_change_pct + 
                    0.2 * latest['RSI'] + 
                    20 * (latest['MACD_diff'] > 0) +
                    15 * (latest['Close'] > latest['SMA_20'] > latest['SMA_50']) +
                    15 * (latest['Volume'] > (df['Volume'].rolling(20).mean().iloc[-1] * 2)
                ))))
            )
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            return None