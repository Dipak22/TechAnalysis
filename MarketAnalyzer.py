
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_and_preprocess_vix_data():
    
    """Load and preprocess VIX data from CSV with proper datetime parsing"""
    try:
        # Load VIX data with proper datetime parsing
        vix_data = pd.read_csv(
            'hist_india_vix_-18-06-2024-to-17-06-2025.csv',
            parse_dates=['Date '],
            date_parser=lambda x: pd.to_datetime(x, format='%d-%b-%Y')
              # Specify your actual date format
        )

        vix_data.columns = [
        'date', 'vix_open', 'vix_high', 'vix_low', 'vix_close',
        'vix_prev_close', 'vix_change', 'vix_pct_change']

        
        # Set date as index with proper datetime handling
        vix_data['date'] = pd.to_datetime(vix_data['date'], format='%Y-%m-%d').dt.tz_localize(None)  # Specify format
        vix_data.set_index('date', inplace=True)
        return vix_data
    except Exception as e:
        print(f"Error loading VIX data: {str(e)}")
        vix_data = None
        
class MarketAnalyzer:
    def __init__(self, vix_data):
        self.vix_data = vix_data
        self.sensex = None
        self.nifty_smallcap = None
        self.nifty_midcap = None
        self.sector_data = None
        self.nifty_500 = None
        self._fetch_market_data()
        self._calculate_indicators()
        self._generate_signals()
        
    def _fetch_market_data(self):
        """Fetch all required market data"""
        # Benchmark Indices
        self.sensex = fetch_data("^BSESN")
        self.nifty_smallcap = fetch_data("0P0001PR8B.BO")  # SBI NIFTY Small-cap index 250
        self.nifty_midcap = fetch_data("0P0001NYM3.BO")    # ICICI Pru Nifty Midcap 150 Idx Reg Gr

        # Sectoral Indices
        sectors = {
            "Nifty Bank": "^NSEBANK",
            "Nifty IT": "^CNXIT",
            "Nifty FMCG": "^CNXFMCG",
            "Nifty Auto": "^CNXAUTO",
            "Nifty Pharma": "^CNXPHARMA",
        }
        self.sector_data = {sector: fetch_data(ticker, "3mo") for sector, ticker in sectors.items()}

        # For Advance-Decline Ratio
        self.nifty_500 = fetch_data("^CRSLDX")  # Nifty 500 (proxy)
        
    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        # Moving Averages (12 and 26 days)
        self.sensex['MA_12'] = self.sensex['Close'].rolling(window=12).mean()
        self.sensex['MA_26'] = self.sensex['Close'].rolling(window=26).mean()

        # VIX Thresholds
        self.vix_mean = self.vix_data['vix_close'].mean()
        self.vix_std = self.vix_data['vix_close'].std()
        self.vix_75perc = np.percentile(self.vix_data['vix_close'], 75)

        # RSI (14-day)
        self.sensex['RSI'] = self._compute_rsi(self.sensex['Close'])

        # Advance-Decline Ratio
        self.nifty_500['Advance'] = self.nifty_500['Close'] > self.nifty_500['Open']
        self.nifty_500['Decline'] = self.nifty_500['Close'] < self.nifty_500['Open']
        self.ad_ratio = self.nifty_500['Advance'].sum() / max(self.nifty_500['Decline'].sum(), 1)

        # Sectoral Performance (5-day returns)
        self.sector_returns = pd.DataFrame({
            sector: data['Close'].pct_change(5).iloc[-1] * 100
            for sector, data in self.sector_data.items()
        }, index=["5d Return %"]).T.sort_values("5d Return %", ascending=False)

        # Smallcap/Midcap Performance
        self.smallcap_return = self.nifty_smallcap['Close'].pct_change(5).iloc[-1] * 100
        self.midcap_return = self.nifty_midcap['Close'].pct_change(5).iloc[-1] * 100
        
    def _compute_rsi(self, series, window=14):
        """Calculate RSI for given series"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
        
    def _generate_signals(self):
        """Generate all trading signals"""
        # Trend Signal
        self.trend = "Bullish" if (self.sensex['Close'].iloc[-1] > self.sensex['MA_12'].iloc[-1] and 
                                  self.sensex['Close'].iloc[-1] > self.sensex['MA_26'].iloc[-1]) else "Bearish"

        # VIX Signal
        current_vix = self.vix_data['vix_close'].iloc[-1]
        if current_vix < self.vix_mean:
            self.vix_sentiment = "Low (Bullish)"
        elif current_vix > self.vix_75perc:
            self.vix_sentiment = "Very High (Bearish)"
        else:
            self.vix_sentiment = "Moderate (Neutral)"

        # Sector Signal
        self.top_sector = self.sector_returns.index[0]
        if "Bank" in self.top_sector or "Auto" in self.top_sector or "IT" in self.top_sector:
            self.sector_sentiment = "Risk-On (Cyclicals Leading)"
        elif "FMCG" in self.top_sector or "Pharma" in self.top_sector:
            self.sector_sentiment = "Risk-Off (Defensives Leading)"
        else:
            self.sector_sentiment = "Neutral (Mixed)"

        # Breadth Signal
        if self.ad_ratio > 1.5:
            self.breadth_sentiment = "Broad Participation (Bullish)"
        elif self.ad_ratio < 0.7:
            self.breadth_sentiment = "Weak Participation (Bearish)"
        else:
            self.breadth_sentiment = "Neutral"

        # RSI Signal
        rsi = self.sensex['RSI'].iloc[-1]
        if rsi > 70:
            self.rsi_sentiment = "Overbought (Caution)"
        elif rsi < 30:
            self.rsi_sentiment = "Oversold (Opportunity)"
        else:
            self.rsi_sentiment = "Neutral"

        # Combined Signal
        if self.trend == "Bullish" and self.vix_sentiment == "Low (Bullish)" and "Risk-On" in self.sector_sentiment:
            self.combined_signal = "STRONG BULLISH"
        elif self.trend == "Bullish" and self.vix_sentiment == "Very High (Bearish)":
            self.combined_signal = "CAUTION (Bullish but fragile)"
        elif self.trend == "Bearish" and self.vix_sentiment == "Very High (Bearish)":
            self.combined_signal = "BEARISH"
        else:
            self.combined_signal = "NEUTRAL"
            
    def get_market_analysis(self):
        """Return a comprehensive market analysis dictionary"""
        return {
            "trend": {
                "current_price": self.sensex['Close'].iloc[-1],
                "ma_12": self.sensex['MA_12'].iloc[-1],
                "ma_26": self.sensex['MA_26'].iloc[-1],
                "trend": self.trend,
                "rsi": self.sensex['RSI'].iloc[-1],
                "rsi_sentiment": self.rsi_sentiment
            },
            "volatility": {
                "current_vix": self.vix_data['vix_close'].iloc[-1],
                "vix_mean": self.vix_mean,
                "vix_75th_percentile": self.vix_75perc,
                "vix_sentiment": self.vix_sentiment
            },
            "breadth": {
                "advance_decline_ratio": self.ad_ratio,
                "breadth_sentiment": self.breadth_sentiment,
                "smallcap_return": self.smallcap_return,
                "midcap_return": self.midcap_return
            },
            "sectors": {
                "sector_returns": self.sector_returns,
                "top_sector": self.top_sector,
                "sector_sentiment": self.sector_sentiment,
                "weakest_sector": self.sector_returns.index[-1]
            },
            "combined_signal": self.combined_signal
        }
        
    def get_trading_recommendation(self):
        """Get a trading recommendation based on the analysis"""
        analysis = self.get_market_analysis()
        
        if analysis["combined_signal"] == "STRONG BULLISH":
            return {
                "action": "BUY",
                "confidence": "High",
                "reason": "Broad rally with low volatility and cyclicals leading",
                "focus_sectors": [s for s in self.sector_returns.index if "Bank" in s or "Auto" in s or "IT" in s]
            }
        elif analysis["combined_signal"] == "BEARISH":
            return {
                "action": "SELL or SHORT",
                "confidence": "High",
                "reason": "Downtrend with high volatility and defensive sectors leading",
                "hedge_sectors": [s for s in self.sector_returns.index if "FMCG" in s or "Pharma" in s]
            }
        elif analysis["combined_signal"] == "CAUTION (Bullish but fragile)":
            return {
                "action": "HOLD or Light BUY with tight stops",
                "confidence": "Medium",
                "reason": "Bullish trend but volatility is high indicating potential reversal",
                "focus_sectors": [s for s in self.sector_returns.index if "FMCG" in s or "Pharma" in s]
            }
        else:  # NEUTRAL
            return {
                "action": "HOLD or Wait",
                "confidence": "Low",
                "reason": "Mixed signals in the market",
                "suggested_action": "Wait for clearer trend confirmation"
            }
            
    def plot_analysis(self):
        """Generate analysis plots"""
        plt.figure(figsize=(16, 18))
        gs = GridSpec(4, 2, figure=plt.gcf())

        # Plot 1: SENSEX and MAs
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(self.sensex['Close'], label='SENSEX', color='blue')
        ax1.plot(self.sensex['MA_12'], label='12-Day MA', linestyle='--', color='orange')
        ax1.plot(self.sensex['MA_26'], label='26-Day MA', linestyle='--', color='red')
        ax1.set_title('SENSEX Trend Analysis (12/26 MA)')
        ax1.legend()

        # Plot 2: India VIX Zones
        ax2 = plt.subplot(gs[1, :])
        ax2.plot(self.vix_data['vix_close'], label='India VIX', color='green')
        ax2.axhline(y=self.vix_mean, linestyle='--', color='black', label='Mean')
        ax2.axhline(y=self.vix_75perc, linestyle='--', color='red', label='75th %ile (High VIX)')
        ax2.fill_between(self.vix_data.index, self.vix_mean, self.vix_75perc, color='yellow', alpha=0.1, label='Moderate Volatility')
        ax2.fill_between(self.vix_data.index, self.vix_75perc, self.vix_data['vix_close'].max(), color='red', alpha=0.1, label='High Volatility')
        ax2.set_title('India VIX with Dynamic Zones')
        ax2.legend()

        # Plot 3: Sectoral Heatmap
        ax3 = plt.subplot(gs[2, 0])
        self.sector_returns.plot(kind='bar', ax=ax3, color=['green' if x > 0 else 'red' for x in self.sector_returns['5d Return %']])
        ax3.axhline(y=0, color='black', linestyle='-')
        ax3.set_title('Sectoral Performance (5-Day Returns %)')

        # Plot 4: Smallcap vs Midcap
        ax4 = plt.subplot(gs[2, 1])
        ax4.bar(['Smallcap 250', 'Midcap 150'], [self.smallcap_return, self.midcap_return], 
                color=['green' if x > 0 else 'red' for x in [self.smallcap_return, self.midcap_return]])
        ax4.set_title('Smallcap vs Midcap Returns (5-Day)')

        # Plot 5: RSI
        ax5 = plt.subplot(gs[3, 0])
        ax5.plot(self.sensex['RSI'], label='RSI (14)', color='purple')
        ax5.axhline(y=70, linestyle='--', color='red', label='Overbought')
        ax5.axhline(y=30, linestyle='--', color='green', label='Oversold')
        ax5.set_title('SENSEX RSI')
        ax5.legend()

        # Plot 6: Advance-Decline Ratio
        ax6 = plt.subplot(gs[3, 1])
        ax6.plot(self.nifty_500.index, self.nifty_500['Advance'].rolling(5).sum() / self.nifty_500['Decline'].rolling(5).sum(), 
                 label='A/D Ratio (5d)', color='blue')
        ax6.axhline(y=1.0, linestyle='--', color='black', label='Neutral')
        ax6.set_title('Advance-Decline Ratio (Nifty 500 Proxy)')
        ax6.legend()

        plt.tight_layout()
        return plt

# Helper function (outside class)
def fetch_data(ticker, period="1y"):
    """Fetch data from Yahoo Finance"""
    tck = yf.Ticker(ticker)
    return tck.history(period=period)

# Example usage:
if __name__ == "__main__":
    # First get your VIX data (example)
    vix_data = load_and_preprocess_vix_data()
    
    # Create analyzer instance
    analyzer = MarketAnalyzer(vix_data)
    
    # Get market analysis
    analysis = analyzer.get_market_analysis()
    print("Market Analysis:")
    print(analysis)
    
    # Get trading recommendation
    recommendation = analyzer.get_trading_recommendation()
    print("\nTrading Recommendation:")
    print(recommendation)
    
    # Show plots
    plt = analyzer.plot_analysis()
    plt.show()