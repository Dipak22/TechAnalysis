from dataclasses import dataclass
from datetime import datetime

@dataclass
class StockAnalysis:
    ticker: str
    price: float
    period_change_pct: float
    rsi: float
    sma_20: float
    sma_50: float
    sma_cross: str
    volume: float
    volume_spike: bool
    obv_trend: str
    signal: str
    momentum_score: float
    analysis_date: datetime = datetime.now()