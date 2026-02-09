# Market Regime Analysis

A Python toolkit for detecting and analyzing market regimes using statistical and machine learning techniques. The project provides multiple approaches to regime detection including Hidden Markov Models (HMM) and K-Means clustering, with comprehensive PDF report generation.

## Overview

This project contains three main analysis tools:

| Script | Method | Data Sources | Output |
|--------|--------|--------------|--------|
| `hmm_regime_analysis.py` | Gaussian HMM | Polygon (any ticker) | PDF report + CLI |
| `market_regime_analysis.py` | K-Means Clustering | Polygon + FRED | PDF report |
| `trading_regime.py` | Decision Tree | Polygon (futures) | Trading rules + backtest |

## Features

### HMM Regime Analysis (`hmm_regime_analysis.py`)

Single-asset regime detection using Hidden Markov Models:

- **Data**: Daily OHLCV from Polygon API for any supported ticker
- **Features**: Log returns + rolling realized volatility (annualized)
- **Model**: Gaussian HMM with configurable states (3-6 recommended)
- **Labels**: Auto-labeled regimes (Low/Mid/High Vol x Risk-On/Off)
- **Forecast**: Next regime prediction using transition matrix probabilities

**CLI Usage:**
```bash
python hmm_regime_analysis.py --api-key YOUR_KEY --ticker SPY --years 5 --states 4 --out spy_hmm.pdf
python hmm_regime_analysis.py --api-key YOUR_KEY --ticker AAPL --start 2018-01-01 --end 2026-01-01 --states 5
python hmm_regime_analysis.py --api-key YOUR_KEY --ticker QQQ --years 10 --states 6 --vol-window 30
```

**CLI Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--ticker` | Ticker symbol (SPY, AAPL, QQQ, I:VIX, C:BTCUSD) | Required |
| `--years` | Years of history (if no --start) | 5 |
| `--start/--end` | Date range (YYYY-MM-DD) | Auto |
| `--states` | Number of HMM states | 4 |
| `--vol-window` | Rolling volatility window | 20 |
| `--out` | Output PDF path | Auto-generated |

### Cross-Asset Regime Analysis (`market_regime_analysis.py`)

Multi-factor regime detection using macroeconomic indicators:

- **Data Sources**:
  - S&P 500 price (SPY proxy) - Polygon
  - VIX implied volatility (I:VIX) - Polygon
  - Treasury yields (3M, 2Y, 10Y, 30Y) - FRED
  - Fed liquidity (Total Assets, RRP, TGA) - FRED

- **Features**:
  - Returns and momentum
  - VIX-implied volatility
  - Yield curve slope, curvature, and level
  - Net liquidity and liquidity change

- **Six Defined Regimes**:
  1. **Crisis** - High volatility + negative momentum
  2. **High Volatility Rally** - Positive momentum with elevated vol
  3. **Low Volatility Rally** - Positive momentum with low vol
  4. **Slowdown** - Inverted yield curve
  5. **Recovery** - Transitional regime
  6. **Goldilocks** - Low vol + positive slope + positive liquidity

**Usage:**
```python
from market_regime_analysis import RegimeAnalyzer

analyzer = RegimeAnalyzer(
    polygon_api_key="YOUR_POLYGON_KEY",
    fred_api_key="YOUR_FRED_KEY"
)

results = analyzer.run_analysis(
    start_date="2022-01-01",
    end_date="2025-01-01",
    n_regimes=6
)

analyzer.print_regime_summary()
analyzer.generate_regime_report("regime_report.pdf")
print(f"Current regime: {analyzer.get_current_regime()}")
```

### Trading Regime Strategy (`trading_regime.py`)

Decision tree-based trading signal generation:

- **Indicators**: RSI, ADX, SMA, VWAP, rolling volatility
- **Model**: Decision tree classifier (max depth 4)
- **Strategy**: Long/flat based on next-bar return prediction
- **Backtest**: Sharpe ratio, equity curves, PnL tracking

## Installation

```bash
pip install hmmlearn scikit-learn pandas numpy matplotlib requests seaborn
```

Optional for FRED fallback:
```bash
pip install pandas-datareader
```

## API Keys Required

- **Polygon.io** - Market data (stocks, indices, crypto, futures)
- **FRED** - Federal Reserve economic data (optional, for cross-asset analysis)

## PDF Report Contents

Reports include:

1. **Summary Page** - Current regime, distribution, key statistics
2. **Price Overlay** - Price chart with regime shading
3. **Volatility Analysis** - Time series and distribution by regime
4. **Return Statistics** - Cumulative returns and distribution
5. **Yield Curve Charts** - Slope, levels, and regime comparison (cross-asset only)
6. **Liquidity Analysis** - Net liquidity and changes (cross-asset only)
7. **Transition Matrix** - Regime transition probabilities heatmap

## Project Structure

```
market/
├── hmm_regime_analysis.py       # HMM-based single-asset analysis
├── market_regime_analysis.py    # K-Means cross-asset analysis
├── trading_regime.py            # Decision tree trading strategy
├── hmm_regime_SPY_*.pdf         # Example SPY report
├── hmm_regime_AAPL_*.pdf        # Example AAPL report
├── regime_analysis_report.pdf   # Example cross-asset report
└── README.md
```

## Methodology

### Hidden Markov Model Approach

1. Compute daily log returns and rolling realized volatility
2. Standardize features (z-score normalization)
3. Fit Gaussian HMM with full covariance
4. Extract hidden state sequence via Viterbi algorithm
5. Label states by volatility tercile and return sign
6. Forecast using row of transition matrix for current state

### K-Means Clustering Approach

1. Aggregate features from multiple data sources (equity, vol, rates, liquidity)
2. Standardize feature matrix
3. Apply K-Means with 6 clusters
4. Deterministically label clusters based on feature characteristics
5. Compute empirical transition probabilities

## License

MIT
