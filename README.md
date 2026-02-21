# Market Regime Analysis

A Python toolkit for detecting and classifying financial market regimes using statistical and machine learning methods.

## What It Does

Identifies which "regime" the market is currently in — e.g. crisis, low-vol rally, goldilocks — and generates PDF reports with charts, statistics, and regime transition probabilities.

## Tools

| Script | Method | Purpose |
|--------|--------|---------|
| `market/hmm_regime_analysis.py` | Hidden Markov Model | Single-asset regime detection via daily returns + volatility |
| `market/market_regime_analysis.py` | K-Means Clustering | Cross-asset regime detection using equity, rates, and Fed liquidity |
| `market/trading_regime.py` | Decision Tree | Regime-based trading signals with backtesting |

## Regimes (Cross-Asset)

1. **Crisis** — high vol, negative momentum
2. **High Vol Rally** — positive momentum, elevated vol
3. **Low Vol Rally** — positive momentum, low vol
4. **Slowdown** — inverted yield curve
5. **Recovery** — transitional
6. **Goldilocks** — low vol, positive yield slope, positive liquidity

## Data Sources

- **Polygon.io** — equities, indices (VIX), crypto, futures
- **FRED** — Treasury yields, Fed balance sheet, RRP, TGA

## Quick Start

```bash
pip install hmmlearn scikit-learn pandas numpy matplotlib seaborn requests pandas-datareader

# HMM analysis on SPY
python market/hmm_regime_analysis.py --api-key YOUR_KEY --ticker SPY --years 5 --states 4

# Cross-asset analysis
python market/market_regime_analysis.py  # configure API keys inside the script
```

## Output

Each run produces a PDF report containing:
- Current regime summary and historical distribution
- Price chart with regime shading
- Volatility and return analysis by regime
- Yield curve and liquidity charts (cross-asset only)
- Regime transition probability matrix

## Project Structure

```
regime-analysis/
├── README.md
└── market/
    ├── hmm_regime_analysis.py
    ├── market_regime_analysis.py
    ├── trading_regime.py
    └── README.md          # detailed documentation
```
