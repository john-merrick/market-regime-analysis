import os
import math
import requests
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 0) Config
# -----------------------------
API_KEY = POLYGON_KEY = "116xopGPdlPmt4vYkX9BWphPgXtO8wy9"
BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com")
FUT_VER  = os.environ.get("FUTURES_API_VERSION", "v1")  # docs show vX placeholder

if not API_KEY:
    raise RuntimeError("Set MASSIVE_API_KEY in your environment.")

HEADERS = {
    # Massive SDK examples use Bearer auth; keep this consistent.
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

# ES contract multiplier (E-mini S&P 500): $50 per index point
ES_MULTIPLIER = 50.0


# -----------------------------
# 1) Data fetch (Futures Aggregates)
# -----------------------------
def fetch_futures_aggs(
    ticker: str,
    start_date: str,
    end_date: str,
    resolution: str = "1min",
    limit: int = 50000,
    base_url: str = BASE_URL,
    fut_ver: str = FUT_VER,
) -> pd.DataFrame:
    """
    Fetch futures aggregate bars from:
      GET /futures/vX/aggs/{ticker}

    Docs show query params like:
      resolution=1min
      window_start.gte=YYYY-MM-DD
      window_start.lte=YYYY-MM-DD
      limit=...
    """
    url = f"{base_url}/futures/{fut_ver}/aggs/{ticker}"
    params = {
        "resolution": resolution,
        "window_start.gte": start_date,
        "window_start.lte": end_date,
        "limit": limit,
        "sort": "window_start.asc",
    }

    rows = []
    while True:
        r = requests.get(url, headers=HEADERS, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", []) or []
        rows.extend(results)

        next_url = data.get("next_url")
        if not next_url:
            break

        # next_url is already a fully qualified URL; subsequent calls typically don't need params
        url = next_url
        params = None

    if not rows:
        raise RuntimeError(f"No data returned for {ticker} in {start_date}..{end_date}")

    df = pd.DataFrame(rows)

    # window_start is documented as an integer timestamp (often nanoseconds in examples).
    # We'll interpret as ns since epoch.
    df["dt"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    df = df.set_index("dt").sort_index()

    # Standardize column names
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})

    # Optional: keep session_end_date if provided (can help grouping)
    # df["session_end_date"] may exist depending on plan/endpoint response
    return df[["open", "high", "low", "close", "volume"] + (["session_end_date"] if "session_end_date" in df.columns else [])]


# -----------------------------
# 2) Indicators (RSI, ADX, VWAP, etc.)
# -----------------------------
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx


def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    # Typical price is common in VWAP computations
    tp = (high + low + close) / 3.0
    pv = tp * volume
    vwap = pv.rolling(window).sum() / volume.rolling(window).sum().replace(0.0, np.nan)
    return vwap


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_15"] = out["close"].pct_change(15)

    out["rsi_14"] = rsi_wilder(out["close"], 14)
    out["adx_14"] = adx_wilder(out["high"], out["low"], out["close"], 14)

    out["sma_14"] = out["close"].rolling(14).mean()
    out["sma_ratio"] = out["sma_14"] / out["close"]

    # Rolling correlation between SMA and close (window=14)
    out["sma_corr"] = out["sma_14"].rolling(14).corr(out["close"])

    out["vol_14"] = out["ret_1"].rolling(14).std()

    # "210-period volatility" described as volatility of 15-period rolling returns:
    out["vol_210"] = out["ret_15"].rolling(14).std()

    out["vwap_14"] = rolling_vwap(out["high"], out["low"], out["close"], out["volume"], 14)
    out["vwap_ratio"] = out["vwap_14"] / out["close"]

    # Label: next-bar return sign (buy=1, sell=0)
    out["y"] = (out["ret_1"].shift(-1) > 0).astype(int)

    feature_cols = [
        "ret_1", "ret_15", "rsi_14", "adx_14",
        "sma_ratio", "sma_corr", "vol_14", "vol_210", "vwap_ratio"
    ]

    # Keep only rows with full feature set + label
    out = out.dropna(subset=feature_cols + ["y"]).copy()
    return out


# -----------------------------
# 3) Train/test, fit tree, export rules
# -----------------------------
def train_test_split_time(df_feat: pd.DataFrame, train_frac: float = 0.7):
    n = len(df_feat)
    split = int(math.floor(n * train_frac))
    train = df_feat.iloc[:split]
    test  = df_feat.iloc[split:]
    return train, test


def fit_tree(train_df: pd.DataFrame, feature_cols: list[str]) -> DecisionTreeClassifier:
    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def export_tree_rules(clf: DecisionTreeClassifier, feature_cols: list[str]) -> str:
    return export_text(clf, feature_names=feature_cols, decimals=6)


# -----------------------------
# 4) Backtest (long/flat; signal shift by 1 bar)
# -----------------------------
def backtest_long_flat(test_df: pd.DataFrame, y_pred: np.ndarray, multiplier: float = ES_MULTIPLIER) -> dict:
    """
    Interpreting model output:
      1 = buy (long)
      0 = sell (flat)

    Delay:
      shift the position forward by 1 bar (signal at t acts at t+1)
    """
    bt = test_df.copy()
    bt["pred"] = y_pred.astype(int)
    bt["pos"] = bt["pred"].shift(1).fillna(0).astype(int)

    bt["bh_ret"] = bt["close"].pct_change().fillna(0.0)
    bt["strat_ret"] = (bt["pos"] * bt["bh_ret"]).fillna(0.0)

    bt["bh_equity"] = (1.0 + bt["bh_ret"]).cumprod()
    bt["strat_equity"] = (1.0 + bt["strat_ret"]).cumprod()

    # Futures PnL in dollars (1 contract)
    bt["pnl_$"] = bt["pos"] * bt["close"].diff().fillna(0.0) * multiplier
    bt["cum_pnl_$"] = bt["pnl_$"].cumsum()

    # Simple Sharpe annualization based on observed bars/day
    if "session_end_date" in bt.columns:
        bars_per_day = bt.groupby("session_end_date").size().median()
    else:
        bars_per_day = bt.groupby(bt.index.date).size().median()

    periods_per_year = float(bars_per_day) * 252.0 if bars_per_day and not np.isnan(bars_per_day) else 252.0 * 390.0

    mu = bt["strat_ret"].mean()
    sd = bt["strat_ret"].std(ddof=0)
    sharpe = (np.sqrt(periods_per_year) * mu / sd) if sd > 0 else np.nan

    return {
        "bars_per_day_median": bars_per_day,
        "periods_per_year": periods_per_year,
        "sharpe": sharpe,
        "total_return_strategy": bt["strat_equity"].iloc[-1] - 1.0,
        "total_return_buy_hold": bt["bh_equity"].iloc[-1] - 1.0,
        "df": bt,
    }


# -----------------------------
# 5) Run
# -----------------------------
if __name__ == "__main__":
    # Example: pick a specific ES contract ticker (you choose)
    TICKER = "SPY"

    # Keep the date window within that contract’s liquid life
    START = "2024-10-01"
    END   = "2024-12-10"

    raw = fetch_futures_aggs(TICKER, START, END, resolution="1min")
    feat = build_features(raw)

    feature_cols = [
        "ret_1", "ret_15", "rsi_14", "adx_14",
        "sma_ratio", "sma_corr", "vol_14", "vol_210", "vwap_ratio"
    ]

    train_df, test_df = train_test_split_time(feat, train_frac=0.7)
    clf = fit_tree(train_df, feature_cols)

    # Evaluate classification accuracy (not the final goal, but useful sanity check)
    y_pred = clf.predict(test_df[feature_cols].values)
    print("Accuracy:", accuracy_score(test_df["y"].values, y_pred))
    print(classification_report(test_df["y"].values, y_pred))

    # Export the “intuitive decision tree” rules
    rules = export_tree_rules(clf, feature_cols)
    print("\n--- Decision Tree Rules ---\n")
    print(rules)

    # Backtest
    bt = backtest_long_flat(test_df, y_pred)
    print("\n--- Backtest Summary ---")
    for k, v in bt.items():
        if k != "df":
            print(f"{k}: {v}")

    # bt["df"] contains the full equity curves / positions / PnL series
