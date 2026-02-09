#!/usr/bin/env python
"""
HMM Regime Analysis (Polygon) — Any Ticker, CLI, PDF Output
----------------------------------------------------------

What it does
- Downloads OHLCV data for any Polygon-supported ticker via /v2/aggs
- Builds features:
    * daily log returns
    * rolling realized volatility (annualized)
- Fits Gaussian HMM (hmmlearn) to [returns, volatility]
- Labels regimes automatically from state means (Low/Mid/High vol × Risk-On/Off)
- Forecasts next regime using the learned transition matrix
- Generates a PDF report (charts + tables + transition matrix heatmap)

Install
    pip install hmmlearn scikit-learn pandas numpy matplotlib requests seaborn

Examples
    python hmm_regime_cli.py --ticker SPY --years 5 --states 4 --vol-window 20 --out spy_hmm.pdf
    python hmm_regime_cli.py --ticker QQQ --start 2018-01-01 --end 2026-01-01 --states 5 --out qqq_hmm.pdf
    python hmm_regime_cli.py --ticker AAPL --years 10 --states 6 --vol-window 30 --out aapl_hmm.pdf

Notes
- Uses realized vol from returns (not VIX). If you want implied vol from VIX where applicable,
  tell me and I’ll add a --vix option that uses I:VIX and aligns series.
"""

import argparse
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import seaborn as sns


# -------------------------
# Data
# -------------------------
def fetch_polygon_ohlcv(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    timespan: str = "day",
    multiplier: int = 1,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Polygon aggregates endpoint.
    ticker examples: SPY, AAPL, QQQ, I:VIX (index), C:BTCUSD (crypto), etc.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"Polygon API error: {r.status_code} {r.text}")

    js = r.json()
    if "results" not in js or not js["results"]:
        raise RuntimeError("No data returned from Polygon (check ticker/date range).")

    df = pd.DataFrame(js["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()

    return df


def compute_features(price: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Features:
    - ret: daily log return
    - vol: rolling realized vol (annualized)
    """
    feat = pd.DataFrame(index=price.index)
    feat["ret"] = np.log(price["close"] / price["close"].shift(1))
    feat["vol"] = feat["ret"].rolling(vol_window).std() * np.sqrt(252)
    feat = feat.dropna()
    return feat


# -------------------------
# HMM
# -------------------------
def fit_hmm(features: pd.DataFrame, n_states: int, n_iter: int = 500, random_state: int = 42) -> Tuple[pd.Series, GaussianHMM, StandardScaler]:
    """
    Fit Gaussian HMM on standardized [ret, vol] features.
    Returns:
        states: pd.Series of inferred hidden state id per day
        model: fitted HMM
        scaler: fitted scaler
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features[["ret", "vol"]].values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    states = pd.Series(model.predict(X), index=features.index, name="state")
    return states, model, scaler


def state_means(features: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    rows = []
    for s in sorted(states.unique()):
        m = features.loc[states == s, ["ret", "vol"]].mean()
        m.name = s
        rows.append(m)
    return pd.DataFrame(rows)


def label_states(means: pd.DataFrame) -> Dict[int, str]:
    """
    Label states using:
    - Vol bucket by rank (Low / Mid / High, using terciles)
    - Risk tag by mean return sign (Risk-On if >= 0 else Risk-Off)
    """
    df = means.copy()
    df["vol_rank"] = df["vol"].rank(method="first")  # 1..N
    n = len(df)

    lo_cut = int(math.ceil(n / 3))
    hi_cut = int(math.ceil(2 * n / 3))

    out = {}
    for st, row in df.iterrows():
        vr = int(row["vol_rank"])
        if vr <= lo_cut:
            vol_tag = "Low Vol"
        elif vr <= hi_cut:
            vol_tag = "Mid Vol"
        else:
            vol_tag = "High Vol"

        risk_tag = "Risk-On" if row["ret"] >= 0 else "Risk-Off"
        out[int(st)] = f"{vol_tag} {risk_tag}"
    return out


def forecast_next_regime(states: pd.Series, model: GaussianHMM, label_map: Dict[int, str]) -> Dict[str, object]:
    """
    Forecast next state/regime using transition matrix row for the current state.
    """
    current_state = int(states.iloc[-1])
    trans = model.transmat_[current_state, :]
    next_state = int(np.argmax(trans))

    next_state_probs = {int(i): float(p) for i, p in enumerate(trans)}
    next_regime_probs = {label_map[int(i)]: float(p) for i, p in enumerate(trans)}

    return {
        "current_state": current_state,
        "current_regime": label_map[current_state],
        "next_state_most_likely": next_state,
        "next_regime_most_likely": label_map[next_state],
        "next_state_probabilities": next_state_probs,
        "next_regime_probabilities": next_regime_probs,
    }


# -------------------------
# PDF Report
# -------------------------
def build_regime_colors(regime_names: List[str]) -> Dict[str, str]:
    """
    Stable mapping regime -> color for this run.
    Uses a categorical palette, deterministic order by sorted regime name.
    """
    # stable order
    ordered = sorted(regime_names)
    palette = sns.color_palette("tab10", n_colors=max(10, len(ordered)))
    colors = {name: palette[i % len(palette)] for i, name in enumerate(ordered)}
    return colors


def add_summary_page(pdf: PdfPages, ticker: str, price: pd.DataFrame, features: pd.DataFrame, regimes: pd.Series,
                     means: pd.DataFrame, forecast: Dict[str, object], label_map: Dict[int, str]):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"HMM Regime Report — {ticker}", fontsize=20, fontweight="bold", y=0.98)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    plt.text(0.5, 0.92, f"Generated: {generated}", ha="center", fontsize=10)

    plt.text(0.1, 0.86, f"Data period: {price.index.min().date()} → {price.index.max().date()}",
             fontsize=12, fontweight="bold")
    plt.text(0.1, 0.83, f"Feature period: {features.index.min().date()} → {features.index.max().date()}",
             fontsize=12, fontweight="bold")

    plt.text(0.1, 0.79, f"Current regime: {forecast['current_regime']}", fontsize=12, fontweight="bold", color="red")
    plt.text(0.1, 0.76, f"Most likely next: {forecast['next_regime_most_likely']}", fontsize=12, fontweight="bold")

    # Distribution table
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis("off")
    counts = regimes.value_counts()
    total = len(regimes)
    rows = []
    for r in sorted(counts.index.tolist()):
        rows.append([r, int(counts[r]), f"{(counts[r]/total*100):.1f}%"])
    if not rows:
        rows = [["N/A", 0, "0.0%"]]
    tbl = ax1.table(cellText=rows, colLabels=["Regime", "Days", "Pct"], cellLoc="left", loc="center",
                    colWidths=[0.6, 0.2, 0.2])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    ax1.set_title("Regime Distribution", fontweight="bold", pad=12)

    # State means table
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis("off")
    m = means.copy()
    m["label"] = m.index.map(lambda s: label_map[int(s)])
    m["ret_ann_%"] = (m["ret"] * 252 * 100).round(2)
    m["vol_%"] = (m["vol"] * 100).round(2)
    m = m[["label", "ret_ann_%", "vol_%"]].sort_index()
    tbl2 = ax2.table(cellText=m.values, rowLabels=m.index.astype(str), colLabels=m.columns,
                     cellLoc="center", loc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1, 1.8)
    ax2.set_title("Mean Features by Hidden State", fontweight="bold", pad=12)

    # Next-regime probabilities
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis("off")
    probs = forecast["next_regime_probabilities"]
    probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    lines = ["Next-regime probabilities (from transition matrix):", ""]
    for name, p in probs_sorted:
        lines.append(f"• {name}: {p:.3f}")
    text = "\n".join(lines)
    ax3.text(0.02, 0.98, text, va="top", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_price_regime_overlay(pdf: PdfPages, ticker: str, price: pd.DataFrame, regimes: pd.Series, regime_colors: Dict[str, str]):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.plot(price.index, price["close"], linewidth=1.6, label=f"{ticker} Close")

    # shade regimes
    for r in sorted(regimes.unique()):
        mask = regimes == r
        if not mask.any():
            continue
        dates = regimes.index[mask]
        color = regime_colors.get(r, (0.5, 0.5, 0.5))
        for d in dates:
            ax.axvspan(d, d + timedelta(days=1), alpha=0.18, color=color)

    ax.set_title(f"{ticker} Price with HMM Regime Overlay", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    # legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=regime_colors[r], edgecolor="none", alpha=0.5, label=r) for r in sorted(regimes.unique())]
    ax.legend(handles=patches, loc="upper left", fontsize=10, frameon=True)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_volatility_page(pdf: PdfPages, features: pd.DataFrame, regimes: pd.Series, regime_colors: Dict[str, str]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))

    vol_pct = features["vol"] * 100
    ax1.plot(features.index, vol_pct, linewidth=1.5, label="Realized Vol (ann.)")
    ax1.axhline(vol_pct.mean(), linestyle="--", linewidth=1.2, label="Average")
    ax1.fill_between(features.index, 0, vol_pct, alpha=0.15)

    ax1.set_title("Realized Volatility (Annualized)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Volatility (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # boxplot by regime
    data, labels = [], []
    for r in sorted(regimes.unique()):
        idx = regimes.index[regimes == r]
        m = features.index.intersection(idx)
        if len(m) > 0:
            data.append((features.loc[m, "vol"] * 100).values)
            labels.append(r)

    if data:
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for patch, r in zip(bp["boxes"], labels):
            patch.set_facecolor(regime_colors.get(r, (0.5, 0.5, 0.5)))
            patch.set_alpha(0.7)

        ax2.set_title("Volatility Distribution by Regime", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Volatility (%)")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_return_stats_page(pdf: PdfPages, features: pd.DataFrame, regimes: pd.Series, regime_colors: Dict[str, str]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))

    # rolling cum return for context
    cum = (features["ret"].fillna(0)).cumsum()
    ax1.plot(features.index, cum, linewidth=1.5)
    ax1.set_title("Cumulative Log Return (Feature Window)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cumulative log return")
    ax1.grid(True, alpha=0.3)

    # boxplot of daily returns by regime
    data, labels = [], []
    for r in sorted(regimes.unique()):
        idx = regimes.index[regimes == r]
        m = features.index.intersection(idx)
        if len(m) > 0:
            data.append((features.loc[m, "ret"] * 100).values)  # percent daily
            labels.append(r)

    if data:
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for patch, r in zip(bp["boxes"], labels):
            patch.set_facecolor(regime_colors.get(r, (0.5, 0.5, 0.5)))
            patch.set_alpha(0.7)

        ax2.axhline(0, linestyle="--", linewidth=1)
        ax2.set_title("Daily Return Distribution by Regime", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Daily return (%)")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_transition_matrix_page(pdf: PdfPages, model: GaussianHMM, label_map: Dict[int, str]):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    trans = pd.DataFrame(model.transmat_)
    # rename with labels for readability
    state_labels = [f"{i}: {label_map.get(i, str(i))}" for i in range(trans.shape[0])]
    trans.index = state_labels
    trans.columns = state_labels

    sns.heatmap(trans * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Probability (%)"}, linewidths=0.5, linecolor="black", ax=ax)
    ax.set_title("HMM Transition Matrix", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    out_path: str,
    ticker: str,
    price: pd.DataFrame,
    features: pd.DataFrame,
    regimes: pd.Series,
    means: pd.DataFrame,
    model: GaussianHMM,
    label_map: Dict[int, str],
    forecast: Dict[str, object],
):
    regime_colors = build_regime_colors(sorted(regimes.unique().tolist()))

    with PdfPages(out_path) as pdf:
        add_summary_page(pdf, ticker, price, features, regimes, means, forecast, label_map)
        add_price_regime_overlay(pdf, ticker, price.loc[features.index], regimes, regime_colors)
        add_volatility_page(pdf, features, regimes, regime_colors)
        add_return_stats_page(pdf, features, regimes, regime_colors)
        add_transition_matrix_page(pdf, model, label_map)


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HMM regime analysis (any Polygon ticker) with PDF output.")
    p.add_argument("--api-key", type=str, default=None, help="Polygon API key (or set env POLYGON_API_KEY).")
    p.add_argument("--ticker", type=str, required=True, help="Ticker (e.g., SPY, AAPL, QQQ, I:VIX, C:BTCUSD).")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (optional).")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (optional).")
    p.add_argument("--years", type=int, default=5, help="If --start not set, use last N years (default 5).")
    p.add_argument("--states", type=int, default=4, help="Number of HMM states (default 4). Try 3–6.")
    p.add_argument("--vol-window", type=int, default=20, help="Rolling window for realized vol (default 20).")
    p.add_argument("--timespan", type=str, default="day", choices=["day", "hour", "minute"], help="Polygon timespan.")
    p.add_argument("--multiplier", type=int, default=1, help="Polygon multiplier (default 1).")
    p.add_argument("--out", type=str, default=None, help="Output PDF path.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for HMM.")
    p.add_argument("--iters", type=int, default=500, help="HMM max iterations.")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key
    if api_key is None:
        # no env tool here; keep simple
        raise RuntimeError("Provide --api-key (Polygon).")

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    out_path = args.out or f"hmm_regime_{args.ticker.replace(':','_')}_{start_date}_to_{end_date}.pdf"

    print(f"Fetching {args.ticker} from {start_date} to {end_date} ...")
    price = fetch_polygon_ohlcv(
        api_key=api_key,
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        timespan=args.timespan,
        multiplier=args.multiplier,
    )

    print("Computing features (returns + realized vol) ...")
    features = compute_features(price, vol_window=args.vol_window)

    if len(features) < max(250, args.vol_window * 5):
        print(f"[WARN] Only {len(features)} feature rows. Consider longer history or smaller vol_window.")

    print(f"Fitting HMM (states={args.states}) ...")
    states, model, scaler = fit_hmm(features, n_states=args.states, n_iter=args.iters, random_state=args.seed)
    means = state_means(features, states)
    label_map = label_states(means)
    regimes = states.map(label_map).rename("regime")

    forecast = forecast_next_regime(states, model, label_map)

    # Console summary
    print("\n" + "=" * 80)
    print(f"HMM regime analysis for {args.ticker}")
    print("=" * 80)
    print(f"Feature window: {features.index.min().date()} → {features.index.max().date()}")
    print(f"Current regime: {forecast['current_regime']}")
    print(f"Most likely next regime: {forecast['next_regime_most_likely']}")
    print("\nNext regime probabilities:")
    for k, v in sorted(forecast["next_regime_probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.3f}")

    # PDF
    print(f"\nGenerating PDF report: {out_path}")
    generate_pdf_report(
        out_path=out_path,
        ticker=args.ticker,
        price=price,
        features=features,
        regimes=regimes,
        means=means,
        model=model,
        label_map=label_map,
        forecast=forecast,
    )
    print("Done.")


if __name__ == "__main__":
    main()
