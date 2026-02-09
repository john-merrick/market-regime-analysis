"""
Market Regime Analysis for S&P 500 & US Treasuries
Analyzes market conditions using liquidity, volatility, and yield curve data

UPDATED:
- Uses VIX (Polygon I:VIX) as volatility
- Ensures exactly SIX labeled regimes:
    1) Crisis
    2) High Volatility Rally
    3) Low Volatility Rally
    4) Slowdown
    5) Recovery
    6) Goldilocks
- Ensures the SAME colour mapping is used consistently across the entire PDF report
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RegimeAnalyzer:
    """
    Market regime detection using:
    - S&P 500 price proxy (SPY) (Polygon API)
    - VIX implied volatility (Polygon API, I:VIX)
    - US Treasury yields (FRED API)
    - Fed liquidity data (FRED API)
    """

    # Canonical regime order (also used for transition matrix axes)
    REGIME_ORDER = [
        'Crisis',
        'High Volatility Rally',
        'Low Volatility Rally',
        'Slowdown',
        'Recovery',
        'Goldilocks'
    ]

    # Canonical colours (used everywhere in the report)
    REGIME_COLORS = {
        'Crisis': 'red',
        'High Volatility Rally': 'orange',
        'Low Volatility Rally': 'deepskyblue',
        'Slowdown': 'gold',
        'Recovery': 'lightgreen',
        'Goldilocks': 'green'
    }

    def __init__(self, polygon_api_key: str, fred_api_key: str = None):
        self.polygon_key = polygon_api_key
        self.fred_key = fred_api_key
        self.data = {}

    # -----------------------------
    # Polygon fetchers
    # -----------------------------
    def fetch_sp500_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch S&P 500 price data from Polygon (SPY proxy)"""
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': self.polygon_key}

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Polygon API error (SPY): {response.status_code} {response.text}")

        data = response.json()
        if 'results' not in data:
            raise Exception("No SPY data returned from Polygon")

        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.normalize()
        df = df.rename(columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume'})
        df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]

        print(f"S&P 500 (SPY) data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        return df

    def fetch_vix_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch VIX index level from Polygon using ticker I:VIX.
        Returns a daily Series indexed by normalized date with values in index points (e.g., 17.5).
        """
        url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': self.polygon_key}

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Polygon API error (VIX): {response.status_code} {response.text}")

        data = response.json()
        if 'results' not in data:
            raise Exception("No VIX data returned from Polygon")

        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.normalize()
        df = df.rename(columns={'c': 'close'})
        vix = df.set_index('date')['close'].astype(float)

        print(f"VIX data: {len(vix)} rows from {vix.index.min()} to {vix.index.max()}")
        return vix

    def build_volatility_from_vix(self, vix_level: pd.Series) -> pd.Series:
        """Convert VIX level (e.g., 18) into decimal implied vol (e.g., 0.18)."""
        vix_level = vix_level.copy()
        vix_level.index = pd.to_datetime(vix_level.index).normalize()
        return (vix_level / 100.0).rename("volatility")

    # -----------------------------
    # FRED fetchers
    # -----------------------------
    def fetch_fred_data(self, series_id: str, start_date: str) -> pd.Series:
        """Fetch data from FRED API; fallback to pandas_datareader if available."""
        if self.fred_key:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_key,
                'file_type': 'json',
                'observation_start': start_date
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data:
                    df = pd.DataFrame(data['observations'])
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    return df.set_index('date')['value']
            else:
                print(f"[WARN] FRED API HTTP {response.status_code} for {series_id}. Trying fallback...")

        try:
            from pandas_datareader import data as pdr
            out = pdr.DataReader(series_id, 'fred', start_date)
            if isinstance(out, pd.DataFrame):
                if out.shape[1] == 1:
                    return out.iloc[:, 0]
                return out.squeeze()
            return out
        except Exception as e:
            print(f"[WARN] Could not fetch {series_id}: {e}")
            return pd.Series(dtype=float)

    def fetch_treasury_yields(self, start_date: str) -> pd.DataFrame:
        """Fetch Treasury yields from FRED"""
        yields = {
            '3M': 'DGS3MO',
            '2Y': 'DGS2',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }

        df = pd.DataFrame()
        for name, series in yields.items():
            s = self.fetch_fred_data(series, start_date)
            if not s.empty:
                df[name] = s

        df = df.ffill().bfill()
        print(f"Treasury yields: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        return df

    def fetch_liquidity_data(self, start_date: str) -> pd.DataFrame:
        """
        Fetch Fed liquidity indicators from FRED:
        - WALCL: Federal Reserve Total Assets
        - RRPONTSYD: Reverse Repo
        - WTREGEN: Treasury General Account
        Net Liquidity = WALCL - RRPONTSYD - WTREGEN
        """
        liquidity = pd.DataFrame()

        fed_assets = self.fetch_fred_data('WALCL', start_date)
        if not fed_assets.empty:
            liquidity['fed_assets'] = fed_assets

        rrp = self.fetch_fred_data('RRPONTSYD', start_date)
        if not rrp.empty:
            liquidity['reverse_repo'] = rrp

        tga = self.fetch_fred_data('WTREGEN', start_date)
        if not tga.empty:
            liquidity['tga'] = tga

        liquidity = liquidity.ffill().bfill()

        if all(col in liquidity.columns for col in ['fed_assets', 'reverse_repo', 'tga']):
            liquidity['net_liquidity'] = liquidity['fed_assets'] - liquidity['reverse_repo'] - liquidity['tga']
        else:
            liquidity['net_liquidity'] = liquidity.get('fed_assets', 0)

        print(f"Liquidity data: {len(liquidity)} rows from {liquidity.index.min()} to {liquidity.index.max()}")
        return liquidity

    # -----------------------------
    # Feature engineering
    # -----------------------------
    def calculate_yield_curve_features(self, yields: pd.DataFrame) -> pd.DataFrame:
        """Calculate yield curve characteristics"""
        features = pd.DataFrame(index=yields.index)
        features['slope'] = yields['10Y'] - yields['2Y']                          # 10-2
        features['curvature'] = 2 * yields['2Y'] - yields['3M'] - yields['10Y']   # proxy
        features['level'] = yields.mean(axis=1)
        features['term_spread'] = yields['30Y'] - yields['3M']
        return features

    def prepare_regime_features(
        self,
        sp500: pd.DataFrame,
        yields: pd.DataFrame,
        liquidity: pd.DataFrame,
        vix: pd.Series
    ) -> pd.DataFrame:
        """Prepare feature matrix for regime detection"""
        sp500 = sp500.copy()

        sp500['returns'] = np.log(sp500['close'] / sp500['close'].shift(1))
        sp500['momentum'] = sp500['close'].pct_change(60)

        vix_vol = self.build_volatility_from_vix(vix)

        yc_features = self.calculate_yield_curve_features(yields)

        liq_features = pd.DataFrame(index=liquidity.index)
        liq_features['net_liquidity'] = liquidity['net_liquidity']
        liq_features['liquidity_change'] = liquidity['net_liquidity'].pct_change(20)
        liq_features['liquidity_trend'] = liquidity['net_liquidity'].rolling(60).mean()

        sp500.index = pd.to_datetime(sp500.index).normalize()
        vix_vol.index = pd.to_datetime(vix_vol.index).normalize()
        yc_features.index = pd.to_datetime(yc_features.index).normalize()
        liq_features.index = pd.to_datetime(liq_features.index).normalize()

        sp500_daily = sp500[['returns', 'momentum']].copy()
        vix_daily = vix_vol.to_frame().resample('D').ffill()
        yc_daily = yc_features.resample('D').ffill()
        liq_daily = liq_features.resample('D').ffill()

        features = sp500_daily.join(vix_daily, how='left')
        features = features.join(yc_daily, how='left')
        features = features.join(liq_daily, how='left')

        features = features.ffill().dropna()

        print(f"Features shape after preparation: {features.shape}")
        if features.empty or len(features) < 100:
            print("[WARN] Features are sparse. Check overlap between SPY/VIX and FRED dates.")
        return features

    # -----------------------------
    # Regime detection & labeling
    # -----------------------------
    def analyze_regime_characteristics(self, features: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """Mean feature values per cluster"""
        stats = pd.DataFrame()
        for r in regimes.unique():
            stats[r] = features[regimes == r].mean()
        return stats

    def label_regimes_six(self, regime_stats: pd.DataFrame) -> Dict[int, str]:
        """
        Deterministically map clusters -> EXACTLY six regime labels:
            'Crisis', 'High Volatility Rally', 'Low Volatility Rally',
            'Slowdown', 'Recovery', 'Goldilocks'
        """

        vol = regime_stats.loc['volatility'].copy()
        mom = regime_stats.loc['momentum'].copy()
        slope = regime_stats.loc['slope'].copy()
        liq = regime_stats.loc['liquidity_change'].copy()

        labels: Dict[int, str] = {}
        remaining = set(regime_stats.columns.tolist())

        # Crisis: high vol + negative momentum
        crisis_score = vol.rank(pct=True) + (-mom).rank(pct=True)
        crisis_cluster = crisis_score.loc[list(remaining)].idxmax()
        labels[crisis_cluster] = 'Crisis'
        remaining.remove(crisis_cluster)

        # Slowdown: most inverted curve
        slowdown_cluster = slope.loc[list(remaining)].idxmin()
        labels[slowdown_cluster] = 'Slowdown'
        remaining.remove(slowdown_cluster)

        # Goldilocks: low vol + positive slope + positive liquidity + positive momentum
        goldi_score = (-vol).rank(pct=True) + slope.rank(pct=True) + liq.rank(pct=True) + mom.rank(pct=True)
        goldi_cluster = goldi_score.loc[list(remaining)].idxmax()
        labels[goldi_cluster] = 'Goldilocks'
        remaining.remove(goldi_cluster)

        # Rally clusters among remaining
        rem_list = list(remaining)
        pos_mom = [c for c in rem_list if mom.loc[c] > 0]
        if len(pos_mom) == 0:
            pos_mom = mom.loc[rem_list].sort_values(ascending=False).index.tolist()

        if len(pos_mom) >= 2:
            high_vol_rally = vol.loc[pos_mom].idxmax()
            labels[high_vol_rally] = 'High Volatility Rally'
            remaining.remove(high_vol_rally)

            rem_list = list(remaining)
            pos_mom = [c for c in rem_list if mom.loc[c] > 0]
            if len(pos_mom) == 0:
                pos_mom = mom.loc[rem_list].sort_values(ascending=False).index.tolist()

            low_vol_rally = vol.loc[pos_mom].idxmin()
            labels[low_vol_rally] = 'Low Volatility Rally'
            remaining.remove(low_vol_rally)
        else:
            only = pos_mom[0]
            rem_vol_median = vol.loc[list(remaining)].median()
            if vol.loc[only] >= rem_vol_median:
                labels[only] = 'High Volatility Rally'
            else:
                labels[only] = 'Low Volatility Rally'
            remaining.remove(only)

            rem_list = list(remaining)
            if len(rem_list) > 0:
                next_best = mom.loc[rem_list].idxmax()
                other = 'Low Volatility Rally' if labels.get(only) == 'High Volatility Rally' else 'High Volatility Rally'
                labels[next_best] = other
                remaining.remove(next_best)

        # Recovery: whatever remains
        if len(remaining) == 1:
            rec = next(iter(remaining))
            labels[rec] = 'Recovery'
            remaining.remove(rec)
        elif len(remaining) > 1:
            for c in sorted(list(remaining), key=lambda x: mom.loc[x], reverse=True):
                labels[c] = 'Recovery'
            remaining.clear()

        return labels

    def detect_regimes(self, features: pd.DataFrame, n_regimes: int = 6) -> Tuple[pd.Series, pd.DataFrame]:
        """Detect market regimes using K-Means clustering (use n_regimes=6)."""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=100)
        clusters = kmeans.fit_predict(features_scaled)

        cluster_series = pd.Series(clusters, index=features.index, name='cluster')
        regime_stats = self.analyze_regime_characteristics(features, cluster_series)

        label_map = self.label_regimes_six(regime_stats)
        regimes = cluster_series.map(label_map).rename('regime')

        return regimes, regime_stats

    # -----------------------------
    # End-to-end
    # -----------------------------
    def run_analysis(self, start_date: str, end_date: str, n_regimes: int = 6) -> Dict[str, object]:
        """Run complete regime analysis"""
        if n_regimes != 6:
            print("[WARN] For exactly six regimes, set n_regimes=6.")

        print("Fetching S&P 500 data from Polygon...")
        sp500 = self.fetch_sp500_data(start_date, end_date)

        print("Fetching VIX data from Polygon...")
        vix = self.fetch_vix_data(start_date, end_date)

        print("Fetching Treasury yields from FRED...")
        yields = self.fetch_treasury_yields(start_date)

        print("Fetching liquidity data from FRED...")
        liquidity = self.fetch_liquidity_data(start_date)

        print("Preparing features...")
        features = self.prepare_regime_features(sp500, yields, liquidity, vix=vix)

        print("Detecting regimes...")
        regimes, regime_stats = self.detect_regimes(features, n_regimes=n_regimes)

        self.data = {
            'sp500': sp500,
            'vix': vix,
            'yields': yields,
            'liquidity': liquidity,
            'features': features,
            'regimes': regimes,
            'regime_stats': regime_stats
        }
        return self.data

    def get_current_regime(self) -> str:
        if 'regimes' in self.data and len(self.data['regimes']) > 0:
            return str(self.data['regimes'].iloc[-1])
        return "No analysis run yet"

    def print_regime_summary(self):
        if not self.data:
            print("No analysis data available. Run analysis first.")
            return

        regimes = self.data['regimes']
        regime_stats = self.data['regime_stats']

        print("\n" + "=" * 60)
        print("MARKET REGIME ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nCurrent Regime: {regimes.iloc[-1]}")
        print(f"Analysis Period: {regimes.index[0].date()} to {regimes.index[-1].date()}")

        print("\n" + "-" * 60)
        print("CLUSTER MEAN FEATURES (regime_stats)")
        print("-" * 60)
        print(regime_stats.round(4))

        print("\n" + "-" * 60)
        print("REGIME DISTRIBUTION")
        print("-" * 60)

        counts = regimes.value_counts()
        for r in self.REGIME_ORDER:
            if r in counts.index:
                c = int(counts[r])
                pct = c / len(regimes) * 100
                print(f"{r}: {c} days ({pct:.1f}%)")

        present = set(regimes.unique().tolist())
        expected = set(self.REGIME_ORDER)
        if present != expected:
            print("\n[WARN] Regime labels present differ from expected set of 6.")
            print("Missing:", sorted(list(expected - present)))
            print("Extra:", sorted(list(present - expected)))

    # -----------------------------
    # PDF reporting
    # -----------------------------
    def generate_regime_report(self, output_path: str = 'regime_analysis_report.pdf'):
        if not self.data:
            print("No analysis data available. Run analysis first.")
            return

        with PdfPages(output_path) as pdf:
            self._create_summary_page(pdf)
            self._create_sp500_regime_chart(pdf)
            self._create_volatility_chart(pdf)
            self._create_yield_curve_chart(pdf)
            self._create_liquidity_chart(pdf)
            self._create_regime_heatmap(pdf)
            self._create_transition_matrix(pdf)

        print(f"\nReport generated: {output_path}")

    def _regime_color_list(self, regimes_in_order):
        """Helper to map a list of regime names to colour list using canonical mapping."""
        return [self.REGIME_COLORS.get(r, 'gray') for r in regimes_in_order]

    def _create_summary_page(self, pdf):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Market Regime Analysis Report', fontsize=20, fontweight='bold', y=0.98)

        plt.text(0.5, 0.92, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 ha='center', fontsize=10)

        regimes = self.data['regimes']
        regime_stats = self.data['regime_stats']

        # plt.text(0.1, 0.85, f"Analysis Period: {regimes.index[0].date()} to {regimes.index[-1].date()}",
        #          fontsize=12, fontweight='bold')
        # plt.text(0.1, 0.82, f"Current Regime: {regimes.iloc[-1]}",
        #          fontsize=12, fontweight='bold', color='red')

        # Regime distribution table
        ax1 = fig.add_subplot(2, 2, 1)
        counts = regimes.value_counts()
        pcts = (counts / len(regimes) * 100).round(1)

        table_data = []
        for r in self.REGIME_ORDER:
            if r in counts.index:
                table_data.append([r, int(counts[r]), f"{pcts[r]}%"])
            else:
                table_data.append([r, 0, "0.0%"])

        table = ax1.table(cellText=table_data,
                          colLabels=['Regime', 'Days', 'Percentage'],
                          cellLoc='left', loc='center',
                          colWidths=[0.55, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # colour the first column cells to match regime colours (consistent)
        for i, r in enumerate(self.REGIME_ORDER, start=1):  # +1 for header row
            cell = table[(i, 0)]
            cell.set_facecolor(self.REGIME_COLORS.get(r, 'white'))
            cell.set_alpha(0.25)

        ax1.axis('off')
        ax1.set_title('Regime Distribution', fontweight='bold', pad=20)

        # Regime characteristics table (cluster means)
        ax2 = fig.add_subplot(2, 2, 2)
        stats_display = regime_stats.T[['returns', 'volatility', 'slope', 'liquidity_change']].copy()
        stats_display.columns = ['Return', 'VIX Vol', 'Yield Slope', 'Liquidity Δ']
        stats_display['Return'] = (stats_display['Return'] * 252 * 100).round(2)
        stats_display['VIX Vol'] = (stats_display['VIX Vol'] * 100).round(2)
        stats_display['Yield Slope'] = stats_display['Yield Slope'].round(2)
        stats_display['Liquidity Δ'] = (stats_display['Liquidity Δ'] * 100).round(2)

        table2 = ax2.table(cellText=stats_display.values,
                           rowLabels=stats_display.index,
                           colLabels=stats_display.columns,
                           cellLoc='center', loc='center',
                           colWidths=[0.25, 0.25, 0.25, 0.25])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        ax2.axis('off')
        ax2.set_title('Average Cluster Characteristics', fontweight='bold', pad=20)

        # Regime pie chart (consistent colours + consistent order)
        ax3 = fig.add_subplot(2, 2, 3)
        pie_counts = [int(counts.get(r, 0)) for r in self.REGIME_ORDER]
        pie_labels = [r for r in self.REGIME_ORDER if int(counts.get(r, 0)) > 0]
        pie_sizes = [int(counts.get(r, 0)) for r in self.REGIME_ORDER if int(counts.get(r, 0)) > 0]
        pie_colors = [self.REGIME_COLORS[r] for r in self.REGIME_ORDER if int(counts.get(r, 0)) > 0]

        if sum(pie_sizes) > 0:
            ax3.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                    colors=pie_colors, startangle=90)
        ax3.set_title('Regime Time Allocation', fontweight='bold')

        # Key insights
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        insights = [
            "Key Insights:",
            "",
            f"• Current: {regimes.iloc[-1]}",
            f"• Most Common: {counts.idxmax()} ({(counts.max()/len(regimes)*100):.1f}%)",
            f"• Total Regimes (labels present): {len(counts)}",
        ]
        y = 0.9
        for line in insights:
            weight = 'bold' if line.startswith("Key") else 'normal'
            ax4.text(0.1, y, line, fontsize=11, fontweight=weight, va='top')
            y -= 0.12

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_sp500_regime_chart(self, pdf):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[3, 1])

        sp500 = self.data['sp500']
        regimes = self.data['regimes']

        ax1.plot(sp500.index, sp500['close'], color='black', linewidth=1.5, label='S&P 500 (SPY)')

        for r in self.REGIME_ORDER:
            mask = regimes == r
            if not mask.any():
                continue
            dates = regimes[mask].index
            color = self.REGIME_COLORS.get(r, 'gray')
            for d in dates:
                if d in sp500.index:
                    ax1.axvspan(d, d + timedelta(days=1), alpha=0.25, color=color)

        ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax1.set_title('S&P 500 Price with Regime Overlay', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # Timeline (factorize) + legend with consistent colours
        regime_numeric = pd.Series(pd.Categorical(regimes, categories=self.REGIME_ORDER).codes, index=regimes.index)

        ax2.fill_between(regime_numeric.index, 0, regime_numeric.values, step='post', alpha=0.7)

        # Add a consistent legend (patches)
        from matplotlib.patches import Patch
        patches = [Patch(facecolor=self.REGIME_COLORS[r], edgecolor='none', alpha=0.6, label=r) for r in self.REGIME_ORDER]
        ax2.legend(handles=patches, loc='upper left', ncol=2, fontsize=9, frameon=True)

        ax2.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_volatility_chart(self, pdf):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))

        features = self.data['features']
        regimes = self.data['regimes']

        ax1.plot(features.index, features['volatility'] * 100, linewidth=1.5, label='VIX (Implied Vol)')
        ax1.axhline(features['volatility'].mean() * 100, linestyle='--', label='Average')
        ax1.fill_between(features.index, 0, features['volatility'] * 100, alpha=0.25)

        ax1.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
        ax1.set_title('VIX (Implied Volatility)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Box plot (order + consistent colours)
        regime_vol_data, labels = [], []
        for r in self.REGIME_ORDER:
            mask = regimes == r
            if not mask.any():
                continue
            dates = regimes[mask].index
            m = features.index.intersection(dates)
            if len(m) > 0:
                regime_vol_data.append(features.loc[m, 'volatility'].values * 100)
                labels.append(r)

        if regime_vol_data:
            bp = ax2.boxplot(regime_vol_data, labels=labels, patch_artist=True)
            for patch, r in zip(bp['boxes'], labels):
                patch.set_facecolor(self.REGIME_COLORS.get(r, 'gray'))
                patch.set_alpha(0.7)

            ax2.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Regime', fontsize=12, fontweight='bold')
            ax2.set_title('Volatility Distribution by Regime', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_yield_curve_chart(self, pdf):
        fig = plt.figure(figsize=(11, 8.5))

        yields = self.data['yields']
        features = self.data['features']
        regimes = self.data['regimes']

        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(features.index, features['slope'], linewidth=1.5)
        ax1.axhline(0, linestyle='--', linewidth=1)
        ax1.set_title('Yield Curve Slope (10Y - 2Y)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Spread (pct-pts)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        for col in ['3M', '2Y', '10Y', '30Y']:
            if col in yields.columns:
                ax2.plot(yields.index, yields[col], label=col, linewidth=1.5)
        ax2.set_title('Treasury Yield Levels', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Yield (%)', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Average yield curve by regime (consistent colours)
        ax3 = plt.subplot(2, 2, 3)
        for r in self.REGIME_ORDER:
            mask = regimes == r
            if not mask.any():
                continue
            dates = regimes[mask].index
            m = yields.index.intersection(dates)
            if len(m) > 0:
                avg_curve = yields.loc[m, ['3M', '2Y', '10Y', '30Y']].mean()
                ax3.plot(
                    ['3M', '2Y', '10Y', '30Y'],
                    avg_curve.values,
                    marker='o',
                    linewidth=2,
                    label=r,
                    color=self.REGIME_COLORS.get(r, 'gray')
                )
        ax3.set_title('Average Yield Curve by Regime', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Maturity', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Yield (%)', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Slope distribution by regime (consistent colours/order)
        ax4 = plt.subplot(2, 2, 4)
        slope_data, labels = [], []
        for r in self.REGIME_ORDER:
            mask = regimes == r
            if not mask.any():
                continue
            dates = regimes[mask].index
            m = features.index.intersection(dates)
            if len(m) > 0:
                slope_data.append(features.loc[m, 'slope'].values)
                labels.append(r)

        if slope_data:
            bp = ax4.boxplot(slope_data, labels=labels, patch_artist=True)
            for patch, r in zip(bp['boxes'], labels):
                patch.set_facecolor(self.REGIME_COLORS.get(r, 'gray'))
                patch.set_alpha(0.7)
            ax4.axhline(0, linestyle='--', linewidth=1)
            ax4.set_title('Yield Slope by Regime', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Slope (pct-pts)', fontsize=10, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_liquidity_chart(self, pdf):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))

        liquidity = self.data['liquidity']
        features = self.data['features']
        regimes = self.data['regimes']

        ax1.plot(liquidity.index, liquidity['net_liquidity'] / 1e6, linewidth=1.5, label='Net Liquidity')
        ax1.set_ylabel('Net Liquidity ($T)', fontsize=12, fontweight='bold')
        ax1.set_title('Fed Net Liquidity (Assets - RRP - TGA)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        liq_data, labels = [], []
        for r in self.REGIME_ORDER:
            mask = regimes == r
            if not mask.any():
                continue
            dates = regimes[mask].index
            m = features.index.intersection(dates)
            if len(m) > 0:
                liq_data.append(features.loc[m, 'liquidity_change'].values * 100)
                labels.append(r)

        if liq_data:
            bp = ax2.boxplot(liq_data, labels=labels, patch_artist=True)
            for patch, r in zip(bp['boxes'], labels):
                patch.set_facecolor(self.REGIME_COLORS.get(r, 'gray'))
                patch.set_alpha(0.7)
            ax2.axhline(0, linestyle='--', linewidth=1)
            ax2.set_ylabel('Liquidity Change (%)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Regime', fontsize=12, fontweight='bold')
            ax2.set_title('Liquidity Changes by Regime', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_regime_heatmap(self, pdf):
        """
        Heatmap is by CLUSTER (not label), but we keep a consistent colourbar.
        If you want heatmap rows ordered by regime label, that requires mapping cluster->label order;
        you can do that later if you want.
        """
        fig, ax = plt.subplots(figsize=(11, 8.5))

        regime_stats = self.data['regime_stats'].T
        stats_normalized = (regime_stats - regime_stats.mean()) / regime_stats.std()

        sns.heatmap(stats_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, cbar_kws={'label': 'Z-Score'}, ax=ax,
                    linewidths=0.5, linecolor='black')

        ax.set_title('Cluster Characteristics Heatmap (Normalized)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_transition_matrix(self, pdf):
        fig, ax = plt.subplots(figsize=(11, 8.5))

        regimes = self.data['regimes']
        labels = self.REGIME_ORDER
        n = len(labels)
        mat = np.zeros((n, n))

        for i in range(len(regimes) - 1):
            cur = regimes.iloc[i]
            nxt = regimes.iloc[i + 1]
            if cur in labels and nxt in labels:
                mat[labels.index(cur), labels.index(nxt)] += 1

        row_sums = mat.sum(axis=1, keepdims=True)
        probs = np.divide(mat, row_sums, where=row_sums != 0, out=np.zeros_like(mat))

        sns.heatmap(probs * 100, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Probability (%)'},
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor='black', ax=ax)

        ax.set_title('Regime Transition Probability Matrix', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Next Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('Current Regime', fontsize=12, fontweight='bold')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    POLYGON_KEY = "116xopGPdlPmt4vYkX9BWphPgXtO8wy9"
    FRED_KEY = "f1bebd5081a74baa210b913b10c2c8cb"

    analyzer = RegimeAnalyzer(
        polygon_api_key=POLYGON_KEY,
        fred_api_key=FRED_KEY
    )

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')

    # IMPORTANT: set n_regimes=6 to ensure 6 clusters and 6 labels
    results = analyzer.run_analysis(
        start_date=start_date,
        end_date=end_date,
        n_regimes=6
    )

    analyzer.print_regime_summary()
    analyzer.generate_regime_report('regime_analysis_report.pdf')

    print(f"\nCurrent market regime: {analyzer.get_current_regime()}")
    print(f"\nLatest S&P 500 close (SPY): ${results['sp500']['close'].iloc[-1]:.2f}")
    print(f"Latest VIX level: {results['vix'].iloc[-1]:.2f}")
    print(f"Current VIX-implied vol: {results['features']['volatility'].iloc[-1]:.2%}")
    print(f"10Y-2Y spread: {results['features']['slope'].iloc[-1]:.2f}%")
