import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, kci
from causallearn.utils.GraphUtils import GraphUtils
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.independence_tests.robust_parcorr import RobustParCorr
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    print("Warning: tigramite not available. PCMCI+ and LPCMCI+ methods will not be available.")
import warnings
import argparse
from typing import Dict, List, Tuple, Optional

# Suppress warnings from yfinance and sklearn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Global Settings ---

# Use a 3-year window for a decent amount of data
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

# Strategy 2 settings
N_PCA_COMPONENTS = 50

# Strategy 3 settings
N_CLUSTERS = 75  # We want a representative basket of 75 stocks

# Proxies for Strategy 1 and 3
# 11 GICS Sector ETFs + S&P 500 Index
GICS_SECTORS = [
    "SPY",  # S&P 500
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLC",  # Communication Services
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLU",  # Utilities
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLRE", # Real Estate
]

# Other key market variables
OTHER_PROXIES = [
    "^VIX", # Volatility Index
    "^TNX", # 10-Year Treasury Yield
]

PROXY_TICKERS = GICS_SECTORS + OTHER_PROXIES

# Ground truth causal graph for synthetic data (contemporaneous relationships)
# Format: {child: [list of parents]}
KNOWN_CAUSAL_GRAPH = {
    'SPY': ['^VIX'],  # VIX influences SPY
    'XLK': ['SPY', '^VIX'],  # SPY and VIX influence Technology
    'XLF': ['SPY', '^TNX'],  # SPY and TNX influence Financials
    'XLV': ['SPY'],  # SPY influences Healthcare
    'XLC': ['XLK', 'SPY'],  # Technology and SPY influence Communication
    'XLY': ['SPY', '^VIX'],  # SPY and VIX influence Consumer Discretionary
    'XLP': ['SPY'],  # SPY influences Consumer Staples
    'XLE': ['SPY'],  # SPY influences Energy
    'XLU': ['SPY'],  # SPY influences Utilities
    'XLI': ['SPY', '^VIX'],  # SPY and VIX influence Industrials
    'XLB': ['SPY'],  # SPY influences Materials
    'XLRE': ['SPY', '^TNX'],  # SPY and TNX (negatively) influence Real Estate
    '^VIX': [],  # VIX is exogenous (root node)
    '^TNX': [],  # TNX is exogenous (root node)
}


# --- 2. Data Fetching Functions ---

def generate_synthetic_market_data(n_timesteps=1000):
    """
    Generates synthetic market data with known causal structure.
    This simulates a simplified market with sector ETFs and market indices.

    Causal structure:
    - VIX influences all sectors negatively
    - TNX (interest rates) influences Financials (XLF) positively
    - SPY influences all sector ETFs
    - Technology (XLK) influences Communication (XLC)
    """
    print("\n--- Generating Synthetic Market Data ---")
    print("Simulating market with known causal relationships...")

    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='D')

    # Generate base factors
    vix = np.random.randn(n_timesteps) * 2  # Volatility
    tnx = np.cumsum(np.random.randn(n_timesteps) * 0.1) + 2  # Interest rates (cumulative for stationarity later)

    # Market index (SPY) - influenced by VIX
    spy = np.cumsum(0.001 + np.random.randn(n_timesteps) * 0.02 - 0.05 * (vix / 10))

    # Sector ETFs - influenced by SPY and other factors
    xlk = spy + np.cumsum(np.random.randn(n_timesteps) * 0.015) - 0.03 * vix  # Technology
    xlf = spy + np.cumsum(np.random.randn(n_timesteps) * 0.018) + 0.02 * tnx  # Financials (+ interest rates)
    xlv = spy + np.cumsum(np.random.randn(n_timesteps) * 0.012)  # Healthcare
    xlc = xlk * 0.3 + spy * 0.7 + np.cumsum(np.random.randn(n_timesteps) * 0.013)  # Comm (influenced by tech)
    xly = spy + np.cumsum(np.random.randn(n_timesteps) * 0.016) - 0.02 * vix  # Consumer Discretionary
    xlp = spy + np.cumsum(np.random.randn(n_timesteps) * 0.010)  # Consumer Staples (defensive)
    xle = spy + np.cumsum(np.random.randn(n_timesteps) * 0.025)  # Energy (volatile)
    xlu = spy + np.cumsum(np.random.randn(n_timesteps) * 0.011)  # Utilities (defensive)
    xli = spy + np.cumsum(np.random.randn(n_timesteps) * 0.014) - 0.02 * vix  # Industrials
    xlb = spy + np.cumsum(np.random.randn(n_timesteps) * 0.017)  # Materials
    xlre = spy + np.cumsum(np.random.randn(n_timesteps) * 0.013) - 0.03 * tnx  # Real Estate (- interest rates)

    # Convert to prices (exp of log returns for sectors, keep VIX and TNX as levels)
    data = {
        'SPY': np.exp(spy) * 300,
        'XLK': np.exp(xlk) * 150,
        'XLF': np.exp(xlf) * 35,
        'XLV': np.exp(xlv) * 130,
        'XLC': np.exp(xlc) * 70,
        'XLY': np.exp(xly) * 170,
        'XLP': np.exp(xlp) * 70,
        'XLE': np.exp(xle) * 50,
        'XLU': np.exp(xlu) * 65,
        'XLI': np.exp(xli) * 100,
        'XLB': np.exp(xlb) * 80,
        'XLRE': np.exp(xlre) * 40,
        '^VIX': 20 + vix,  # VIX as levels
        '^TNX': tnx,  # TNX as levels
    }

    df = pd.DataFrame(data, index=dates)

    # Create MultiIndex structure similar to yfinance output
    multi_df = pd.DataFrame()
    for col in df.columns:
        for field in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if field == 'Volume':
                multi_df[(field, col)] = np.random.randint(1000000, 10000000, n_timesteps)
            else:
                # Add small variations for OHLC
                multi_df[(field, col)] = df[col] * (1 + np.random.randn(n_timesteps) * 0.001)

    multi_df.index = dates
    multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)

    print(f"Generated {len(multi_df)} days of synthetic data for {len(data)} tickers")
    print("\nKnown Causal Relationships:")
    print("  - VIX → {SPY, XLK, XLY, XLI} (negative influence)")
    print("  - TNX → XLF (positive influence)")
    print("  - TNX → XLRE (negative influence)")
    print("  - SPY → All sector ETFs")
    print("  - XLK → XLC (Technology influences Communication)")

    return multi_df

def get_sp500_tickers():
    """
    Scrapes the current list of S&P 500 tickers from Wikipedia.
    """
    print("Fetching S&P 500 ticker list from Wikipedia...")
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_df = table[0]
        # Tickers in yfinance sometimes use '-' instead of '.' (e.g., BRK-B)
        tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching ticker list: {e}")
        print("Using a small fallback list for demo purposes.")
        return ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM", "JNJ", "XOM", "META", "TSLA"]

def download_data(tickers, start, end):
    """
    Downloads OHLCV data for a list of tickers.
    """
    print(f"Downloading data for {len(tickers)} assets from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError("No data downloaded. Check ticker symbols and date range.")

    # Handle single-ticker download (yfinance returns different structure)
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([data.columns, tickers])

    return data

# --- 3. Core Preprocessing Function ---

def create_feature_matrix(data_df):
    """
    Converts raw OHLCV data into a stationary feature matrix.
    - Uses log returns for equities/ETFs.
    - Uses simple differences for interest rates (TNX) and volatility (VIX).
    - Cleans and fills NaN values.
    """
    adj_close = data_df['Adj Close'].copy()

    # 1. Calculate log returns for all tickers by default
    features_df = np.log(adj_close) - np.log(adj_close.shift(1))

    # 2. Handle special non-equity tickers (VIX, TNX)
    # These are not "compounding" assets, so log returns are less appropriate.
    # We use simple arithmetic differences to make them stationary.
    if '^VIX' in features_df.columns:
        features_df['^VIX'] = adj_close['^VIX'].diff()

    if '^TNX' in features_df.columns:
        features_df['^TNX'] = adj_close['^TNX'].diff()

    # 3. Clean NaNs
    # Drop the first row (all NaNs from shift/diff)
    features_df = features_df.iloc[1:]

    # Drop columns that have *only* NaNs (e.g., ticker didn't exist)
    features_df = features_df.dropna(axis=1, how='all')

    # Fill sporadic NaNs (e.g., a stock didn't trade one day) with 0
    # This assumes "no change" for that day.
    features_df = features_df.fillna(0)

    return features_df

def create_lagged_features(data_matrix: pd.DataFrame, max_lag: int = 1) -> pd.DataFrame:
    """
    Creates lagged features for time-series causal discovery.

    This addresses the temporal nature of financial causality where
    variables at time t-k influence variables at time t.

    Args:
        data_matrix: DataFrame with features (timesteps × variables)
        max_lag: Maximum number of lags to include (default: 1)

    Returns:
        DataFrame with original and lagged features
    """
    print(f"\nCreating lagged features (max_lag={max_lag})...")

    lags = [data_matrix]

    for lag in range(1, max_lag + 1):
        lagged = data_matrix.shift(lag).add_suffix(f'_L{lag}')
        lags.append(lagged)

    lagged_matrix = pd.concat(lags, axis=1)

    # Drop NaNs introduced by shifting
    lagged_matrix = lagged_matrix.iloc[max_lag:]

    print(f"Lagged matrix shape: {lagged_matrix.shape}")
    print(f"Original variables: {data_matrix.shape[1]}")
    print(f"With lags: {lagged_matrix.shape[1]} variables")

    return lagged_matrix

def validate_causal_graph(discovered_graph: Dict[str, List[str]],
                         true_graph: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Validates discovered causal graph against ground truth.

    Computes precision, recall, F1 score, and Structural Hamming Distance (SHD).

    Args:
        discovered_graph: Dictionary mapping variables to their discovered parents
        true_graph: Dictionary mapping variables to their true parents

    Returns:
        Dictionary with validation metrics
    """
    print(f"\n{'='*60}")
    print("VALIDATION AGAINST GROUND TRUTH")
    print(f"{'='*60}\n")

    # Get all variables that exist in both graphs
    all_vars = set(discovered_graph.keys()) & set(true_graph.keys())

    # Count true positives, false positives, false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Track detailed comparison
    comparison = {}

    for child in all_vars:
        discovered_parents = set(discovered_graph.get(child, []))
        true_parents = set(true_graph.get(child, []))

        # Filter to only include parents that exist in both graphs
        discovered_parents = discovered_parents & all_vars
        true_parents = true_parents & all_vars

        tp = len(discovered_parents & true_parents)
        fp = len(discovered_parents - true_parents)
        fn = len(true_parents - discovered_parents)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

        comparison[child] = {
            'discovered': list(discovered_parents),
            'true': list(true_parents),
            'correct': list(discovered_parents & true_parents),
            'missed': list(true_parents - discovered_parents),
            'spurious': list(discovered_parents - true_parents)
        }

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Structural Hamming Distance (total edge errors)
    shd = false_positives + false_negatives

    # Print detailed comparison
    print("Detailed Comparison by Variable:")
    print("-" * 60)
    for child, details in comparison.items():
        if details['true'] or details['discovered']:
            print(f"\n{child}:")
            print(f"  True parents:       {details['true'] if details['true'] else '(none)'}")
            print(f"  Discovered parents: {details['discovered'] if details['discovered'] else '(none)'}")
            if details['correct']:
                print(f"  ✓ Correct:          {details['correct']}")
            if details['missed']:
                print(f"  ✗ Missed:           {details['missed']}")
            if details['spurious']:
                print(f"  ✗ Spurious:         {details['spurious']}")

    print(f"\n{'='*60}")
    print("VALIDATION METRICS")
    print(f"{'='*60}")
    print(f"Precision: {precision:.3f} ({true_positives}/{true_positives + false_positives} edges correct)")
    print(f"Recall:    {recall:.3f} ({true_positives}/{true_positives + false_negatives} edges found)")
    print(f"F1 Score:  {f1_score:.3f}")
    print(f"SHD:       {shd} (edge errors: {false_positives} spurious + {false_negatives} missed)")
    print(f"{'='*60}\n")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'shd': shd,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# --- 4. Strategy Implementations ---

def strategy_1_proxy(proxy_data):
    """
    Strategy 1: The Proxy (Aggregation) Approach
    The matrix *is* the preprocessed proxy data.
    """
    print("\n--- Running Strategy 1: Proxy (Aggregation) ---")
    matrix = create_feature_matrix(proxy_data)
    name = "Strategy 1: Proxy (Sectors + Market)"
    return matrix, name

def strategy_2_pca(sp500_data):
    """
    Strategy 2: The Factor (Dimensionality Reduction) Approach
    Uses PCA on the full S&P 500 log-return matrix.
    """
    print("\n--- Running Strategy 2: PCA Factors ---")

    # 1. Create the full N=500 feature matrix
    log_returns_df = create_feature_matrix(sp500_data)

    # 2. Initialize and run PCA
    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca_features = pca.fit_transform(log_returns_df)

    # 3. Format as a clean DataFrame
    pca_df = pd.DataFrame(
        pca_features,
        columns=[f'PC_{i+1}' for i in range(N_PCA_COMPONENTS)],
        index=log_returns_df.index
    )

    print(f"Explained Variance (Top 5 Components): {pca.explained_variance_ratio_[:5].sum():.2%}")
    name = f"Strategy 2: PCA ({N_PCA_COMPONENTS} Factors)"
    return pca_df, name

def strategy_3_clustering(sp500_data, proxy_data):
    """
    Strategy 3: The Representative (Clustering/Selection) Approach
    - Clusters all 500 stocks.
    - Selects the "medoid" (closest stock) from each cluster.
    - Combines these with the proxy tickers.
    """
    print(f"\n--- Running Strategy 3: Clustering (k={N_CLUSTERS}) + Proxies ---")

    # 1. Create feature matrices for S&P 500 and proxies
    log_returns_df = create_feature_matrix(sp500_data)
    proxy_features_df = create_feature_matrix(proxy_data)

    # 2. Transpose data for clustering
    # We want to cluster the *stocks* (N) based on their *time series* (T).
    # sklearn.cluster.KMeans expects (n_samples, n_features) -> (N_stocks, T_timesteps)
    data_for_clustering = log_returns_df.T

    # 3. Run KMeans
    print(f"Clustering {data_for_clustering.shape[0]} stocks...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(data_for_clustering)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 4. Find the "medoid" (most representative stock) for each cluster
    selected_tickers = []
    for i in range(N_CLUSTERS):
        # Get all stocks belonging to this cluster
        cluster_stocks = data_for_clustering[labels == i]

        if not cluster_stocks.empty:
            # Get the cluster center for this cluster
            cluster_center = centers[i].reshape(1, -1)

            # Calculate Euclidean distance from each stock to the center
            distances = cdist(cluster_stocks, cluster_center, 'euclidean')

            # Find the index of the stock with the minimum distance
            medoid_index_in_cluster = np.argmin(distances)

            # Get the ticker name (from the DataFrame's index)
            medoid_ticker = cluster_stocks.index[medoid_index_in_cluster]
            selected_tickers.append(medoid_ticker)

    print(f"Selected {len(selected_tickers)} representative stocks.")

    # 5. Create the final matrix
    cluster_stock_df = log_returns_df[selected_tickers]

    # 6. Combine with proxies
    final_df = pd.concat([cluster_stock_df, proxy_features_df], axis=1)

    # 7. Remove duplicate columns (if a proxy was also selected as a medoid)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    name = f"Strategy 3: Clustering ({len(selected_tickers)} stocks) + Proxies"
    return final_df, name

# --- 5. Causal Discovery ---

def perform_causal_discovery(data_matrix, alpha=0.05, max_vars=15, test_type='fisherz'):
    """
    Performs causal discovery using the PC algorithm.

    Args:
        data_matrix: DataFrame with features (timesteps x variables)
        alpha: Significance level for conditional independence tests
        max_vars: Maximum number of variables to analyze (for computational efficiency)
        test_type: Type of conditional independence test ('fisherz' or 'kci')
                  - fisherz: Fast, assumes Gaussian data (parametric)
                  - kci: Slower, handles non-Gaussian data (non-parametric)

    Returns:
        Tuple of (causal_graph, parent_dict)
    """
    print(f"\n{'='*60}")
    print("CAUSAL DISCOVERY ANALYSIS")
    print(f"{'='*60}")

    # Limit number of variables for computational efficiency
    if data_matrix.shape[1] > max_vars:
        print(f"\nNote: Limiting analysis to first {max_vars} variables for efficiency")
        data_matrix = data_matrix.iloc[:, :max_vars]

    # Select independence test
    if test_type == 'kci':
        indep_test = kci
        test_name = "Kernel CI (non-parametric)"
        print("\n⚠️  Using KCI test - this may take significantly longer than Fisher-Z")
    else:
        indep_test = fisherz
        test_name = "Fisher-Z (parametric)"

    print(f"\nRunning PC Algorithm on {data_matrix.shape[1]} variables...")
    print(f"Sample size: {data_matrix.shape[0]} timesteps")
    print(f"Significance level (alpha): {alpha}")
    print(f"Independence test: {test_name}")

    # Convert to numpy array
    data_array = data_matrix.values

    # Run PC algorithm
    try:
        cg = pc(data_array, alpha=alpha, indep_test=indep_test, stable=True,
                uc_rule=0, uc_priority=2, mvpc=False, correction_name='MV_Crtn_Fisher_Z',
                background_knowledge=None, verbose=False, show_progress=False)

        # Get the learned graph
        graph = cg.G.graph

        print(f"\n{'='*60}")
        print("CAUSAL RELATIONSHIPS DISCOVERED")
        print(f"{'='*60}\n")

        # Extract parent-child relationships
        variable_names = data_matrix.columns.tolist()
        parent_dict = {}

        for i, var in enumerate(variable_names):
            parents = []
            for j, potential_parent in enumerate(variable_names):
                if i != j:
                    # Check if there's an edge from j to i
                    # In the PC algorithm: -1 means <-, 1 means ->, 0 means no edge, -2 means undirected
                    edge = graph[j, i]
                    if edge == 1:  # j -> i (j is a parent of i)
                        parents.append(potential_parent)
                    elif edge == -1:  # i -> j (i is a parent of j)
                        pass  # This will be captured when we process j

            parent_dict[var] = parents

        # Print discovered relationships
        for var, parents in parent_dict.items():
            if parents:
                print(f"📊 {var} has causal parents: {', '.join(parents)}")
            else:
                print(f"📊 {var} has no causal parents (root node or isolated)")

        # Print summary statistics
        total_edges = sum(len(parents) for parents in parent_dict.values())
        print(f"\n{'='*60}")
        print(f"Summary: Discovered {total_edges} causal relationships")
        print(f"{'='*60}\n")

        return cg, parent_dict

    except Exception as e:
        print(f"\nError during causal discovery: {e}")
        print("This can happen with insufficient data or numerical issues.")
        return None, {}

def perform_pcmci_analysis(data_matrix: pd.DataFrame, max_lag: int = 5,
                           alpha: float = 0.05, method: str = 'pcmciplus',
                           use_robust: bool = False) -> Tuple[Optional[object], Dict[str, List[str]]]:
    """
    Performs time-series causal discovery using PCMCI+ or LPCMCI+.

    These methods are specifically designed for time-series data and are
    superior to manually creating lagged features + PC algorithm.

    Args:
        data_matrix: DataFrame with features (timesteps × variables)
        max_lag: Maximum time lag to consider (default: 5 days)
        alpha: Significance level for conditional independence tests
        method: 'pcmciplus' or 'lpcmciplus'
                - pcmciplus: Finds one optimal set of parents from all lags (faster, robust)
                - lpcmciplus: Finds best parents at each specific lag (slower, more detailed)
        use_robust: If True, use RobustParCorr (better for non-Gaussian data)

    Returns:
        Tuple of (PCMCI object, parent_dict)
    """
    if not TIGRAMITE_AVAILABLE:
        print("\n✗ Error: tigramite library not installed.")
        print("Install with: pip install tigramite")
        return None, {}

    print(f"\n{'='*60}")
    print(f"TIME-SERIES CAUSAL DISCOVERY: {method.upper()}")
    print(f"{'='*60}")

    print(f"\nRunning {method.upper()} on {data_matrix.shape[1]} variables...")
    print(f"Sample size: {data_matrix.shape[0]} timesteps")
    print(f"Maximum lag: {max_lag}")
    print(f"Significance level (alpha): {alpha}")

    # Select independence test
    if use_robust:
        indep_test = RobustParCorr(significance='analytic')
        test_name = "Robust ParCorr (non-Gaussian robust)"
    else:
        indep_test = ParCorr(significance='analytic')
        test_name = "ParCorr (Gaussian)"

    print(f"Independence test: {test_name}")

    if method == 'lpcmciplus':
        print("\n⚠️  LPCMCI+ is significantly slower than PCMCI+")
        print("    It tests each lag separately to find lag-specific effects")

    # Convert DataFrame to numpy array and create variable names
    data_array = data_matrix.values
    var_names = data_matrix.columns.tolist()

    # Create tigramite dataframe
    dataframe = pp.DataFrame(
        data_array,
        datatime=np.arange(len(data_array)),
        var_names=var_names
    )

    try:
        # Initialize PCMCI
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=indep_test,
            verbosity=0
        )

        # Run the selected method
        if method == 'lpcmciplus':
            # Note: LPCMCI+ may not be available in all tigramite versions
            # Check if the method exists
            if hasattr(pcmci, 'run_lpcmciplus'):
                results = pcmci.run_lpcmciplus(
                    link_assumptions=None,
                    tau_min=0,
                    tau_max=max_lag,
                    pc_alpha=alpha
                )
            else:
                print("\n⚠️  LPCMCI+ not available in this tigramite version")
                print("    Falling back to PCMCI+ (which is still excellent for time-series)")
                print("    For LPCMCI+, upgrade to tigramite>=5.3.0")
                results = pcmci.run_pcmciplus(
                    link_assumptions=None,
                    tau_min=0,
                    tau_max=max_lag,
                    pc_alpha=alpha
                )
        else:  # pcmciplus
            results = pcmci.run_pcmciplus(
                link_assumptions=None,
                tau_min=0,
                tau_max=max_lag,
                pc_alpha=alpha
            )

        # Extract results
        graph = results['graph']
        val_matrix = results['val_matrix']
        p_matrix = results['p_matrix']

        print(f"\n{'='*60}")
        print("CAUSAL RELATIONSHIPS DISCOVERED")
        print(f"{'='*60}\n")

        # Build parent dictionary
        # graph shape: (N_vars, N_vars, tau_max+1)
        # graph[i, j, tau] indicates edge type from j at lag tau to i at lag 0
        parent_dict = {}

        for i, child_var in enumerate(var_names):
            parents = []

            # Check all possible parent variables
            for j, parent_var in enumerate(var_names):
                # Check all lags from 0 to max_lag
                for tau in range(max_lag + 1):
                    # graph values: '' = no link, 'o-o' = undirected, '-->' = directed, '<--' = reverse directed, 'x-x' = conflict
                    edge_type = graph[i, j, tau]

                    if edge_type == '-->' or edge_type == '-->':  # Directed edge from j to i
                        if tau == 0:
                            parents.append(f"{parent_var}")
                        else:
                            parents.append(f"{parent_var}_L{tau}")

            parent_dict[child_var] = parents

        # Print discovered relationships
        for var, parents in parent_dict.items():
            if parents:
                parents_str = ', '.join(parents)
                print(f"📊 {var} has causal parents: {parents_str}")
            else:
                print(f"📊 {var} has no causal parents (root node or isolated)")

        # Print summary statistics
        total_edges = sum(len(parents) for parents in parent_dict.values())
        print(f"\n{'='*60}")
        print(f"Summary: Discovered {total_edges} causal relationships")
        print(f"{'='*60}\n")

        # Print lag-specific insights
        lag_counts = {}
        for parents in parent_dict.values():
            for parent in parents:
                if '_L' in parent:
                    lag = int(parent.split('_L')[1])
                    lag_counts[lag] = lag_counts.get(lag, 0) + 1
                else:
                    lag_counts[0] = lag_counts.get(0, 0) + 1

        if lag_counts:
            print("Lag Distribution of Discovered Links:")
            for lag in sorted(lag_counts.keys()):
                print(f"  Lag {lag}: {lag_counts[lag]} links")
            print()

        return pcmci, parent_dict

    except Exception as e:
        print(f"\nError during {method.upper()} analysis: {e}")
        print("This can happen with insufficient data or numerical issues.")
        import traceback
        traceback.print_exc()
        return None, {}

# --- 6. Main Execution ---

def main(use_synthetic=False, strategy_num=1, perform_causal_analysis=True,
         use_lags=False, max_lag=1, test_type='fisherz', validate=True,
         method='pc', use_robust=False):
    """
    Main function to run strategies and causal discovery.

    Args:
        use_synthetic: If True, use synthetic data instead of real market data
        strategy_num: Which strategy to run (1, 2, or 3)
        perform_causal_analysis: If True, perform causal discovery on the results
        use_lags: If True, create lagged features for time-series causality (only for PC)
        max_lag: Maximum number of lags to include (default: 1 for PC, 5 for PCMCI/LPCMCI)
        test_type: Type of CI test ('fisherz' or 'kci') - only for PC method
        validate: If True and using synthetic data, validate against ground truth
        method: Causal discovery method ('pc', 'pcmciplus', 'lpcmciplus')
        use_robust: If True, use robust tests for non-Gaussian data (RobustParCorr for PCMCI)
    """
    print(f"\n{'='*70}")
    print(f"MARKET DATA CAUSAL DISCOVERY ANALYSIS")
    print(f"{'='*70}\n")

    if use_synthetic:
        # Generate synthetic data
        raw_data = generate_synthetic_market_data(n_timesteps=1000)
        sp500_tickers = []  # Not used for synthetic data
        downloaded_proxies = PROXY_TICKERS
        raw_proxy_data = raw_data
        raw_sp500_data = raw_data
    else:
        # 1. Get Ticker Lists
        sp500_tickers = get_sp500_tickers()
        all_tickers = sorted(list(set(sp500_tickers + PROXY_TICKERS)))

        # 2. Download ALL required data in one go
        try:
            raw_data = download_data(all_tickers, START_DATE, END_DATE)
        except Exception as e:
            print(f"Failed to download data: {e}")
            print("\nFalling back to synthetic data...")
            return main(use_synthetic=True, strategy_num=strategy_num,
                       perform_causal_analysis=perform_causal_analysis,
                       use_lags=use_lags, max_lag=max_lag,
                       test_type=test_type, validate=validate,
                       method=method, use_robust=use_robust)

        # 3. Filter raw data for different strategies
        # Get tickers that were *successfully* downloaded (some may fail)
        downloaded_sp500 = list(set(sp500_tickers) & set(raw_data['Adj Close'].columns))
        downloaded_proxies = list(set(PROXY_TICKERS) & set(raw_data['Adj Close'].columns))

        if len(downloaded_proxies) == 0:
            print("\nNo data successfully downloaded. Using synthetic data instead...")
            return main(use_synthetic=True, strategy_num=strategy_num,
                       perform_causal_analysis=perform_causal_analysis,
                       use_lags=use_lags, max_lag=max_lag,
                       test_type=test_type, validate=validate,
                       method=method, use_robust=use_robust)

        print(f"\nSuccessfully downloaded data for {len(downloaded_sp500)} S&P 500 tickers.")
        print(f"Successfully downloaded data for {len(downloaded_proxies)} proxy tickers.")

        # Create filtered MultiIndex DataFrames
        raw_sp500_data = raw_data.loc(axis=1)[:, downloaded_sp500]
        raw_proxy_data = raw_data.loc(axis=1)[:, downloaded_proxies]

    # --- Execute Selected Strategy ---
    result_matrix = None
    strategy_name = None

    if strategy_num == 1:
        result_matrix, strategy_name = strategy_1_proxy(raw_proxy_data)
    elif strategy_num == 2 and not use_synthetic:
        result_matrix, strategy_name = strategy_2_pca(raw_sp500_data)
    elif strategy_num == 3 and not use_synthetic:
        result_matrix, strategy_name = strategy_3_clustering(raw_sp500_data, raw_proxy_data)
    else:
        # Default to strategy 1 for synthetic data
        result_matrix, strategy_name = strategy_1_proxy(raw_proxy_data)

    print(f"\n{'='*70}")
    print(f"Result for '{strategy_name}':")
    print(f"{'='*70}")
    print(f"Matrix Shape: {result_matrix.shape} (Timesteps × Variables)")
    print(f"\nFirst 5 rows:\n")
    print(result_matrix.head())
    print("\n" + "-" * 70 + "\n")

    # --- Perform Causal Discovery ---
    if perform_causal_analysis:
        # Select causal discovery method
        if method in ['pcmciplus', 'lpcmciplus']:
            # Use PCMCI+ or LPCMCI+ (time-series specific methods)
            # These methods handle lags internally, so we don't need create_lagged_features
            pcmci_max_lag = max_lag if max_lag > 1 else 5  # Default to 5 for PCMCI if not specified

            causal_graph, parent_dict = perform_pcmci_analysis(
                result_matrix,
                max_lag=pcmci_max_lag,
                alpha=0.05,
                method=method,
                use_robust=use_robust
            )
        else:
            # Use PC algorithm (standard constraint-based method)
            analysis_matrix = result_matrix

            # Apply lagged features if requested (for PC only)
            if use_lags:
                analysis_matrix = create_lagged_features(result_matrix, max_lag=max_lag)

            # Adjust max_vars based on whether we're using lags
            max_vars_limit = 10 if use_lags else 15
            if test_type == 'kci':
                max_vars_limit = min(max_vars_limit, 8)  # KCI is very slow

            causal_graph, parent_dict = perform_causal_discovery(
                analysis_matrix,
                alpha=0.05,
                max_vars=max_vars_limit,
                test_type=test_type
            )

        if causal_graph is not None:
            print("\n✓ Causal discovery completed successfully!")

            # --- Validate Against Ground Truth (Synthetic Data Only) ---
            if use_synthetic and validate and method == 'pc' and not use_lags:
                # Only validate contemporaneous relationships (no lags) for PC method
                # Filter parent_dict to only include variables in ground truth
                filtered_parent_dict = {
                    k: v for k, v in parent_dict.items()
                    if k in KNOWN_CAUSAL_GRAPH
                }
                metrics = validate_causal_graph(filtered_parent_dict, KNOWN_CAUSAL_GRAPH)

            return result_matrix, parent_dict
        else:
            print("\n✗ Causal discovery failed.")
            return result_matrix, {}
    else:
        return result_matrix, {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Market Data Causal Discovery Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with PC algorithm
  python causal.py --synthetic

  # RECOMMENDED: Use PCMCI+ for time-series causal discovery (faster, robust)
  python causal.py --synthetic --method pcmciplus

  # Use PCMCI+ with longer lags (detect effects up to 10 days)
  python causal.py --synthetic --method pcmciplus --max-lag 10

  # Use LPCMCI+ for lag-specific effects (slower, more detailed)
  python causal.py --synthetic --method lpcmciplus --max-lag 5

  # Use robust tests for non-Gaussian data (with PCMCI methods)
  python causal.py --synthetic --method pcmciplus --robust

  # PC algorithm with manual lagged features (old approach)
  python causal.py --synthetic --method pc --use-lags --max-lag 2

  # PC with non-parametric KCI test (slower but handles non-Gaussian data)
  python causal.py --synthetic --method pc --test-type kci

  # Skip validation
  python causal.py --synthetic --no-validate
        """
    )

    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of real market data')
    parser.add_argument('--strategy', type=int, default=1, choices=[1, 2, 3],
                       help='Strategy to use: 1=Proxy, 2=PCA, 3=Clustering (default: 1)')
    parser.add_argument('--method', type=str, default='pc',
                       choices=['pc', 'pcmciplus', 'lpcmciplus'],
                       help='Causal discovery method: pc (standard), pcmciplus (time-series, recommended), '
                            'lpcmciplus (lag-specific, slow) (default: pc)')
    parser.add_argument('--no-causal', action='store_true',
                       help='Skip causal discovery analysis')
    parser.add_argument('--use-lags', action='store_true',
                       help='Create lagged features (PC method only, not needed for PCMCI)')
    parser.add_argument('--max-lag', type=int, default=None,
                       help='Maximum lag: 1-2 for PC with --use-lags, 5-10 for PCMCI methods')
    parser.add_argument('--test-type', type=str, default='fisherz', choices=['fisherz', 'kci'],
                       help='CI test for PC method: fisherz (fast, Gaussian) or kci (slow, non-parametric)')
    parser.add_argument('--robust', action='store_true',
                       help='Use robust tests for non-Gaussian data (PCMCI methods: RobustParCorr)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation against ground truth (synthetic data only)')

    args = parser.parse_args()

    # Set default max_lag based on method if not specified
    if args.max_lag is None:
        args.max_lag = 1 if args.method == 'pc' else 5

    main(use_synthetic=args.synthetic,
         strategy_num=args.strategy,
         perform_causal_analysis=not args.no_causal,
         use_lags=args.use_lags,
         max_lag=args.max_lag,
         test_type=args.test_type,
         validate=not args.no_validate,
         method=args.method,
         use_robust=args.robust)
