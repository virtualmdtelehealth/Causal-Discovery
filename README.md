# Market Data Causal Discovery

A Python tool for discovering causal relationships in financial market data using constraint-based causal discovery algorithms. This project implements three different data reduction strategies and applies the PC (Peter-Clark) algorithm to uncover causal parent-child relationships between market variables.

## Overview

This project addresses the challenge of performing causal discovery on high-dimensional financial data (e.g., S&P 500 with 500+ stocks). It implements three strategies to reduce dimensionality while preserving meaningful causal relationships:

1. **Strategy 1: Proxy (Aggregation)** - Uses sector ETFs and market indices as proxies
2. **Strategy 2: PCA (Factor)** - Reduces dimensions using Principal Component Analysis
3. **Strategy 3: Clustering (Representative)** - Selects representative stocks via K-means clustering

After dimensionality reduction, the PC algorithm is applied to discover causal relationships and identify causal parents for each variable.

## Features

### Core Capabilities
- **Flexible Data Sources**: Works with both real market data (via yfinance) and synthetic data
- **Multiple Dimensionality Reduction Strategies**: Choose the approach that best fits your analysis needs
- **Synthetic Data Generation**: Includes a synthetic market data generator with known causal structure for testing and validation

### Advanced Causal Discovery
- **Multiple Independence Tests**:
  - **Fisher-Z Test**: Fast, parametric test assuming Gaussian data (default)
  - **Kernel CI (KCI) Test**: Non-parametric test for non-Gaussian financial data
- **Time-Series Causality**:
  - Lagged feature creation to capture temporal causal relationships
  - Critical for financial data where causality is often delayed
- **Ground Truth Validation**:
  - Automatic validation against known causal structure (synthetic data)
  - Precision, Recall, F1-Score, and Structural Hamming Distance (SHD) metrics
  - Detailed comparison showing correct, missed, and spurious edges

### Output
- **Comprehensive Results**: Displays discovered causal relationships in an easy-to-read format
- **Validation Metrics**: Quantifies algorithm performance against ground truth

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.28
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- causal-learn >= 0.1.3.3
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- networkx >= 3.1
- lxml >= 4.9.0
- html5lib >= 1.1

## Usage

### Basic Usage with Synthetic Data

The simplest way to test the tool is with synthetic data:

```bash
python causal.py --synthetic
```

This generates 1000 days of synthetic market data with known causal relationships and performs causal discovery.

### Using Real Market Data

To use real market data from Yahoo Finance:

```bash
python causal.py
```

Note: Real market data download may fail due to rate limiting or API restrictions. In such cases, the script automatically falls back to synthetic data.

### Command-Line Options

```bash
python causal.py [OPTIONS]
```

#### Basic Options
- `--synthetic`: Use synthetic data instead of real market data
- `--strategy {1,2,3}`: Choose dimensionality reduction strategy (default: 1)
  - 1: Proxy/Aggregation approach
  - 2: PCA/Factor approach
  - 3: Clustering/Representative approach
- `--no-causal`: Skip causal discovery analysis (only perform data reduction)

#### Advanced Options
- `--use-lags`: Create lagged features for time-series causal discovery
- `--max-lag N`: Maximum number of lags to include (default: 1)
- `--test-type {fisherz,kci}`: Conditional independence test
  - `fisherz`: Fisher-Z test (fast, assumes Gaussian data) - default
  - `kci`: Kernel CI test (slow, handles non-Gaussian data)
- `--no-validate`: Skip validation against ground truth (synthetic data only)

### Examples

#### Basic Usage

```bash
# Basic usage with synthetic data and validation
python causal.py --synthetic

# Run with real data using Strategy 2 (PCA)
python causal.py --strategy 2

# Run with synthetic data but skip causal discovery
python causal.py --synthetic --no-causal
```

#### Advanced Time-Series Analysis

```bash
# Use lagged features to capture temporal causality
python causal.py --synthetic --use-lags --max-lag 1

# Use longer lags (2-day lagged relationships)
python causal.py --synthetic --use-lags --max-lag 2
```

#### Non-Gaussian Data Handling

```bash
# Use KCI test for non-Gaussian financial data (slower but more robust)
python causal.py --synthetic --test-type kci

# Combine KCI with lagged features
python causal.py --synthetic --use-lags --test-type kci
```

#### Validation and Testing

```bash
# Run without validation (faster)
python causal.py --synthetic --no-validate

# Full analysis: lags + KCI + validation
python causal.py --synthetic --use-lags --max-lag 1 --test-type kci
```

## Output

### Sample Output

```
======================================================================
MARKET DATA CAUSAL DISCOVERY ANALYSIS
======================================================================

--- Generating Synthetic Market Data ---
Simulating market with known causal relationships...
Generated 1000 days of synthetic data for 14 tickers

Known Causal Relationships:
  - VIX → {SPY, XLK, XLY, XLI} (negative influence)
  - TNX → XLF (positive influence)
  - TNX → XLRE (negative influence)
  - SPY → All sector ETFs
  - XLK → XLC (Technology influences Communication)

--- Running Strategy 1: Proxy (Aggregation) ---

======================================================================
Result for 'Strategy 1: Proxy (Sectors + Market)':
======================================================================
Matrix Shape: (999, 14) (Timesteps × Variables)

============================================================
CAUSAL DISCOVERY ANALYSIS
============================================================

Running PC Algorithm on 14 variables...
Sample size: 999 timesteps
Significance level (alpha): 0.05

============================================================
CAUSAL RELATIONSHIPS DISCOVERED
============================================================

📊 SPY has causal parents: XLC, XLRE
📊 XLK has causal parents: XLC
📊 XLF has no causal parents (root node or isolated)
📊 XLV has causal parents: XLRE
...

============================================================
Summary: Discovered 5 causal relationships
============================================================

✓ Causal discovery completed successfully!
```

## Data Reduction Strategies

### Strategy 1: Proxy (Aggregation)

Uses sector ETFs and market indices as proxies for the entire market:

- **GICS Sector ETFs**: SPY, XLK, XLF, XLV, XLC, XLY, XLP, XLE, XLU, XLI, XLB, XLRE
- **Market Variables**: ^VIX (Volatility Index), ^TNX (10-Year Treasury Yield)

This approach is computationally efficient and uses economically meaningful variables.

### Strategy 2: PCA (Factor)

Applies Principal Component Analysis to extract 50 latent factors from the full S&P 500 return matrix. This approach:

- Captures maximum variance with fewer variables
- Creates orthogonal factors
- Reduces dimensionality from 500+ to 50

### Strategy 3: Clustering (Representative)

Uses K-means clustering to:

1. Cluster all S&P 500 stocks based on their time series behavior
2. Select the "medoid" (most representative stock) from each cluster
3. Combine selected stocks with sector proxies

This approach maintains interpretability while reducing dimensionality.

## Causal Discovery Algorithm

The tool uses the **PC (Peter-Clark) algorithm**, a constraint-based method for causal discovery:

- **Conditional Independence Tests**:
  - **Fisher-Z Test** (default): Fast, parametric test assuming Gaussian data
  - **Kernel CI (KCI) Test**: Non-parametric test for non-Gaussian financial data
    - Recommended for financial time series (leptokurtic, fat-tailed distributions)
    - Significantly slower but more robust
- **Significance Level**: α = 0.05 (default)
- **Output**: Directed acyclic graph (DAG) showing causal relationships

### Time-Series Causality

Financial causality is often **lagged** (e.g., VIX spike at t-1 causes SPY drop at t). The tool addresses this with:

- **Lagged Features**: Automatically creates features for variables at t-k where k = 1, 2, ...
- **Example**: With `--use-lags --max-lag 2`, each variable V becomes:
  - V (current value)
  - V_L1 (1-day lag)
  - V_L2 (2-day lag)
- This allows the PC algorithm to discover temporal causal relationships

### Interpreting Results

- **Causal Parents**: Variables that directly influence another variable
- **Root Nodes**: Variables with no causal parents (exogenous variables)
- **Direction**: A → B means "A causes B" or "A is a causal parent of B"
- **Lagged Relationships**: A_L1 → B means "A at time t-1 causes B at time t"

### Validation Metrics

When using synthetic data, the tool automatically validates discovered relationships:

- **Precision**: Proportion of discovered edges that are correct
  - `Precision = True Positives / (True Positives + False Positives)`
- **Recall**: Proportion of true edges that were discovered
  - `Recall = True Positives / (True Positives + False Negatives)`
- **F1 Score**: Harmonic mean of precision and recall
  - `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Structural Hamming Distance (SHD)**: Total number of edge errors
  - `SHD = False Positives + False Negatives`

Example validation output:
```
============================================================
VALIDATION METRICS
============================================================
Precision: 0.750 (3/4 edges correct)
Recall:    0.600 (3/5 edges found)
F1 Score:  0.667
SHD:       3 (edge errors: 1 spurious + 2 missed)
============================================================
```

## Synthetic Data Generation

The synthetic data generator creates realistic market data with known causal structure:

```python
# Known relationships in synthetic data:
VIX → SPY, XLK, XLY, XLI  (negative influence)
TNX → XLF                  (positive influence)
TNX → XLRE                 (negative influence)
SPY → All sector ETFs      (market-wide influence)
XLK → XLC                  (technology influences communication)
```

This allows for validation and testing of the causal discovery algorithm.

## Technical Details

### Data Preprocessing

1. **Log Returns**: Applied to equity prices for stationarity
2. **Simple Differences**: Applied to VIX and TNX (already stationary indices)
3. **Missing Data**: Filled with zeros (assumes no change)

### Computational Complexity

- **Strategy 1**: O(n) - most efficient
- **Strategy 2**: O(n²) - PCA computation
- **Strategy 3**: O(nk) - K-means clustering
- **PC Algorithm**: O(p²n) where p = number of variables, n = sample size
- **Fisher-Z Test**: O(n) per test
- **KCI Test**: O(n²) per test - significantly slower

#### Performance Considerations

- **Without Lags + Fisher-Z**: Fast, handles 15+ variables easily
- **With Lags + Fisher-Z**: Moderate, recommended max 10-12 variables
- **Without Lags + KCI**: Slow, recommended max 8-10 variables
- **With Lags + KCI**: Very slow, recommended max 6-8 variables

### Key Assumptions and Limitations

#### For Financial Data
1. **Non-Gaussianity**: Financial returns are fat-tailed and skewed
   - **Solution**: Use `--test-type kci` for more robust results
2. **Temporal Causality**: Markets exhibit lagged causal relationships
   - **Solution**: Use `--use-lags` to capture time-series dependencies
3. **Sample Size**: Causal discovery requires sufficient data (typically n > 500)
   - More variables require more data for reliable discovery

#### PC Algorithm Assumptions
1. **Causal Sufficiency**: No hidden confounders
   - In markets, this is often violated (unobserved macroeconomic factors)
2. **Causal Markov Condition**: Variables are independent of non-descendants given parents
3. **Faithfulness**: All conditional independencies in data reflect graph structure

#### Practical Limitations
1. **Real Data**: Yahoo Finance API may rate-limit or block requests
2. **Computational Cost**: KCI test can be very slow for large datasets
3. **Validation**: Only available for synthetic data with known ground truth

## Project Structure

```
Causal-Discovery/
├── causal.py           # Main script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── LICENSE            # License file
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{causal_discovery_market,
  title = {Market Data Causal Discovery},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/Causal-Discovery}
}
```

## References

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
2. Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). "Kernel-based Conditional Independence Test and Application in Causal Discovery". UAI.
3. causal-learn library: https://github.com/py-why/causal-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Best Practices and Recommendations

### For Financial Time-Series Data

Based on the characteristics of financial data, here are recommended configurations:

#### 1. **Start with Validation on Synthetic Data**
```bash
python causal.py --synthetic
```
This allows you to understand algorithm performance before applying to real data.

#### 2. **Use Lagged Features for Real Causality**
```bash
python causal.py --synthetic --use-lags --max-lag 1
```
Financial causality is almost always lagged. VIX spike at t-1 causes SPY drop at t, not simultaneously.

#### 3. **Use KCI Test for Non-Gaussian Data**
```bash
python causal.py --synthetic --test-type kci
```
Financial returns are leptokurtic (fat-tailed). KCI is more robust than Fisher-Z but slower.

#### 4. **Recommended Full Pipeline**
```bash
# For exploratory analysis (fast)
python causal.py --synthetic --use-lags --max-lag 1

# For robust results (slow but accurate)
python causal.py --synthetic --use-lags --max-lag 1 --test-type kci
```

### Interpreting Results

1. **Compare Against Ground Truth**: When using synthetic data, always check validation metrics
   - Precision < 0.5: Many spurious edges, consider increasing alpha or using KCI
   - Recall < 0.5: Missing true edges, consider decreasing alpha or adding lags

2. **Look for Economic Sense**: Discovered relationships should have economic interpretation
   - VIX → Equities (volatility causes price changes) ✓
   - Random Stock A → VIX (unlikely) ✗

3. **Validate with Multiple Methods**:
   - Try both Fisher-Z and KCI
   - Try different lag values (1, 2, 3 days)
   - Compare across strategies (Proxy vs PCA vs Clustering)

4. **Be Skeptical of Spurious Edges**: Observational data often contains:
   - Common causes (hidden confounders)
   - Feedback loops
   - Selection bias

## Troubleshooting

### Issue: yfinance fails to download data

**Solution**: Use the `--synthetic` flag to use synthetic data:
```bash
python causal.py --synthetic
```

### Issue: Memory error with Strategy 2 or 3

**Solution**: Reduce the number of components/clusters or use Strategy 1:
```bash
python causal.py --strategy 1
```

### Issue: No causal relationships discovered

**Possible causes**:
- Insufficient data (increase sample size)
- Alpha too low (data cannot reject independence)
- Variables are truly independent
- Causality is lagged (try `--use-lags`)

**Solutions**:
```bash
# Try with lagged features
python causal.py --synthetic --use-lags --max-lag 1

# Try with higher alpha (more permissive)
# (requires code modification to pass alpha parameter)

# Try KCI test instead of Fisher-Z
python causal.py --synthetic --test-type kci
```

### Issue: KCI test is too slow

**Solution**: Reduce the number of variables:
```bash
# The code automatically limits variables based on test type
# But you can use Strategy 1 (Proxy) which has fewer variables
python causal.py --synthetic --strategy 1 --test-type kci
```

### Issue: Validation metrics are poor (low precision/recall)

**Possible causes**:
- Using Fisher-Z on non-Gaussian data (use KCI)
- Missing temporal relationships (use lags)
- Alpha is too high or too low

**Solutions**:
```bash
# Use robust KCI test
python causal.py --synthetic --test-type kci

# Add lagged features
python causal.py --synthetic --use-lags --max-lag 1

# Combine both
python causal.py --synthetic --use-lags --test-type kci
```

### Issue: Import errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

If `kci` import fails:
```bash
pip install --upgrade causal-learn
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This tool is for research and educational purposes. Causal relationships discovered from observational data should be interpreted carefully and validated with domain knowledge and additional analysis.
