# Market Data Causal Discovery

A Python tool for discovering causal relationships in financial market data using constraint-based causal discovery algorithms. This project implements three different data reduction strategies and applies the PC (Peter-Clark) algorithm to uncover causal parent-child relationships between market variables.

## Overview

This project addresses the challenge of performing causal discovery on high-dimensional financial data (e.g., S&P 500 with 500+ stocks). It implements three strategies to reduce dimensionality while preserving meaningful causal relationships:

1. **Strategy 1: Proxy (Aggregation)** - Uses sector ETFs and market indices as proxies
2. **Strategy 2: PCA (Factor)** - Reduces dimensions using Principal Component Analysis
3. **Strategy 3: Clustering (Representative)** - Selects representative stocks via K-means clustering

After dimensionality reduction, the PC algorithm is applied to discover causal relationships and identify causal parents for each variable.

## Features

- **Flexible Data Sources**: Works with both real market data (via yfinance) and synthetic data
- **Multiple Dimensionality Reduction Strategies**: Choose the approach that best fits your analysis needs
- **Causal Discovery**: Implements the PC algorithm with Fisher's Z test for conditional independence
- **Synthetic Data Generation**: Includes a synthetic market data generator with known causal structure for testing
- **Comprehensive Output**: Displays discovered causal relationships in an easy-to-read format

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

Options:
- `--synthetic`: Use synthetic data instead of real market data
- `--strategy {1,2,3}`: Choose dimensionality reduction strategy (default: 1)
  - 1: Proxy/Aggregation approach
  - 2: PCA/Factor approach
  - 3: Clustering/Representative approach
- `--no-causal`: Skip causal discovery analysis (only perform data reduction)

### Examples

```bash
# Run with synthetic data using Strategy 1 (Proxy)
python causal.py --synthetic --strategy 1

# Run with real data using Strategy 2 (PCA)
python causal.py --strategy 2

# Run with synthetic data but skip causal discovery
python causal.py --synthetic --no-causal
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

- **Conditional Independence Test**: Fisher's Z test
- **Significance Level**: α = 0.05 (default)
- **Output**: Directed acyclic graph (DAG) showing causal relationships

### Interpreting Results

- **Causal Parents**: Variables that directly influence another variable
- **Root Nodes**: Variables with no causal parents (exogenous variables)
- **Direction**: A → B means "A causes B" or "A is a causal parent of B"

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

### Limitations

1. **Sample Size**: Causal discovery requires sufficient data (typically n > 500)
2. **Assumptions**: PC algorithm assumes:
   - Causal sufficiency (no hidden confounders)
   - Causal Markov condition
   - Faithfulness condition
3. **Real Data**: Yahoo Finance API may rate-limit or block requests

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

### Issue: Import errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This tool is for research and educational purposes. Causal relationships discovered from observational data should be interpreted carefully and validated with domain knowledge and additional analysis.
