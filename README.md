# PaleoAST - Paleontological Advanced Statistical Toolkit

A comprehensive Python-based statistical toolkit for paleontological data analysis, featuring multivariate statistics, geometric morphometrics, biodiversity analysis, and time series methods.

## Features

### Statistical Analysis
- **Principal Component Analysis (PCA)** - Dimensionality reduction with variance-covariance and correlation-based methods
- **Principal Coordinate Analysis (PCoA)** - Metric MDS for non-Euclidean distance matrices
- **Non-metric Multidimensional Scaling (NMDS)** - Rank-based ordination with SMACOF optimization
- **ANOSIM** - Analysis of Similarities for group comparisons
- **PERMANOVA** - Permutational MANOVA for multivariate group testing

### Diversity Analysis
- **Alpha Diversity Indices** - Shannon, Simpson, Fisher's Alpha, Margalef, Pielou's Evenness
- **Chao-1 Estimation** - Species richness extrapolation
- **Rarefaction Curves** - Sample-size standardization

### Geometric Morphometrics
- **Generalized Procrustes Analysis (GPA)** - Landmark superimposition
- **Thin-Plate Spline (TPS)** - Shape deformation visualization
- **Relative Warps Analysis** - PCA in shape space

### Time Series Analysis
- **Lomb-Scargle Periodogram** - Spectral analysis for unevenly sampled data
- **Peak Detection** - Significant periodic signal identification

## Installation

```bash
# Clone the repository
git clone https://github.com/paleoast/PaleoAST.git
cd PaleoAST

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

## Requirements

- Python 3.10+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0

## Quick Start

```python
import numpy as np
from statistics.pca import PCAAnalyzer
from ecology.diversity import compute_diversity_indices

# PCA Analysis
data = np.random.randn(30, 8) * 10 + 50
pca = PCAAnalyzer()
result = pca.analyze(data, n_components=3)

# Diversity Analysis
abundances = np.array([45, 23, 15, 12, 8, 5, 3, 2, 1, 1])
div_result = compute_diversity_indices(abundances, "Sample_A")
```

## Project Structure

```
PaleoAST/
├── config/          - Configuration and constants
├── models/         - Data models and state management
├── statistics/      - Statistical analysis engines
├── morphometrics/   - Geometric morphometrics
├── ecology/         - Biodiversity analysis
├── stratigraphy/    - Time series and spectral analysis
├── visualization/    - Publication-quality plotting
├── controllers/     - MVC controllers
└── main.py          - Application entry point
```

## Documentation

See [ARCHITECTURE_BLUEPRINT.md](ARCHITECTURE_BLUEPRINT.md) for detailed mathematical foundations and API documentation.

## License

MIT License

## Citation

If you use PaleoAST in your research, please cite:

```
PaleoAST Development Team (2024). PaleoAST: Paleontological Advanced 
Statistical Toolkit. Version 1.0.0. https://github.com/paleoast/PaleoAST
```
