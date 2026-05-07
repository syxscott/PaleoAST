# =============================================================================
# FILE: main.py
# =============================================================================
"""
PaleoAST - Paleontological Advanced Statistical Toolkit

A comprehensive desktop application for paleontological data analysis including:
    - Multivariate statistics (PCA, PCoA, NMDS)
    - Group comparison tests (ANOSIM, PERMANOVA)
    - Diversity analysis (Shannon, Simpson, rarefaction)
    - Geometric morphometrics (GPA, TPS, Relative Warps)
    - Time series analysis (Lomb-Scargle periodogram)

Author: PaleoAST Development Team
Version: 1.0.0
"""

import sys
import numpy as np


def print_banner() -> None:
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗ ██╗███████╗███████╗     ██████╗ ██████╗ ███╗   ██╗   ║
    ║  ██╔════╝ ██║██╔════╝██╔════╝    ██╔════╝██╔═══██╗████╗  ██║   ║
    ║  ██║      ██║███████╗█████╗      ██║     ██║   ██║██╔██╗ ██║   ║
    ║  ██║      ██║╚════██║██╔══╝      ██║     ██║   ██║██║╚██╗██║   ║
    ║  ╚██████╗ ██║███████║███████╗    ╚██████╗╚██████╔╝██║ ╚████║   ║
    ║   ╚═════╝ ╚═╝╚══════╝╚══════╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ║
    ║                                                               ║
    ║   Paleontological Advanced Statistical Toolkit                 ║
    ║   Version 1.0.0                                               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_info() -> None:
    """Print system information."""
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print()


def run_demo() -> None:
    """
    Run a demonstration of PaleoAST capabilities.
    
    This demonstrates the core statistical functionality without GUI.
    """
    print("\n" + "=" * 60)
    print("PaleoAST Demonstration")
    print("=" * 60 + "\n")
    
    # Import analysis modules
    from statistics.pca import PCAAnalyzer
    from statistics.distance_metrics import compute_distance_matrix
    from ecology.diversity import compute_diversity_indices
    from morphometrics.gpa import GPAAnalyzer
    from stratigraphy.spectral_analysis import SpectralAnalyzer
    
    print("1. Principal Component Analysis (PCA)")
    print("-" * 40)
    
    # Generate sample paleontological data
    np.random.seed(42)
    n_samples = 30
    n_variables = 8
    
    # Simulate fossil measurements
    data = np.random.randn(n_samples, n_variables) * 10 + 50
    
    # Run PCA
    pca = PCAAnalyzer()
    result = pca.analyze(data, n_components=3, method='correlation')
    
    print(f"   Samples: {result.scores.shape[0]}")
    print(f"   Variables: {n_variables}")
    print(f"   Components extracted: {result.n_components}")
    print(f"   Variance explained (PC1): {result.explained_variance[0]:.1f}%")
    print(f"   Variance explained (PC2): {result.explained_variance[1]:.1f}%")
    print(f"   Variance explained (PC3): {result.explained_variance[2]:.1f}%")
    print()
    
    print("2. Distance Metrics")
    print("-" * 40)
    
    # Compute various distance matrices
    for metric in ['euclidean', 'bray_curtis', 'jaccard']:
        dm = compute_distance_matrix(data, metric=metric)
        print(f"   {metric.capitalize()} distance matrix computed: {dm.matrix.shape}")
    
    print()
    
    print("3. Diversity Analysis")
    print("-" * 40)
    
    # Simulate species abundance data
    abundances = np.array([45, 23, 15, 12, 8, 5, 3, 2, 1, 1])
    
    div_result = compute_diversity_indices(abundances, "Fossil_Site_A")
    
    print(f"   Sample: {div_result.sample_name}")
    print(f"   Taxa Richness (S): {div_result.taxa_count}")
    print(f"   Total Individuals (N): {div_result.individuals}")
    
    if 'shannon' in div_result.indices:
        print(f"   Shannon Index (H'): {div_result.indices['shannon'].value:.4f}")
    
    if 'simpson' in div_result.indices:
        print(f"   Simpson Index (1-D): {div_result.indices['simpson'].value:.4f}")
    
    if 'fisher_alpha' in div_result.indices:
        print(f"   Fisher's Alpha: {div_result.indices['fisher_alpha'].value:.4f}")
    
    print()
    
    print("4. Generalized Procrustes Analysis (GPA)")
    print("-" * 40)
    
    # Simulate landmark data
    n_specimens = 15
    n_landmarks = 8
    configurations = np.zeros((n_specimens, n_landmarks, 2))
    
    for i in range(n_specimens):
        # Base configuration
        base = np.random.randn(n_landmarks, 2) * 5
        # Add variation (rotation, translation, scaling)
        angle = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(0.8, 1.2)
        translation = np.random.randn(2) * 3
        
        # Apply transformation
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        configurations[i] = scale * (base @ R.T) + translation
    
    # Run GPA
    gpa = GPAAnalyzer()
    gpa_result = gpa.analyze(configurations)
    
    print(f"   Specimens analyzed: {gpa_result.aligned_configurations.shape[0]}")
    print(f"   Landmarks: {gpa_result.aligned_configurations.shape[1]}")
    print(f"   Iterations to convergence: {gpa_result.n_iterations}")
    print(f"   Converged: {gpa_result.converged}")
    
    print()
    
    print("5. Spectral Analysis (Lomb-Scargle)")
    print("-" * 40)
    
    # Simulate cyclostratigraphy data
    n_points = 200
    time = np.sort(np.random.uniform(0, 100, n_points))
    
    # Signal with 405 ka and 100 ka cycles (Milankovitch)
    signal = (3 * np.sin(2 * np.pi * time / 40) + 
              2 * np.sin(2 * np.pi * time / 100) +
              1.5 * np.sin(2 * np.pi * time / 20))
    noise = np.random.randn(n_points) * 0.5
    values = signal + noise
    
    # Run spectral analysis
    spectral = SpectralAnalyzer()
    spec_result = spectral.analyze(time, values, n_frequencies=500)
    
    print(f"   Data points: {len(time)}")
    print(f"   Frequencies analyzed: {len(spec_result.frequencies)}")
    
    if spec_result.peak_period:
        print(f"   Dominant period: {spec_result.peak_period:.2f}")
    
    print()
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def main() -> None:
    """
    Main entry point for PaleoAST.
    
    Usage:
        python main.py           - Run demo
        python main.py --gui     - Launch GUI (future)
        python main.py --help    - Show help
    """
    print_banner()
    print_info()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--gui':
            print("GUI mode not yet implemented.")
            print("Running in demo mode instead...\n")
            run_demo()
        elif sys.argv[1] == '--help':
            print("""
PaleoAST - Paleontological Advanced Statistical Toolkit

Usage:
    python main.py           Run demonstration
    python main.py --gui     Launch GUI (future)
    python main.py --help    Show this help

For more information, visit: https://github.com/paleoast/PaleoAST
            """)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Run 'python main.py --help' for usage information.")
    else:
        run_demo()


if __name__ == '__main__':
    main()
