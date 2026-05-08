"""Full regression test for PaleoAST after bug fixes."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')

passed = 0
failed = 0
errors = []

def test(name, func):
    global passed, failed
    try:
        func()
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL: {name} -> {e}")

# ============================================================
# 1. Module imports
# ============================================================
print("=== Module Imports ===")
def t1():
    from phylogenetics import PhyloTree, PhyloNode, FitchAlgorithm, HeuristicSearch
    from phylogenetics import NNIOperation, TBROperation, StrictConsensusTree, UPGMA, NeighborJoining
test("phylogenetics import", t1)

def t2():
    from parsers import BaseLexer, LexerError, NexusLexer, NexusTokenType
    from parsers import NewickParser, NewickTree, TreeNode, BinaryCache, BinaryCacheHeader, ChunkType
test("parsers import", t2)

def t3():
    from hpc import ProcessPool, Task, TaskScheduler
test("hpc import", t3)

def t4():
    from reporting import ReportBuilder, LatexCompiler
test("reporting import", t4)

def t5():
    from state_machine import NFA, DFA, RegexCompiler, Token, TokenType, LexerTokenizer
test("state_machine import", t5)

# ============================================================
# 2. Statistics module
# ============================================================
print()
print("=== Statistics Module ===")
def t_pca():
    from statistics.pca import PCAAnalyzer
    data = np.random.randn(30, 8) * 10 + 50
    result = PCAAnalyzer().analyze(data, n_components=3)
    assert result.scores.shape == (30, 3)
test("PCA", t_pca)

def t_pcoa():
    from statistics.pcoa import PCoAAnalyzer
    data = np.random.randn(20, 5)
    dm = squareform(pdist(data))
    result = PCoAAnalyzer().analyze(dm)
    assert result.coordinates.shape[0] == 20
test("PCoA", t_pcoa)

def t_nmds():
    from statistics.nmds import NMDSAnalyzer
    data = np.random.randn(15, 4)
    dm = squareform(pdist(data))
    result = NMDSAnalyzer().analyze(dm, n_dimensions=2, max_iterations=200)
    assert result.coordinates.shape == (15, 2)
    assert result.stress >= 0
test("NMDS", t_nmds)

def t_anosim():
    from statistics.anosim import ANOSIMAnalyzer
    data = np.random.randn(20, 5)
    dm = squareform(pdist(data))
    groups = [0]*10 + [1]*10
    result = ANOSIMAnalyzer().analyze(dm, groups)
    assert hasattr(result, "statistic")
test("ANOSIM", t_anosim)

def t_permanova():
    from statistics.permanova import PERMANOVAAnalyzer
    g1 = np.random.randn(10, 5) + 5
    g2 = np.random.randn(10, 5) - 5
    data = np.vstack([g1, g2])
    dm = squareform(pdist(data))
    groups = [0]*10 + [1]*10
    result = PERMANOVAAnalyzer().analyze(dm, groups)
    assert result.f_statistic >= 0, f"F-statistic={result.f_statistic} should be >= 0"
    assert 0 <= result.p_value <= 1
test("PERMANOVA", t_permanova)

def t_dist():
    from statistics.distance_metrics import compute_distance_matrix
    data = np.random.randint(0, 10, (10, 5)).astype(float)
    result = compute_distance_matrix(data, metric="euclidean")
    assert result.matrix.shape == (10, 10)
test("Distance metrics", t_dist)

# ============================================================
# 3. Ecology module
# ============================================================
print()
print("=== Ecology Module ===")
def t_div():
    from ecology.diversity import compute_diversity_indices
    abundances = np.array([45, 23, 15, 12, 8, 5, 3, 2, 1, 1])
    result = compute_diversity_indices(abundances, "Test")
    assert result.taxa_count == 10
    assert "shannon" in result.indices
test("Diversity", t_div)

def t_raref():
    from ecology.rarefaction import compute_rarefaction
    abundances = np.array([45, 23, 15, 12, 8, 5, 3, 2, 1, 1])
    result = compute_rarefaction(abundances, "Test", n_points=10)
    assert len(result.sample_sizes) > 0
test("Rarefaction", t_raref)

# ============================================================
# 4. Morphometrics module
# ============================================================
print()
print("=== Morphometrics Module ===")
def t_gpa():
    from morphometrics.gpa import GPAAnalyzer
    configs = np.random.randn(5, 8, 2) * 2 + 10
    result = GPAAnalyzer().analyze(configs)
    assert result.aligned_configurations.shape == (5, 8, 2)
test("GPA", t_gpa)

def t_tps():
    from morphometrics.tps import TPSAnalyzer
    source = np.random.randn(8, 2) * 2 + 10
    target = source + np.random.randn(8, 2) * 0.5
    result = TPSAnalyzer().analyze(source, target)
    assert result.bending_energy >= 0, f"Bending energy={result.bending_energy} should be >= 0"
test("TPS bending energy", t_tps)

def t_rw():
    from morphometrics.gpa import GPAAnalyzer
    from morphometrics.relative_warps import RelativeWarpsAnalyzer
    configs = np.random.randn(10, 6, 2) * 2 + 10
    gpa_result = GPAAnalyzer().analyze(configs)
    result = RelativeWarpsAnalyzer().analyze(gpa_result.aligned_configurations, n_components=3)
    assert result.relative_warps.shape == (10, 3)
test("Relative Warps", t_rw)

# ============================================================
# 5. Stratigraphy module
# ============================================================
print()
print("=== Stratigraphy Module ===")
def t_spec():
    from stratigraphy.spectral_analysis import SpectralAnalyzer
    t = np.sort(np.random.uniform(0, 100, 200))
    signal = 3 * np.sin(2 * np.pi * t / 10) + np.random.randn(200) * 0.5
    result = SpectralAnalyzer().analyze(t, signal)
    assert len(result.frequencies) > 0
test("Spectral analysis", t_spec)

# ============================================================
# 6. Phylogenetics module
# ============================================================
print()
print("=== Phylogenetics Module ===")
def t_tree():
    from phylogenetics.tree import PhyloNode, PhyloTree
    root = PhyloNode(name="root")
    A = PhyloNode(name="A", branch_length=0.1)
    B = PhyloNode(name="B", branch_length=0.2)
    root.add_child(A)
    root.add_child(B)
    tree = PhyloTree(root)
    newick = tree.to_newick()
    assert "A" in newick and "B" in newick
test("PhyloTree", t_tree)

def t_fitch():
    from phylogenetics.fitch import FitchAlgorithm
    from phylogenetics.tree import PhyloNode, PhyloTree
    root = PhyloNode(name="root")
    n1 = PhyloNode(name="n1")
    A = PhyloNode(name="A")
    B = PhyloNode(name="B")
    C = PhyloNode(name="C")
    n1.add_child(A)
    n1.add_child(B)
    root.add_child(n1)
    root.add_child(C)
    tree = PhyloTree(root)
    states = {"A": "a", "B": "a", "C": "b"}
    fitch = FitchAlgorithm()
    result = fitch.run(tree, states)
    assert result.parsimony_score >= 0
test("Fitch", t_fitch)

def t_upgma():
    from phylogenetics.distance_methods import UPGMA, DistanceMatrix
    dm = np.array([[0,5,9,9,8],[5,0,10,10,9],[9,10,0,8,7],[9,10,8,0,3],[8,9,7,3,0]], dtype=float)
    labels = ["A","B","C","D","E"]
    dist = DistanceMatrix.from_array(dm, labels)
    tree = UPGMA().build(dist)
    assert tree is not None
test("UPGMA", t_upgma)

# ============================================================
# 7. Macroevolution module
# ============================================================
print()
print("=== Macroevolution Module ===")
def t_cohort():
    from macroevolution.cohort import CohortSurvivorshipAnalysis
    cohort = CohortSurvivorshipAnalysis()
    fossil_records = [(5.0, 3.0), (4.0, 2.0), (6.0, 1.0), (3.5, 0.5)]
    intervals = [(6.0, 5.0), (5.0, 4.0), (4.0, 3.0), (3.0, 2.0), (2.0, 1.0), (1.0, 0.0)]
    result = cohort.analyze(fossil_records, intervals)
    assert len(result.survival_rates) == 6
test("Cohort survivorship", t_cohort)

def t_fbd():
    from macroevolution.fbd import FossilizedBirthDeathProcess
    fbd = FossilizedBirthDeathProcess(lambda_=0.1, mu=0.05, psi=0.02)
    sp = fbd.survival_probability(10.0)
    assert 0 <= sp <= 1
test("FBD", t_fbd)

# ============================================================
# 8. Morpho3D module
# ============================================================
print()
print("=== Morpho3D Module ===")
def t_quat():
    from morpho3d.quaternion import Quaternion
    q1 = Quaternion(1, 0, 0, 0)
    q2 = Quaternion(0, 1, 0, 0)
    q3 = q1 * q2
    assert q3 is not None
test("Quaternion", t_quat)

def t_gpa3d():
    from morpho3d.gpa3d import GPA3D
    configs = list(np.random.randn(5, 10, 3) * 2 + 10)
    result = GPA3D().analyze(configs)
    assert result.aligned_configurations.shape == (5, 10, 3)
test("GPA3D", t_gpa3d)

def t_tps3d():
    from morpho3d.tps3d import TPS3D
    source = np.random.randn(10, 3) * 2 + 10
    target = source + np.random.randn(10, 3) * 0.3
    tps = TPS3D()
    result = tps.analyze(source, target)
    assert result.bending_energy >= 0, f"Bending energy={result.bending_energy} should be >= 0"
test("TPS3D", t_tps3d)

# ============================================================
# 9. Visualization module
# ============================================================
print()
print("=== Visualization Module ===")
def t_pca_plot():
    from visualization.pca_plot import PCAPlotter
    from statistics.pca import PCAAnalyzer
    data = np.random.randn(20, 5)
    result = PCAAnalyzer().analyze(data, n_components=3)
    fig = PCAPlotter().plot_scores(result, groups=[0]*10+[1]*10)
    assert fig is not None
test("PCA Plotter", t_pca_plot)

def t_div_plot():
    from visualization.diversity_plot import DiversityPlotter
    from ecology.diversity import compute_diversity_indices
    abundances = np.array([45, 23, 15, 12, 8, 5, 3, 2, 1, 1])
    result = compute_diversity_indices(abundances, "Test")
    fig = DiversityPlotter().plot_diversity_comparison([result], index="shannon")
    assert fig is not None
test("Diversity Plotter", t_div_plot)

# ============================================================
# 10. Utils module
# ============================================================
print()
print("=== Utils Module ===")
def t_valid():
    from utils.validators import validate_data_array, check_missing_values
    data = np.random.randn(10, 5)
    validate_data_array(data)
    assert check_missing_values(data)["total_nan"] == 0
test("Validators", t_valid)

def t_matops():
    from utils.matrix_ops import ensure_matrix, center_matrix, pairwise_distances
    data = np.random.randn(10, 5)
    m = ensure_matrix(data)
    c = center_matrix(m)
    d = pairwise_distances(m)
    assert c.shape == (10, 5) and d.shape == (10, 10)
test("Matrix ops", t_matops)

# ============================================================
# 11. Models module
# ============================================================
print()
print("=== Models Module ===")
def t_dm():
    from models.data_matrix import DataMatrix
    data = np.random.randn(10, 5)
    dm = DataMatrix(data, row_labels=[f"S{i}" for i in range(10)], col_labels=[f"V{i}" for i in range(5)])
    assert dm.shape == (10, 5)
test("DataMatrix", t_dm)

# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print()
    print("Failed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")
print("=" * 60)
