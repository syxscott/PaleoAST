# _core Reference Implementation Subpackage

## Purpose

The `_core` subpackage provides **single, canonical reference implementations** of
core phylogenetic and morphometric algorithms used throughout PaleoAST. This
eliminates the current problem of multiple divergent implementations of the same
algorithm across different modules.

## Design Principles

### 1. Single Reference Implementation

Each algorithm exists in exactly ONE place: here in `_core/`. Other modules
MUST NOT reimplement algorithms that already exist here.

**Current redundant implementations being consolidated:**
| Algorithm | Current Locations | Reference Location |
|-----------|------------------|-------------------|
| Kabsch SVD rotation | morpho3d/quaternion.py, morphometrics/gpa.py | _core/rotation.py |
| Brownian VCV | phylogenetics/signal.py, statistics/pcm.py | _core/vcv.py |
| PIC (Felsenstein) | phylogenetics/pic.py, statistics/pcm.py | _core/pic.py |

### 2. Wrapper Pattern for Backward Compatibility

Old APIs in other modules are preserved as thin wrappers that call these
implementations. Wrappers are marked with deprecation warnings:

```python
def deprecated_rotation(X, Y):
    """Deprecated: use _core.rotation.kabsch_rotation() instead."""
    import warnings
    warnings.warn(
        "This function is deprecated. Use _core.rotation.kabsch_rotation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return kabsch_rotation(X, Y)
```

### 3. Documentation Requirements

Every function in `_core/` MUST include:

- **Mathematical formula** in LaTeX format
- **Literature citation** (author, year, journal)
- **Parameter descriptions** with types
- **Return value descriptions** with types
- **R package verification** notes

### 4. Testing Requirements

Every core function MUST have:

1. **Unit tests** with known golden values
2. **R package cross-validation** tests

Golden values must be obtained by running equivalent R code:

```python
# Python test
def test_kabsch_rotation():
    from _core.rotation import kabsch_rotation
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[0.0, 1.0], [-1.0, 0.0]])
    R = kabsch_rotation(X, Y)
    # Compare with R: procSym() in Morpho package
    assert np.allclose(R, expected_from_R)

# R verification code (documented, not run in Python test)
# library(Morpho)
# X <- matrix(c(1,0,0,1), nrow=2, byrow=TRUE)
# Y <- matrix(c(0,-1,1,0), nrow=2, byrow=TRUE)
# procSym(X, Y)$rot   # Should give same R
```

## Module Structure

```
_core/
    __init__.py      # Package exports
    rotation.py      # Kabsch SVD rotation (Kabsch 1976)
    vcv.py           # Brownian motion VCV (Pagel 1999)
    pic.py           # PIC Felsenstein 1985
    README.md        # This file
```

## API Reference

### rotation.py

```python
def kabsch_rotation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: find R minimizing ||Y - X @ R||_F^2.

    Reference: Kabsch, W. (1976). Acta Crystallographica, A32, 922-923.
    """

def kabsch_rotation_3d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Specialized 3D case of kabsch_rotation()."""
```

### vcv.py

```python
def brownian_vcv(tree, lambda_param: float = 1.0) -> tuple[list[str], np.ndarray]:
    """
    Brownian motion variance-covariance matrix.

    V[i,i] = dist(root, tip_i)
    V[i,j] = dist(root, LCA(i,j))

    Reference: Pagel, M. (1999). Nature, 401(6756), 877-884.
    """

def pagel_lambda_vcv(tree, lambda_param: float) -> tuple[list[str], np.ndarray]:
    """Pagel lambda-transformed VCV matrix."""
```

### pic.py

```python
def compute_pic_felsenstein(
    tree,
    traits: dict[str, float],
    root_variance: float = 0.0,
) -> tuple[list[float], list[tuple[str, str]]]:
    """
    Felsenstein 1985 Phylogenetic Independent Contrasts.

    contrast = (x_i - x_j) / sqrt(v_i + v_j)

    Reference: Felsenstein, J. (1985). American Naturalist, 125(1), 1-15.
    """
```

## Version Control Policy

Any modification to implementations in `_core/` MUST:

1. **Include R package verification** - Run equivalent R code and document results
2. **Update golden values in tests** - If implementation changes, test golden values must be updated
3. **Document in CHANGELOG.md** - Note the change and its rationale

Example CHANGELOG entry:

```markdown
## [Unreleased]

### Changed
- `_core/pic.py`: Fixed variance accumulation formula

  Previous: v_i = sum of branch lengths from parent
  New: v_i = sum of branch lengths from ROOT (matches Felsenstein 1985)

  Verified against R `pic()` in ape 5.7.1:
  - Test tree: "(A:1,B:1,C:1)D:1;"
  - Test traits: A=2.0, B=4.0, C=3.0
  - Before: contrast = -1.414 (incorrect)
  - After: contrast = -2.000 (matches R)
```

## Migration Guide

### For Module Developers

**Before** (in morpho3d/quaternion.py):
```python
class RotationMatrix:
    @staticmethod
    def from_svd(X, Y):
        # Internal SVD implementation
        ...
```

**After**:
```python
class RotationMatrix:
    @staticmethod
    def from_svd(X, Y):
        import warnings
        warnings.warn(
            "Use _core.rotation.kabsch_rotation() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return _core.kabsch_rotation(X, Y)
```

## References

- Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors. Acta Crystallographica, A32, 922-923.
- Felsenstein, J. (1985). Phylogenies and the comparative method. American Naturalist, 125(1), 1-15.
- Pagel, M. (1999). Inferring the historical patterns of biological evolution. Nature, 401(6756), 877-884.
- Pagel, M. (1992). A method for the analysis of comparative data. Journal of Theoretical Biology, 156(4), 431-442.
