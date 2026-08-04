"""
================================================================================
PaleoAST _core Reference Implementation Subpackage
================================================================================

This subpackage provides the single, canonical reference implementations of
core phylogenetic and morphometric algorithms. All other modules in PaleoAST
should use thin wrappers to call these implementations.

Design Principles
================================================================================

1. SINGLETON REFERENCE: Each algorithm has exactly one reference implementation
   here. Other modules MUST NOT reimplement the same algorithm.

2. WRAPPER PATTERN: Old APIs in other modules call these implementations and
   are marked as deprecated.

3. DOCUMENTATION: Every function includes:
   - Mathematical formula
   - Literature citation (author, year)
   - Parameter description
   - Return value description

4. TESTING: Every core function has:
   - Unit tests with known golden values
   - Cross-validation against R packages (ape, geiger, phytools)

Reference Implementations
================================================================================

rotation.py
    kabsch_rotation()     - SVD-based optimal rotation (Kabsch 1976)
    kabsch_rotation_3d()  - Special case for 3D rotations

vcv.py
    brownian_vcv()       - Brownian motion VCV matrix
    pagel_lambda_vcv()   - Pagel lambda-transformed VCV matrix

pic.py
    compute_pic_felsenstein() - Felsenstein 1985 PIC algorithm

Version Control
================================================================================

Any modification to these implementations MUST:
1. Be accompanied by R package verification
2. Update the corresponding unit tests with new golden values
3. Document the change in CHANGELOG.md

References
================================================================================

- Kabsch, W. (1976). A solution for the best rotation to relate two sets of
  vectors. Acta Crystallographica, A32, 922-923.
- Felsenstein, J. (1985). Phylogenies and the comparative method.
  American Naturalist, 125(1), 1-15.
- Pagel, M. (1999). Inferring the historical patterns of biological evolution.
  Nature, 401(6756), 877-884.

Author: PaleoAST Development Team
"""

from _core.rotation import kabsch_rotation, kabsch_rotation_3d
from _core.vcv import brownian_vcv, pagel_lambda_vcv
from _core.pic import compute_pic_felsenstein

__all__ = [
    "kabsch_rotation",
    "kabsch_rotation_3d",
    "brownian_vcv",
    "pagel_lambda_vcv",
    "compute_pic_felsenstein",
]
