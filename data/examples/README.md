# PaleoAST Example Datasets

This directory contains example datasets for testing, demonstration, and teaching purposes in the PaleoAST paleontological analysis platform.

## Dataset Overview

| Dataset | Format | Specimens/Tips | Landmarks/Variables | Primary Use |
|---------|--------|---------------|---------------------|--------------|
| moth_wings | TPS | 30 | 8 landmarks | GPA, EFA, TPS, shape visualization |
| community_abundance | CSV | 20 sites | 28 species | NMDS, PERMANOVA, ANOSIM, diversity |
| primate_tree | NEWICK | 13 tips | - | PIC, phylogenetic signal |
| primate_traits | CSV | 13 species | 8 traits | Phylogenetic comparative methods |

## Dataset 1: Moth Wing Landmarks (moth_wings.tps)

### Description
2D landmark data from mosquito wings (Culex pipiens), based on the classic morphometric dataset from Rohlf (1990). This dataset is widely used in geometric morphometrics for testing Generalized Procrustes Analysis (GPA), Elliptic Fourier Analysis (EFA), and Thin Plate Spline (TPS) interpolation.

### Source
Rohlf, F.J. (1990). Morphometrics. Annual Review of Ecology and Systematics, 21, 299-316.

### Format
TPS (Thin Plate Spline) format - a standard format for landmark data in morphometrics.

### Fields
- `ID`: Specimen identifier (Mosquito_001 to Mosquito_030)
- `LM`: Number of landmarks per specimen (8)
- `DIM`: Dimensionality (2 for 2D)
- `SCALE`: Scale factor (1.0)
- Landmark coordinates: 8 (x,y) pairs per specimen

### Usage
```python
from parsers.tps_parser import TPSParser

parser = TPSParser()
tps_file = parser.parse("data/examples/moth_wings.tps")
landmarks = tps_file.to_matrix()  # Shape: (30, 16)
```

### Citation
If using this dataset, please cite:
```
Rohlf, F.J. (1990). Morphometrics. Annual Review of Ecology and
Systematics, 21, 299-316.
```

---

## Dataset 2: Community Abundance (community_abundance.csv)

### Description
Species abundance matrix simulating plant community data from different habitat types. Based on the structure of classic vegan package datasets (varespec, dune). Contains 20 sites across 5 habitat groups (Grassland, Woodland, Meadow, Wetland, Disturbed) and 28 plant species.

### Source
Simulated data based on:
- Jongman, R.H.G., ter Braak, C.J.F., & van Tongeren, O.F.R. (1995). Data Analysis in Community and Landscape Ecology. Cambridge University Press.
- The vegan R package (varespec and dune datasets)

### Format
CSV with site rows and species abundance columns.

### Fields
| Column | Description |
|--------|-------------|
| site | Site identifier (Site_1 to Site_20) |
| group | Habitat type (Grassland, Woodland, Meadow, Wetland, Disturbed) |
| Achimill-B rachpodi | Species abundance values (0.00-7.89) |

### Usage
```python
import pandas as pd

df = pd.read_csv("data/examples/community_abundance.csv")
# Get species matrix for analysis
species_matrix = df.iloc[:, 2:].values
groups = df['group'].values
```

### Citation
If using this dataset, please cite:
```
Jongman, R.H.G., ter Braak, C.J.F., & van Tongeren, O.F.R. (1995).
Data Analysis in Community and Landscape Ecology. Cambridge University Press.
```

---

## Dataset 3: Primate Phylogeny (primate_tree.nwk)

### Description
Composite primate phylogeny with 13 tips representing major primate groups. Based on the phylogenetic framework from Springer et al. (2012) with branch lengths in millions of years. Includes a polytomy at the colobine/cercopithecine divergence for testing multi-furcation handling.

### Source
Springer, M.S., et al. (2012). Macroevolutionary dynamics and historical biogeography of primate diversification inferred from a species supermatrix. PLoS ONE 7(11): e49521.

### Format
NEWICK format - standard phylogenetic tree format.

### Fields
- Tip labels: 13 primate species names
- Internal nodes: Named at major clades
- Branch lengths: Millions of years

### Usage
```python
from phylogenetics.tree import PhyloTree, PhyloNode

tree = PhyloTree.from_newick(newick_string)
leaf_names = tree.leaf_names  # List of 13 species
```

### Citation
If using this dataset, please cite:
```
Springer, M.S., et al. (2012). Macroevolutionary dynamics and historical
biogeography of primate diversification inferred from a species supermatrix.
PLoS ONE 7(11): e49521.
```

---

## Dataset 4: Primate Traits (primate_traits.csv)

### Description
Continuous and categorical trait data for 13 primate species matching the phylogenetic tree tips. Includes morphological traits (body mass, brain volume), behavioral traits (home range, social group size), and ecological traits (activity cycle).

### Source
Compiled from:
- Smithsonian National Museum of Natural History. (2012). Mammal Species of the World.
- Springer et al. (2012). PLoS ONE 7(11): e49521.

### Format
CSV with species rows and trait columns.

### Fields
| Column | Type | Description |
|--------|------|-------------|
| species | string | Species name (matches tree tips) |
| group | string | Higher taxonomic group |
| body_mass_g | float | Body mass in grams |
| home_range_km2 | float | Home range size in km² |
| litter_size | float | Average number of offspring |
| social_group_size | float | Mean group size |
| day_range_km | float | Daily travel distance in km |
| brain_vol_cm3 | float | Brain volume in cm³ |
| pelage_darkness | float | Pelage darkness score (0-1) |
| activity_cycle | float | Diurnal (1.0) vs nocturnal (0.0) |

### Usage
```python
import pandas as pd

traits = pd.read_csv("data/examples/primate_traits.csv")
# Merge with phylogenetic tree for comparative analyses
```

### Citation
If using this dataset, please cite:
```
Springer, M.S., et al. (2012). Macroevolutionary dynamics and historical
biogeography of primate diversification inferred from a species supermatrix.
PLoS ONE 7(11): e49521.

Smithsonian National Museum of Natural History. (2012). Mammal Species of the World.
```

---

## License

All example datasets are provided for educational and research purposes. The original data sources are cited above. When using these datasets in published work, please cite the original sources as indicated.

## Data Loading Utilities

The `data` package provides convenient loaders for these datasets:

```python
from data import load_moth_wings, load_community, load_primate_tree

# Load landmark data
landmarks, specimen_ids = load_moth_wings()

# Load community data
community_df = load_community()

# Load phylogeny
tree = load_primate_tree()
```

See `data/loader.py` for full documentation.
