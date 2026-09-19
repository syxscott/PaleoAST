# Golden test data

Real published morphometric datasets converted from the R package
**geomorph** (`data/*.rda`, MIT license) into numpy `.npz` fixtures for
regression/golden tests (`tests/golden/`).

## hummingbirds.npz
- Source: `geomorph::hummingbirds` — Berns & Adams (2010), beak *Heron Island*
  dataset of two *Archilochus* hummingbird species (25 specimens).
- `configurations` `(25, 44, 2)` float64: 44 landmark/semilandmark points per
  specimen in x,y order (transposed from the R `land` array `(n, 2, k)`;
  0-based specimen and point indexing preserved).
- `curve_triples_1based` `(15, 3)` int64: geomorph `curvepts` rows
  `(before, slider, after)` with **1-based** point indices, as published.

## scallops.npz
- Source: `geomorph::scallops` — Serb et al. (2011), *Argopecten* bivalve
  outlines, one representative specimen per species (5 specimens).
- `configurations` `(5, 46, 3)` float64: 46 3D points (transposed from the R
  `coorddata` array `(k, 3, n)`).
- `curvslide_triples_1based` `(11, 3)` int64: geomorph `curvslide` triples
  (1-based).
- `surface_slide_1based` `(30,)` int64: geomorph `surfslide` — points
  17–46 (1-based) form the sliding outline curve.
- `species` `(5,)` int64: species label per specimen (1-based).
- `land_pairs_1based` `(19, 2)` int64: geomorph `land.pairs` (1-based).

All coordinate values are stored unchanged (float64). Conversion script kept
out of tree; regenerate with `rdata.read_rda` + transpose as documented above.
