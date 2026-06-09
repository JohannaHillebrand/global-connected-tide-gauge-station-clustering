# Geographically Coherent Clusterings of Tide Gauge Stations based on Time Series with Missing Data

Code for the master's thesis by Johanna Hillebrand, submitted to the Professorship for
Algorithms and Data Structures (Prof. Dr. Melanie Schmidt), Heinrich Heine University
Düsseldorf, September 2024.

Results are publicly available at https://bit.ly/connected-clustering-sea-level-data

## Overview

Tide gauge records date back to 1807 and are the only source of long-term sea level data,
but they are unevenly distributed (heavily biased toward the Northern Hemisphere), have
inconsistent time spans, and contain missing values. This repository implements connected
clustering on a coastline line graph to group geographically coherent stations with similar
sea level behavior, reducing the number of representative stations without losing critical
information.

The key contributions are:

- **Sea level line graph**: an input graph that integrates both geographic proximity and
  time series similarity between station pairs
- **Equitable connected clustering**: partitions the globe by ocean area so that cluster
  centers are distributed proportionally across regions
- **Voronoi-based equitable connected clustering**: refines the partitioning using Voronoi
  polygons generated from the tide gauge stations themselves
- **Global sea level reconstruction**: uses clustered station data combined with satellite
  altimetry (PCA / singular value decomposition) to produce a complete sea level grid

## Data

Download and place in the corresponding directories before running:

| Dataset | Directory |
|---|---|
| [PSMSL RLR monthly](https://www.psmsl.org/data/obtaining/) | `data/rlr_monthly/` |
| [Copernicus altimetry SEALEVEL_GLO_PHY_L4_MY_008_047](https://cds.climate.copernicus.eu) | `data/SEALEVEL_GLO_PHY_L4_MY_008_047/` |
| [ORAS5 ocean reanalysis](https://cds.climate.copernicus.eu) | `data/GIA/` |
| [Natural Earth 10m land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/) | `data/ne_10m_land/` |
| Natural Earth 10m ocean | `data/ocean_polygon/` |

## Setup

`fiona` requires the GDAL system library. Install it before running `poetry install`:

```bash
# Arch / Manjaro
sudo pacman -S gdal

# Ubuntu / Debian
sudo apt install libgdal-dev

# macOS
brew install gdal
```

```bash
poetry install
```

Configure parameters at the top of src/main.py (time range, output directory,
data paths), then enable the desired method by setting its flag to True:

| Flag | Method |
|---|---|
| `connected_clustering_k` | Graph clustering for fixed number of clusters k |
| `connected_clustering_radius` | Graph clustering by similarity radius |
| `calculate_voronoi_diagram` | Voronoi section clustering |
| `reconstruction` | Sea level reconstruction using clustered stations |

```bash
poetry run python -m src.main
```

Results are written to the configured out_dir (default: output/).

## Project Structure
src/
├── main.py                  # Entry point
├── clustering/              # Clustering algorithms
├── preprocessing/           # Data preprocessing
├── evaluation/              # Cluster quality evaluation
├── inner/                   # Core data structures (stations, PCA, plots)
└── start_clustering/        # Wrappers that launch each clustering method


## Citation

If you use this code, please cite the thesis:

```
Johanna Hillebrand. Geographically Coherent Clusterings of Tide Gauge Stations based on
Time Series with Missing Data. Master's Thesis, Heinrich Heine University Düsseldorf, 2024.
https://doi.org/10.5281/zenodo.20391880
```