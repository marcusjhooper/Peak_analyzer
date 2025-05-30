# ATAC-seq Visualization Tool

A web-based application for visualizing ATAC-seq data across different cell type hierarchies, with support for sequence analysis and contribution score visualization.

## Features

### Genomic Viewer
- **Differential Peaks Table**
  - Interactive table displaying differential peaks
  - Filterable and sortable columns
  - Click on a row to view the corresponding genomic region

- **BigWig Viewer**
  - Three-level visualization of ATAC-seq data:
    - Class level
    - Subclass level
    - Supertype level
  - Hierarchical filtering system:
    - Select a class to pre-populate related subclasses
    - Select a subclass to pre-populate related supertypes
    - All files remain selectable regardless of hierarchy
  - Interactive controls:
    - Zoom in/out by 1kb or 10kb
    - Default 1kb zoom out option
    - Update plot button
    - Coordinate input field

### Sequence Analysis
- **Sequence Viewer**
  - View genomic sequences for selected regions
  - Support for custom sequence input
  - Features:
    - Reverse complement display
    - GC content calculation
    - Motif highlighting
    - Line numbers
    - Formatted output

- **CRESTED Model Integration**
  - Load and run contribution score analysis
  - Batch processing support
  - Downloadable plots
  - Model selection options:
    - Pre-configured models
    - Custom model upload

## Data Organization

The app expects the following directory structure:
```
data/
├── class/          # Class-level BigWig files
├── subclass/       # Subclass-level BigWig files
├── supertype/      # Supertype-level BigWig files
├── genome/         # Genome reference files
├── model/          # Model configuration files
└── other/          # Additional data files (e.g., hierarchy information)
```

## Usage

1. **Viewing Genomic Data**
   - Select a region from the differential peaks table or enter coordinates manually
   - Use the class/subclass/supertype dropdowns to select BigWig files
   - Adjust the view using zoom controls
   - The original region is highlighted in light blue when zoomed out

2. **Sequence Analysis**
   - Enter genomic coordinates or paste a custom sequence
   - Choose analysis options (reverse complement, GC content)
   - Search for specific motifs
   - View formatted sequence output

3. **Contribution Score Analysis**
   - Load a CRESTED model
   - Select classes to analyze
   - Run contribution scores
   - Download or view the resulting plots
   - Use batch processing for multiple regions

## Requirements

- Python 3.x
- Dash
- pyBigWig
- pandas
- matplotlib
- pyfaidx
- crested
- tensorflow

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Notes

- The app uses a hierarchical organization of cell types (class → subclass → supertype)
- BigWig files should be named according to their cell type labels
- The hierarchy information is loaded from `data/other/AIT21_cldf.csv`
- Default coordinates are set to `chr19:23980545-23981046` 