# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial analysis course repository for a graduate program at Universidad Nacional de Colombia (Facultad de Minas, Medellín). The course covers remote sensing, spatial data processing, GIS tools, and statistical modeling using Python and R.

The repository serves a dual purpose:
- A GitHub Pages course website (Jekyll, `index.md` as homepage)
- A Reveal.js HTML presentation (`html/` directory)
- Jupyter notebooks and R Markdown notebooks for course labs

## Commands

### HTML Presentation (Reveal.js in `html/`)
```bash
cd html
npm start        # Start dev server (gulp serve)
npm run build    # Build production assets
npm test         # Run QUnit tests
```

### Python Environment
```bash
conda env create -f Notebooks/Python/environment.txt
conda activate carto
jupyter lab
```

The conda environment (`carto`) installs: jupyterlab, pandas, numpy, geopandas, rasterio, pykrige, statsmodels, xarray, pysal, contextily, osmnx, earthengine-api, skgstat, seaborn.

### R Notebooks
Open `.Rmd` files in RStudio or VS Code and knit them individually. No global build command exists.

### Local Jekyll Preview
```bash
bundle exec jekyll serve
```
The GitHub Pages site uses `jekyll-theme-minimal` (`_config.yml`).

### Live Server (VSCode)
Configured on port 5501 (`.vscode/settings.json`).

## Architecture

### Repository Structure

```
├── Notebooks/
│   ├── Python/     # 45+ Jupyter notebooks covering the full course
│   ├── R/          # 20+ R Markdown notebooks
│   ├── data/       # Shared datasets for notebooks (CSV, shapefiles, rasters)
│   └── cache/      # Cached computation results
├── Guias/
│   └── QGIS/       # Python scripts for QGIS/Google Earth Engine automation
├── data/           # Root-level datasets (accidents, air quality, weather CSVs + shapefiles)
├── html/           # Reveal.js presentation framework (course slides)
├── _layouts/       # Jekyll page templates
├── assets/css/     # Site-level SCSS styling
├── index.md        # Course homepage (Jekyll)
└── Programa_AnalisisGeoespacial.tex  # Full course syllabus (LaTeX)
```

### Two Separate Data Directories

There are **two `data/` folders** with overlapping content:
- `data/` — root-level data used by the website/presentations
- `Notebooks/data/` — data used specifically by the Jupyter/R notebooks

### Python Notebooks (`Notebooks/Python/`)

Notebooks are numbered by topic (not strictly sequential). They cover:
- Geospatial data I/O and transformation (geopandas, rasterio, xarray)
- Web mapping (folium, leafmap, contextily)
- Point pattern analysis, spatial clustering
- Spatial regression and geostatistics (pysal, skgstat, pykrige)
- Google Earth Engine (earthengine-api)
- Bayesian spatial models (MCMC), Gaussian processes, Kalman filtering
- Network analysis (osmnx)

### R Notebooks (`Notebooks/R/`)

Numbered 0–20. Cover:
- Spatial point patterns, area data (spdep)
- Bayesian CAR models (CARBayes, spaMM, glmmTMB)
- INLA models (R-INLA): CAR, Gaussian processes, LGCP
- Kriging (gstat)
- Spatially varying coefficient models (Bsvc)

### QGIS Scripts (`Guias/QGIS/`)

Python scripts designed to run **inside QGIS** using the `ee_plugin` and QGIS Python API. They cannot be run standalone — they depend on `from ee_plugin import Map` and QGIS's built-in Python environment.

### `.gitignore` Notes

Large geospatial files are excluded from version control: `.tif`, `.shp`, `.shx`, `.prj`, `.dbf`, `.cpg`, `.jp2`, `.docx`. Jupyter checkpoint directories (`.ipynb_checkpoints/`) are also excluded.
