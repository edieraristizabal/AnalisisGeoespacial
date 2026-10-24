# CLAUDE.md

Guía para Claude Code (y para cualquier persona) que actualice este repositorio. **Cada cambio que se haga en el repositorio debe registrarse al final, en "Registro de cambios".**

## Qué es

Curso **Análisis Geoespacial** (Universidad Nacional de Colombia, sede Medellín, Prof. Edier Aristizábal). Un solo repositorio contiene tres productos:

| Producto | Fuente en `main` | URL publicada |
|---|---|---|
| Página del curso (Jekyll, tema minimal) | `index.md`, `_config.yml`, programa en PDF | https://edieraristizabal.github.io/AnalisisGeoespacial/ |
| Diapositivas (Reveal.js) | `html/AnalisisGeoespacial.html` y sus recursos en `html/` | https://edieraristizabal.github.io/AnalisisGeoespacial/html/AnalisisGeoespacial.html |
| Libro (Jupyter Book 1.x) | `libro/` | https://edieraristizabal.github.io/AnalisisGeoespacial/libro/ |

Remoto: https://github.com/edieraristizabal/AnalisisGeoespacial

## Resumen del contenido

El curso Análisis Geoespacial está orientado para estudiantes de posgrados que desen adquirir conocimientos sobre sensoramiento remoto y procesamiento de datos espaciales en un contexto ambiental. El curso utiliza herramientas tipo sistemas de información geográfica, como *QGIS* y *ArcGIS*, *Google Earth Engine* (GEE) y programación en lenguajes como *Python* y *Javascript* utilizando como IDEs los editores *Jupyter Lab*, *Visual Code* y *Antigravity*.

El libro está formado por notebooks de Python con salidas guardadas y dos capítulos en R Markdown (`16_GP.Rmd`, `17_LGCP.Rmd`, modelos INLA) que se publican solo con el código, igual que en el libro anterior; para mostrarlos con resultados habría que ejecutarlos con R (faltan localmente los paquetes geoR, tidyverse, spocc, rnaturalearth y rgeos, este último retirado de CRAN). `libro/requirements.txt` incluye `jupytext` para leer los `.Rmd`.

Contenido del libro (orden de `libro/_toc.yml`; el título es el primer encabezado de cada archivo):

- `intro`: portada

**Parte I: Intro**

- `00_Ambiente`: Ambiente computacional
- `00_Data`: Datos
- `01_DatosEspaciales`: Análisis Geoespacial
- `02_Mapping`: El arte de hacer mapas

**Parte II: Puntos**

- `03_PointPattern`: Patrón de puntos

**Parte III: Áreas**

- `04_GLMPhyton`: Modelos Lineales Generalizados (GML)
- `06_Coropleta`: Visualización de datos discretos
- `07_MatrizCorrelacion`: Matriz espacial
- `08_ClusterEspacial`: Cluster espacial
- `09_SpatialRegression`: Regresión Espacial
- `10_SAR`: Modelos de regresión para dependencia espacial tipo SAR
- `11_CAR`: Modelos de regresión para dependencia espacial tipo CAR
- `12_Jerarquicos`: Modelos de Regresión para Heterogeneidad Espacial
- `12_CARJerarquicos`: Modelos con dependencia y heterogeneidad espacial
- `13_MGWR`: Regresión Ponderada Geográficamente (GWR)

**Parte IV: Superficies**

- `14_Kriging`: Kriging con Python
- `15_GP`: Procesos Gaussianos con Python
- `16_GP`: Procesos Gaussianos con INLA: el enfoque de malla (SPDE)
- `17_LGCP`: El Proceso de Cox Log-Gaussiano (LGCP)
- `18_GPBoost`: GPBoost: Boosting de Árboles + Procesos Gaussianos
- `18_DB`: Bases de datos geoespaciales

## Ramas: dónde hacer pull y push

**Siempre se trabaja en `main`.** Ese es el único lugar donde se edita, se hace `git pull` y `git push`, tanto para cambios de la página como de las diapositivas o del libro.

**Nunca editar ni hacer push a `gh-pages` a mano.** Esa rama la genera el workflow `.github/workflows/deploy.yml` en cada push a `main`: construye Jekyll en `_site/`, construye el libro y lo copia en `_site/libro/`, y publica todo en `gh-pages` con `force_orphan` (cualquier cambio manual en `gh-pages` se pierde en el siguiente despliegue).

GitHub Pages sirve un solo sitio por repositorio; por eso la página y el libro se combinan en el Action. En **Settings → Pages** la fuente debe ser *Deploy from a branch*, rama `gh-pages`, carpeta `/ (root)`.

Flujo normal:

```bash
git pull origin main
# ... editar ...
# ... agregar una línea al Registro de cambios de este archivo ...
git add -A
git commit -m "Descripción del cambio"
git push origin main      # el Action publica en 2 a 5 minutos (pestaña Actions)
```

## Estructura

```
.github/
    workflows/
.gitignore
09_estadistico-jvsc-332d2dd2-88cc-4fdc-9cbf-8fb916e699e9dda2aa32-dd32-4cc1-b6c9-5df8b17ee472.ipynb
CLAUDE.md
Guias/
    QGIS/
Notebooks/
    Python/
    R/
    cache/
    data/
Programa_AnalisisGeoespacial.pdf
Programa_AnalisisGeoespacial.tex
README.md
_config.yml
_layouts/
    default.html
assets/
build/
data/
html/
    AnalisisGeoespacial.html
    data/
index.md
libro/
```

Material fuente que **no** se sube (libros guía, presentaciones originales, documentos con derechos): está en `C:\Users\edier\Documents\CATEDRA\ANALISISGEOESPACIAL`, fuera de este repositorio.

## Cómo actualizar

**Página del curso.** Editar `index.md`. Los enlaces internos deben ser relativos (`Programa.pdf`, `html/...`), nunca empezar por `/`, porque el sitio vive en `/AnalisisGeoespacial/`. El libro se enlaza como `https://edieraristizabal.github.io/AnalisisGeoespacial/libro/<capítulo>.html`. Si cambia la evaluación, actualizarla también en el programa (`.tex` + PDF) y en las diapositivas.

**Diapositivas.** Editar `html/AnalisisGeoespacial.html` directamente (Reveal.js; ecuaciones con MathJax). Previsualizar con `python -m http.server` desde la raíz y abrir `http://localhost:8000/html/AnalisisGeoespacial.html`.

**Libro.** Capítulos en MyST Markdown (`.md`) o notebooks (`.ipynb`) dentro de `libro/`; un capítulo nuevo debe agregarse a `libro/_toc.yml`; las citas `{cite:t}` / `{cite:p}` deben existir en el archivo `.bib` del libro.

- **El Action no ejecuta los notebooks** (`execute_notebooks: "off"`): publica las salidas guardadas en cada `.ipynb`. Si se cambia código, ejecutar el notebook localmente y guardarlo con sus salidas antes del commit.
- `libro/requirements.txt` contiene solo lo necesario para construir el libro; `libro/requirements-notebooks.txt` (si existe) lista las librerías para ejecutar los notebooks.
- Compilar localmente: `C:\Users\edier\.jbenv\Scripts\jupyter-book.exe build libro` y abrir `libro/_build/html/index.html`.

## Notas del repositorio (CLAUDE.md anterior)

Lo siguiente proviene del CLAUDE.md que ya existía en este repositorio y sigue vigente para los notebooks del curso (carpeta `Notebooks/`), que son distintos de los del libro (`libro/`).

### Commands

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

### Architecture

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

## Preferencias del autor

- Proponer los cambios de contenido del curso y esperar aprobación antes de editarlos.
- Prosa en párrafos, español, sin rayas (em dash): usar comas o paréntesis.
- Las figuras nuevas se buscan primero en los tableros de Pinterest del autor.

## Pendientes conocidos

- Cambiar en Settings → Pages la fuente a la rama `gh-pages`.
- Borrar el repositorio viejo `Libro_AnalisisGeoespacial` cuando se verifique el libro nuevo.
- `libro/how_update.txt` y `libro/append_script.py` son instrucciones y utilidades del flujo anterior (rutas de Google Drive); revisar si se eliminan.
- Los capítulos `16_GP.Rmd` y `17_LGCP.Rmd` se publican sin salidas (ver Resumen).

## Registro de cambios

Cada cambio al repositorio se anota aquí, el más reciente primero, con el formato `- **AAAA-MM-DD**: qué se cambió y por qué (archivos principales).` Esta sección es la memoria del curso entre sesiones: leerla antes de empezar a trabajar.

- **2026-10-24**: CLAUDE.md reescrito con resumen del contenido, estructura, flujo de ramas, pendientes y este registro; el contenido del CLAUDE.md anterior se conserva en "Notas del repositorio".
- **2026-10-24**: Unificación en un solo repositorio: el libro (antes `Libro_AnalisisGeoespacial`) se copió a `libro/` (sin `cache/` ni `.claude/`) con `execute_notebooks: off`; `jupytext` agregado a `libro/requirements.txt` para los capítulos `.Rmd`; cita del libro en `libro/intro.md` actualizada a la URL nueva; workflow `deploy.yml`; `_config.yml` con `baseurl: /AnalisisGeoespacial`; enlaces de `index.md` relativos (antes `/Notebooks/...`) y enlaces al libro en `index.md` y en `html/AnalisisGeoespacial.html` apuntando a `/AnalisisGeoespacial/libro/`.
- **2026-10-24**: Commit de respaldo de los cambios locales pendientes (53 archivos y 6 nuevos en `Notebooks/Python`) antes de unificar; limpieza de `desktop.ini` dentro de `.git/`.
