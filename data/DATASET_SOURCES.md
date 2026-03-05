# WeatherWise.ai — Golden Dataset Sources (Real-World Only)

> **Strategy:** No synthetic data. Every document must be a real publication with messy, diverse layouts to stress-test our `unstructured` hi_res parsing and Multi-Vector Retrieval pipeline.

## Directory Structure

```
data/
├── raw_pdfs/           # Research papers, technical reports, climate assessments
├── satellite_imagery/  # GOES, MODIS, VIIRS snapshots (PNG/JPEG/GeoTIFF)
├── news_articles/      # Archived weather analysis articles (HTML → PDF)
├── docstore/           # (Generated) Parent documents keyed by doc_id
└── DATASET_SOURCES.md  # This file
```

---

## Category 1: Government Technical Reports (PDFs with Charts, Tables, Figures)

These are the highest-value targets — dense PDFs with mixed layouts that will stress the `unstructured` parser.

### 1A. NOAA Technical Memorandums

| Document | URL | Why It Matters |
|----------|-----|----------------|
| **NOAA Technical Memorandum OAR CPO 003** — Climate Risk Reduction (2024) | [PDF](https://repository.library.noaa.gov/view/noaa/66829/noaa_66829_DS1.pdf) | Multi-column layout, embedded charts on climate hazards, tables of engineering thresholds. Tests table extraction. |
| **CRS Report: Extreme Heat, NOAA's Role (R48579)** — June 2025 | [PDF](https://www.congress.gov/crs_external_products/R/PDF/R48579/R48579.1.pdf) | Defines heat dome mechanics, maps NOAA sub-agency roles, includes budget tables and organizational charts. Directly relevant to Bucket 5 queries. |
| **NOAA Storm Data Publication** (Monthly PDFs) | [Storm Events DB](https://www.ncei.noaa.gov/stormevents/) | Monthly chronological listings of severe storms by state. Tables with location, time, magnitude (hail size, wind speeds, snowfall), damage, injuries, deaths. Free PDF download. Tests dense tabular extraction. |
| **NHC Tropical Cyclone Reports** | [NHC Data Archive](https://www.nhc.noaa.gov/data/) | Post-season hurricane reports with track maps, wind speed charts, pressure graphs, damage assessments. Rich multimodal content. |

**How to obtain:** Direct PDF download from links above. Browse [NOAA Repository](https://repository.library.noaa.gov/) for additional technical memorandums by searching "atmospheric blocking", "heat dome", "lake effect", "polar vortex".

### 1B. NOAA/NWS Storm Prediction Center

| Document | URL | Why It Matters |
|----------|-----|----------------|
| **Severe Weather Event Summaries** | [SPC Archive](https://www.spc.noaa.gov/exper/archive/events/) | Individual event case studies with radar imagery, mesoscale discussion text, and outbreak maps. |
| **SPC Climatology & Trend Reports** | [SPC Climo](https://www.spc.noaa.gov/climo/online/) | Historical tornado/hail/wind statistics with trend charts. Tests time-series chart extraction. |
| **SPC Maps, Graphics, and Data** | [SPC Data Page](https://origin-west-www-spc.woc.noaa.gov/wcm/) | Downloadable graphics, probability maps, and CSV data for severe weather events. |

### 1C. Fifth National Climate Assessment (NCA5) — 2023

| Document | URL | Why It Matters |
|----------|-----|----------------|
| **Full Report + Individual Chapters** | [NCA5 Downloads](https://nca2023.globalchange.gov/downloads/) | The gold standard. 32 chapters, hundreds of figures, maps, and tables covering every climate topic. Chapter 2 (Climate Trends) alone has dozens of time-series charts and spatial maps. |
| **Chapter 2: Climate Trends** | [NCA5 Ch.2](https://nca2023.globalchange.gov/chapter/2/) | Temperature anomaly maps, precipitation trend charts, sea level rise graphs. Ideal for testing image summarization pipeline. |

**Priority downloads for initial ingestion:**
- Ch. 2 (Climate Trends) — temperature/precip charts
- Ch. 4 (Water) — drought maps, streamflow tables
- Ch. 15 (Air Quality) — AQI trend figures, PM2.5 maps

---

## Category 2: Peer-Reviewed Research Papers (Open Access PDFs)

These papers contain the deep scientific content needed for Bucket 5 (Explainability) responses.

### 2A. Atmospheric Blocking & Heat Domes

| Paper | URL | Why It Matters |
|-------|-----|----------------|
| **Frontiers in Climate: Climatological Patterns of Heatwaves in South America** (2025) | [PDF](https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2025.1529082/pdf) | Open access. Covers heatwave climatology, blocking patterns, soil desiccation feedback. Contains spatial maps and time-series figures. |
| **Nature npj: Anthropogenic Warming & South American Heatwave** (2025) | [Article](https://www.nature.com/articles/s41612-025-01142-5) | Attribution analysis of heat dome formation. Pressure anomaly charts, 500mb geopotential height maps. Directly maps to our Bucket 5 example query. |
| **AGU Earth's Future: Future Climate & Lake-Effect Snow Storms** (2024) | [Article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024EF004622) | Models how climate change reshapes lake-effect snow. Lake evaporation charts, snowband simulation figures. Tests complex figure extraction. |

### 2B. Lake Effect & Winter Weather

| Paper | URL | Why It Matters |
|-------|-----|----------------|
| **AMS/NASA: Snowfall in Northern Great Lakes — Multisensor Observatory** (2021) | [PDF](https://ntrs.nasa.gov/api/citations/20210026101/downloads/15200477%20-%20BAMS_Snowfall%20in%20the%20Northern%20Great%20Lakes%20Lessons%20Learned%20from%20a%20Multisensor%20Observatory.pdf) | Micro Rain Radar data, Precipitation Imaging Package outputs, snowfall regime classifications. Heavy on instrument diagrams and data plots. |
| **MDPI Climate: Synoptic Climatology of Lake-Effect Snow Events** (2021) | [Article](https://www.mdpi.com/2225-1154/9/3/43) | Open access. Synoptic pattern classifications with composite maps. Tables of event frequencies by lake. |

### 2C. How to Find More

- **AMS Journals** ([journals.ametsoc.org](https://journals.ametsoc.org)) — Search for open access papers. Filter by "Weather and Forecasting", "Journal of Climate", "BAMS".
- **Frontiers in Climate** ([frontiersin.org/journals/climate](https://www.frontiersin.org/journals/climate)) — All open access. Search "blocking pattern", "polar vortex", "tropical cyclone intensification".
- **NASA Technical Reports Server** ([ntrs.nasa.gov](https://ntrs.nasa.gov)) — Free PDFs from NASA-funded atmospheric research.
- **IPCC AR6 WG1** ([Full Downloads](https://www.ipcc.ch/report/ar6/wg1/downloads/)) — The Physical Science Basis. 2,000+ pages of figures, charts, and tables across 12 chapters. Highest figure density of any source on this list.

---

## Category 3: Satellite Imagery (Visual Data for Image Summarization Pipeline)

These test our Claude Vision image-to-text summarization pipeline. Each image gets a technical summary embedded in ChromaDB, with the original stored in `data/satellite_imagery/`.

### 3A. GOES Geostationary Satellite

| Source | URL | Content |
|--------|-----|---------|
| **GOES Imagery Viewer** | [NOAA STAR](https://www.star.nesdis.noaa.gov/GOES/index.php) | Near real-time GOES-East/West. Bands: visible, infrared, water vapor. Download snapshots of hurricanes, frontal systems, convective outbreaks. |
| **GOES on AWS (Open Data)** | [Registry](https://registry.opendata.aws/noaa-goes/) | Full-resolution ABI Level 1b/2 products: cloud top height, aerosol optical depth, clear sky mask. Bulk download via S3. GOES-16 data spans 2018–2024. |
| **GOES-19 First Light Imagery** | [NESDIS News](https://www.nesdis.noaa.gov/news/noaa-debuts-first-imagery-goes-19) | Showcase images from the newest GOES satellite. High-quality full-disk and mesoscale imagery. |

### 3B. NASA Worldview (MODIS/VIIRS)

| Source | URL | Content |
|--------|-----|---------|
| **NASA Worldview** | [worldview.earthdata.nasa.gov](https://worldview.earthdata.nasa.gov/) | 1,200+ layers. Download snapshots of: wildfire smoke plumes, dust storms, hurricane spiral bands, snow cover, aerosol optical depth. Export as PNG/JPEG/GeoTIFF. |
| **FIRMS Fire Satellite Imagery** | [FIRMS](https://firms.modaps.eosdis.nasa.gov/satellite-imagery/) | Near real-time MODIS/VIIRS true-color imagery centered on active fire detections. Perfect for wildfire smoke analysis. |

### 3C. Recommended Snapshots to Capture

Download specific event snapshots that map to our Bucket 5 topics:

| Event | Date Range | What to Capture |
|-------|-----------|-----------------|
| Pacific Northwest Heat Dome | June 2021 | GOES water vapor loop, 500mb height anomaly map |
| Great Lakes Lake-Effect Snow | Nov-Dec 2022 | MODIS true color of snowbands off Lake Erie/Ontario |
| Canadian Wildfire Smoke over US | June 2023 | VIIRS aerosol imagery, PM2.5 plume extent |
| Hurricane Helene | Sep 2024 | GOES infrared, NHC track map, wind speed analysis |
| LA Wildfires | Jan 2025 | VIIRS hotspot detection, smoke plume imagery |

---

## Category 4: Weather Analysis & News Articles (HTML with Embedded Charts)

These provide accessible, well-written explanations with inline data visualizations — useful for testing our parser on non-PDF web content saved as HTML/PDF.

### 4A. Yale Climate Connections (Eye on the Storm)

| Source | URL | Content |
|--------|-----|---------|
| **Eye on the Storm** (Jeff Masters & Bob Henson) | [Yale CC](https://yaleclimateconnections.org/eye-on-the-storm/) | Expert storm analysis articles with embedded satellite imagery, forecast model charts, and historical comparison graphs. Save as PDF via browser print. |
| **Billion-Dollar Weather Disasters 2024** | [Article](https://yaleclimateconnections.org/2025/01/the-planet-had-58-billion-dollar-weather-disasters-in-2024-the-second-highest-on-record/) | Cost/damage infographics, event maps, trend charts. |
| **2025 Global Temperature Record** | [Article](https://yaleclimateconnections.org/2026/01/2025-was-earths-3rd-warmest-year-on-record/) | Temperature anomaly charts, ranking tables, comparison figures. |

### 4B. NOAA Climate.gov

| Source | URL | Content |
|--------|-----|---------|
| **State of the Climate Reports** | [NOAA Climate](https://www.noaa.gov/climate) | Annual and monthly climate summaries with maps, charts, and data tables. |
| **2024 National Climate Report** | [NCEI Report](https://www.ncei.noaa.gov/access/monitoring/monthly-report/national/2024) | US temperature/precip anomaly maps, state-level rankings, time-series plots. |

### 4C. NASA Earth Observatory

| Source | URL | Content |
|--------|-----|---------|
| **Natural Hazards** | [earthobservatory.nasa.gov](https://earthobservatory.nasa.gov/natural-hazards) | Annotated satellite imagery of fires, floods, storms, volcanic eruptions. Each article pairs imagery with scientific explanation. Save as PDF. |

---

## Category 5: Structured Data (Tables & CSVs for Cross-Reference)

Not for embedding directly, but useful for verification and enriching the docstore.

| Source | URL | Content |
|--------|-----|---------|
| **NCEI Climate Data Online** | [CDO](https://www.ncei.noaa.gov/cdo-web/) | Historical weather observations. Download station-level CSV data for temperature, precip, wind, snow. |
| **EPA AirNow Historical Data** | [airnow.gov](https://www.airnow.gov/aqi/aqi-basics/) | Historical AQI data by city. CSV download for PM2.5, ozone trends. |
| **NOAA Severe Storms Data Table** | [Climate.gov](https://www.climate.gov/maps-data/dataset/severe-storms-and-extreme-events-data-table) | Tabular severe event data with filtering by type, date, and region. |

---

## Ingestion Priority Order

For Phase 4a, ingest in this order to incrementally test the pipeline:

### Wave 1 — Smoke Test (3 documents)
1. **CRS Extreme Heat Report** (1C) — Clean government PDF, moderate complexity
2. **Frontiers Heatwave Paper** (2A) — Open access, spatial maps + time-series
3. **1 GOES satellite snapshot** (3C) — Single image to test Vision summarization

### Wave 2 — Stress Test (5-7 documents)
4. **NCA5 Chapter 2** — High figure density, complex multi-column layout
5. **AMS/NASA Great Lakes Snowfall** paper — Instrument diagrams, radar plots
6. **2-3 SPC Event Summaries** — Messy real-world severe weather reports
7. **2-3 satellite snapshots** from different events (wildfire, hurricane, snow)

### Wave 3 — Full Corpus (15-20 documents)
8. Remaining NCA5 chapters (4, 15)
9. Additional research papers from 2A/2B
10. Yale CC / NOAA Climate articles saved as PDF
11. NHC Tropical Cyclone Reports
12. IPCC AR6 WG1 selected chapters
13. Full satellite imagery set from 3C event list

---

## Scraping & Download Notes

- **PDFs:** Direct download where links are provided. For AMS papers behind paywalls, check author preprints on [ResearchGate](https://www.researchgate.net/) or [arXiv](https://arxiv.org/).
- **Satellite Imagery:** Use NASA Worldview's "snapshot" feature (camera icon) to export specific dates/regions as PNG. For bulk GOES data, use the AWS S3 bucket with `aws s3 cp`.
- **Web Articles:** Use browser "Print to PDF" to preserve embedded charts and figures for `unstructured` parsing.
- **Respect Terms:** All sources listed are public domain (US government) or open access. No paywall circumvention needed.
