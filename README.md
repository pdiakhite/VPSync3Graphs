# RAISE-26 VPSync3 Presentation Graphs

Streamlit app for the VPSync3 presentation graphs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data files

The app expects these files under `data/`:

- `dataset_A_news_full_10500.csv`
- `FILTERED_WITH_SENTIMENT.csv`
- `health_risk_analysis.csv`
- `keywords_countries_regions_articles.html`
- `ai_release_timeline_focus.png`
- `data_centers_geocoded.csv` (optional but recommended for fast AI infrastructure hubs)
- `data_centers_global.csv` (input to generate the geocoded file)

### Generate `data_centers_geocoded.csv` (one-time)

This avoids geocoding 1000+ cities on every page load.

```bash
python geocode_data_centers.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app:
   - **Repository**: this repo
   - **Branch**: `main`
   - **Main file path**: `app.py`

