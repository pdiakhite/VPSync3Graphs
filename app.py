"""
Streamlit runner for selected graphs from `Streamlit-raise.py` (one scrollable page).

Graphs (in order):
  1. The 'Mirror' Effect — co-occurrence heatmap
  2. Impact Correlations — work/economy heatmap
  3. Did Precision Improve? — violin/strip from health_risk_analysis.csv
  4. AI Release Timeline vs Media Attention — static `ai_release_timeline_focus.png`
  5. Radial bubble network — Plotly + saved HTML
  6. RAISE All Topics map — mapbox dashboard (AI hubs: load `data_centers_geocoded.csv`; see `geocode_data_centers.py`)

Slices use original Colab line numbers; each chunk is preprocessed separately.
"""

from __future__ import annotations

import builtins
import io
import os
import traceback
import warnings
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAISE_ROOT = BASE_DIR
COLAB_SCRIPT = BASE_DIR / "Streamlit-raise.py"

INJECT_HEALTH_CONFIDENCE_VIOLIN = """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

guardian_df = pd.read_csv(str(__WORK_DIR__ / "health_risk_analysis.csv"))
plt.figure(figsize=(12, 6))
plot_df = guardian_df[guardian_df['Confidence'] > 0].copy()
order = plot_df.groupby('Predicted_Risk')['Confidence'].median().sort_values(ascending=False).index

sns.violinplot(
    x='Predicted_Risk',
    y='Confidence',
    data=plot_df,
    order=order,
    inner=None,
    linewidth=0,
    hue='Predicted_Risk',
    palette="pastel",
    legend=False,
    dodge=False,
)

sns.stripplot(x='Predicted_Risk', y='Confidence', data=plot_df,
              order=order, color='k', size=2, alpha=0.4, jitter=True)

plt.title('Did Precision Improve? AI Confidence by Risk Type', fontsize=14)
plt.ylabel('Model Confidence (0-1)')
plt.xlabel('')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
"""


def _first_existing_file(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def preprocess_chunk(raw: str) -> str:
    lines_out: list[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            indent = line[: len(line) - len(stripped)]
            lines_out.append(f"{indent}pass  # streamlit: skipped {stripped[:60]}")
        else:
            lines_out.append(line)
    text = "\n".join(lines_out)

    ds = "str(__DATASET_A_PATH__)"
    text = text.replace("f'/content/dataset_A_news_full_10500.csv'", ds)
    text = text.replace('f"/content/dataset_A_news_full_10500.csv"', ds)
    text = text.replace('"/content/dataset_A_news_full_10500.csv"', ds)
    text = text.replace("'/content/dataset_A_news_full_10500.csv'", ds)
    text = text.replace("/content/dataset_A_news_full_10500.csv", ds)

    text = text.replace(
        '"/content/impact_correlation_heatmap.png"',
        'str(__WORK_DIR__ / "impact_correlation_heatmap.png")',
    )
    text = text.replace(
        "/content/impact_correlation_heatmap.png",
        'str(__WORK_DIR__ / "impact_correlation_heatmap.png")',
    )
    text = text.replace(
        '"/content/data_centers_global.csv"',
        "str(__DATA_CENTERS_PATH__)",
    )
    text = text.replace(
        "/content/data_centers_global.csv",
        "str(__DATA_CENTERS_PATH__)",
    )

    text = text.replace(
        "from wordcloud import WordCloud\n",
        "try:\n    from wordcloud import WordCloud\nexcept ImportError:\n    WordCloud = None  # type: ignore\n",
        1,
    )

    text = text.replace(
        "subprocess.Popen(['ollama', 'serve'])",
        "pass  # streamlit: ollama serve not started",
    )

    text = text.replace(
        "if df is None:\n    exit(1)",
        "if df is None:\n    raise FileNotFoundError("
        '"Need FILTERED_WITH_SENTIMENT.csv under streamlit/ or Raise group/ for the timeline.")',
    )

    # Timeline: prefer local FILTERED_WITH_SENTIMENT paths
    text = text.replace(
        "possible_paths = [\n    'FILTERED_WITH_SENTIMENT.csv',\n    'mapcreationhtmlfile/FILTERED_WITH_SENTIMENT.csv',",
        "possible_paths = [\n    str(__WORK_DIR__ / 'FILTERED_WITH_SENTIMENT.csv'),\n"
        "    str(__RAISE_ROOT__ / 'streamlit' / 'FILTERED_WITH_SENTIMENT.csv'),\n"
        "    str(__RAISE_ROOT__ / 'FILTERED_WITH_SENTIMENT.csv'),\n"
        "    'FILTERED_WITH_SENTIMENT.csv',\n    'mapcreationhtmlfile/FILTERED_WITH_SENTIMENT.csv',",
        1,
    )

    # Radial: prefer local article CSVs
    text = text.replace(
        "possible_article_paths = [\n    'FILTERED_WITH_SENTIMENT.csv',\n    'scraped_articles_COMPLETE.csv',",
        "possible_article_paths = [\n    str(__WORK_DIR__ / 'FILTERED_WITH_SENTIMENT.csv'),\n"
        "    str(__RAISE_ROOT__ / 'streamlit' / 'FILTERED_WITH_SENTIMENT.csv'),\n"
        "    str(__RAISE_ROOT__ / 'FILTERED_WITH_SENTIMENT.csv'),\n"
        "    str(__RAISE_ROOT__ / 'scraped_articles_COMPLETE.csv'),\n"
        "    'FILTERED_WITH_SENTIMENT.csv',\n    'scraped_articles_COMPLETE.csv',",
        1,
    )

    # Topics map: load pre-geocoded hubs (instant). Live ArcGIS on 1300+ rows freezes the app for tens of minutes.
    _dc_old = """dc_traces = []
dc_path = str(__DATA_CENTERS_PATH__)
if os.path.exists(dc_path):
    print(" Geocoding Infrastructure with ArcGIS...")
    dc_df = pd.read_csv(dc_path).groupby(['City', 'Country']).agg({'Total_Data_Centers': 'sum'}).reset_index()
    geolocator = ArcGIS(timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.1)
    lats, lons, texts = [], [], []
    for _, row in dc_df.iterrows():
        try:
            loc = geocode(f"{row['City']}, {row['Country']}")
            if loc:
                lats.append(loc.latitude); lons.append(loc.longitude)
                texts.append(f" <b>AI HUB</b><br>{row['City']}<br>Centers: {row['Total_Data_Centers']}")
        except: continue
    dc_traces.append(go.Scattermapbox(lat=lats, lon=lons, mode='markers', marker=dict(size=8, color='white'), text=texts, hoverinfo='text', name='\U0001f3e2 AI Infrastructure'))"""
    _dc_new = """dc_traces = []
dc_path = str(__DATA_CENTERS_PATH__)
_coord_csv = None
for _p in (
    __WORK_DIR__ / "data_centers_geocoded.csv",
    __RAISE_ROOT__ / "streamlit" / "data_centers_geocoded.csv",
    __RAISE_ROOT__ / "data_centers_geocoded.csv",
):
    if _p.is_file():
        _coord_csv = _p
        break
if _coord_csv is not None:
    print(" Loading AI infrastructure from pre-geocoded CSV...")
    _gdc = pd.read_csv(_coord_csv)
    lats, lons, texts = [], [], []
    for _, row in _gdc.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        lats.append(lat)
        lons.append(lon)
        try:
            city = str(row["City"])
        except (KeyError, TypeError):
            city = str(row.get("city", ""))
        cnt = row.get("Total_Data_Centers", 0)
        try:
            cnt_i = int(float(cnt))
        except (TypeError, ValueError):
            cnt_i = 0
        texts.append(f" <b>AI HUB</b><br>{city}<br>Centers: {cnt_i}")
    if lats:
        dc_traces.append(go.Scattermapbox(lat=lats, lon=lons, mode='markers', marker=dict(size=8, color='white'), text=texts, hoverinfo='text', name='AI Infrastructure'))
elif os.path.exists(dc_path):
    print(
        " AI infrastructure: missing data_centers_geocoded.csv — from the streamlit folder run: "
        "python geocode_data_centers.py  (one-time, ~15–25 min; avoids freezing the web app.)"
    )"""
    text = text.replace(_dc_old, _dc_new, 1)

    return text


def slice_lines(full_text: str, start: int, end: int) -> str:
    lines = full_text.splitlines()
    return "\n".join(lines[start - 1 : end])


def main() -> None:
    st.set_page_config(page_title="RAISE-26 VPSync3 Presentation Graphs", layout="wide")
    st.title("RAISE-26 VPSync3 Presentation Graphs")

    # Plotly 6.x: silence Scattermapbox deprecation warning (MapLibre migration).
    warnings.filterwarnings(
        "ignore",
        message=r".*scattermapbox.*deprecated.*",
        category=DeprecationWarning,
    )

    if not COLAB_SCRIPT.is_file():
        st.error(
            f"Missing `{COLAB_SCRIPT.name}` in `{RAISE_ROOT}` (expected next to the `streamlit` folder)."
        )
        return

    raw = COLAB_SCRIPT.read_text(encoding="utf-8")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(DATA_DIR)

    orig_plt_show = plt.show
    orig_go_show = go.Figure.show

    def plt_show_streamlit(*_a, **_kw) -> None:
        for num in plt.get_fignums():
            st.pyplot(plt.figure(num), width="stretch")
        plt.close("all")

    def go_show_streamlit(self, *_a, **_kw) -> None:
        st.plotly_chart(self, width="stretch")

    plt.show = plt_show_streamlit
    go.Figure.show = go_show_streamlit

    dataset_a = _first_existing_file([DATA_DIR / "dataset_A_news_full_10500.csv"])
    data_centers = _first_existing_file([DATA_DIR / "data_centers_global.csv"])

    g: dict = {
        "__name__": "__main__",
        "__file__": str(COLAB_SCRIPT),
        "__builtins__": builtins,
        "__WORK_DIR__": DATA_DIR,
        "__RAISE_ROOT__": RAISE_ROOT,
        "__DATASET_A_PATH__": dataset_a,
        "__DATA_CENTERS_PATH__": data_centers,
    }

    # (title, kind, start, end | None, inject | slice filename | None)
    # kind "image": inject = image filename under streamlit/ or Raise group/
    sections: list[tuple[str, str, int | None, int | None, str | None]] = [
        ("1. The 'Mirror' Effect: Co-occurrence of AI Behavioral Themes", "slice", 13, 68, None),
        ("2. Impact Correlations: What Happens Together?", "slice", 233, 294, None),
        (
            "3. Did Precision Improve? AI Confidence by Risk Type",
            "inject",
            None,
            None,
            INJECT_HEALTH_CONFIDENCE_VIOLIN,
        ),
        (
            "4. AI Release Timeline vs Media Attention",
            "image",
            None,
            None,
            "ai_release_timeline_focus.png",
        ),
        ("5. Radial Bubble Network", "slice", 446, 923, None),
        ("6. RAISE All Topics Map", "slice", 1936, 2061, None),
    ]

    for title, kind, a, b, inject in sections:
        st.markdown("---")
        if title.startswith("5. Radial Bubble Network"):
            st.markdown(
                "### 5. Radial Bubble Network "
                "<span style='color:#dc2626'>(Interactive)</span>",
                unsafe_allow_html=True,
            )
        elif title == "6. RAISE All Topics Map":
            st.markdown(
                "### 6. RAISE All Topics Map "
                "<span style='color:#dc2626'>(Interactive)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.subheader(title)

        if kind == "image":
            fname = inject or "ai_release_timeline_focus.png"
            img_path = _first_existing_file([DATA_DIR / fname, RAISE_ROOT / fname])
            if img_path.is_file():
                st.image(str(img_path), width="stretch")
            else:
                st.warning(
                    f"Missing `{fname}`. Place it in `{DATA_DIR}` or `{RAISE_ROOT}` "
                    "(same chart as Colab `ai_release_timeline_focus.png`)."
                )
        else:
            if kind == "inject":
                code = preprocess_chunk(inject or "pass\n")
            else:
                assert a is not None and b is not None
                code = preprocess_chunk(slice_lines(raw, a, b))

            buf_out, buf_err = io.StringIO(), io.StringIO()
            spin = (
                st.spinner("Building map from Dataset A…")
                if title == "6. RAISE All Topics Map"
                else nullcontext()
            )
            try:
                with spin:
                    with redirect_stdout(buf_out), redirect_stderr(buf_err):
                        exec(code, g, g)
            except Exception as e:
                st.error(f"**{type(e).__name__}:** {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
            else:
                out, err = buf_out.getvalue().strip(), buf_err.getvalue().strip()
                # Radial + Topics map: underlying scripts are verbose; hide stdout-only "Printed output"
                if out and not (title.startswith("5. Radial") or title == "6. RAISE All Topics Map"):
                    with st.expander("Printed output"):
                        st.code(out)
                if err:
                    st.warning(err)
                if title == "6. RAISE All Topics Map":
                    _geo = _first_existing_file(
                        [
                            DATA_DIR / "data_centers_geocoded.csv",
                            RAISE_ROOT / "streamlit" / "data_centers_geocoded.csv",
                            RAISE_ROOT / "data_centers_geocoded.csv",
                        ]
                    )
                    if not _geo.is_file():
                        st.caption(
                            "White AI infrastructure hubs need `data_centers_geocoded.csv` in this folder "
                            "(or repo root). Generate once: `python geocode_data_centers.py` (~15–25 min); "
                            "the live app skips online geocoding so the page stays responsive."
                        )

        if title.startswith("5. Radial"):
            html_path = DATA_DIR / "keywords_countries_regions_articles.html"
            if html_path.is_file():
                st.caption("Radial network (exported HTML)")
                st.components.v1.html(
                    html_path.read_text(encoding="utf-8", errors="replace"),
                    height=1050,
                    scrolling=True,
                )

    plt.show = orig_plt_show
    go.Figure.show = orig_go_show


if __name__ == "__main__":
    main()
