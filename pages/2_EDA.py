# Prompt used to generate this page:
#
# Create an exploratory data analysis dashboard in the @DSML-2121-portfolio/pages/2_EDA.py
# file. It should contain the 4 charts listed below. For each chart, include a section where
# I can write a small "how to read this chart". At the top should be an "insights" section
# that I'll fill in. At the bottom should be a "future works" section I'll also fill in.
# There should be a filter button at the top for "exclude outliers", which switches each
# chart to the @DSML-2121-portfolio/assets/cleaned_winequality_red.csv instead of the
# @DSML-2121-portfolio/assets/winequality-red.csv .
# Charts:
# 1.  Violin plot of each column's values, similar to the one in
#     @DSML-2121-portfolio/data_exploration/data_cleaning.ipynb
# 2. Pairwise correlation heatmap of all variables excluding quality, correlation value
#    shown in the cell.
# 3. Barchart with filter for median, standard deviation, mean for all variables.
# 4. Barchart of correlation values of each variable against the quality target variable.

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Wine Quality EDA", page_icon="🍷", layout="wide")

# ---------- TODO: Fill in your own text ----------
INSIGHTS = """
- Many features have signifigant outliers. Chlorides and residual sugars especially. 
- Features tend to correlate with at least one other feature.
- Alcohol is positively correlated with quality, meaning the higher the alcohol content the better the score.
- Volatile acidity is negatively correlated with quality, meaning that it tends to hurt the score as it gets higher. 
"""

HOW_TO_READ = {
    "violin": "The width of the violin shows how many data points have that value. The box-and-whisker inside the violin show the quartiles. Red dots are outlier values. Try toggling the 'Exclude outliers' button at the top of the page to see how it changes.",
    "heatmap": "Each variable appears twice: once along the bottom and again on the side. Follow a variable along an axis to see its correlation to other variables. Values are between -1 and 1, with 0 being no correlation. Negative correlations are red and positive correlations are blue.",
    "summary": "Each chart has a bar for the mean, median and standard deviation - this is helpful for giving specific values to what you see in the violin plot.",
    "quality_corr": "Each bar shows a variable's correlation to quality, ranked from highest (1) to lowest (-1). Blue means the correlation is positive, red is negative.",
}



# ---------- Colors (validated color-blind safe on light and dark backgrounds) ----------
BLUE = "#2a78d6"          # positive / default series
BLUE_DARK = "#184f95"
RED = "#e34948"           # negative / outliers
RED_DARK = "#b23434"
NEUTRAL = "#f0efec"       # diverging midpoint
DIVERGING_SCALE = [
    [0.0, RED_DARK],
    [0.25, RED],
    [0.5, NEUTRAL],
    [0.75, BLUE],
    [1.0, BLUE_DARK],
]

# ---------- Units (from the dataset documentation, Cortez et al. 2009) ----------
UNITS = {
    "fixed acidity": "g tartaric acid/dm³",
    "volatile acidity": "g acetic acid/dm³",
    "citric acid": "g/dm³",
    "residual sugar": "g/dm³",
    "chlorides": "g sodium chloride/dm³",
    "free sulfur dioxide": "mg/dm³",
    "total sulfur dioxide": "mg/dm³",
    "density": "g/cm³",
    "pH": "unitless",
    "sulphates": "g potassium sulphate/dm³",
    "alcohol": "% vol",
    "quality": "score, 0–10",
}

TARGET = "quality"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RAW_PATH = ASSETS_DIR / "winequality-red.csv"
CLEAN_PATH = ASSETS_DIR / "cleaned_winequality_red.csv"


@st.cache_data
def load_data(exclude_outliers: bool) -> pd.DataFrame:
    if exclude_outliers:
        # The cleaned file was saved with its pandas index as the first column
        return pd.read_csv(CLEAN_PATH, index_col=0)
    return pd.read_csv(RAW_PATH)


def how_to_read(key: str) -> None:
    with st.expander("How to read this chart", expanded=True):
        st.markdown(HOW_TO_READ[key])


def chart_text_alternative(summary: str, data: pd.DataFrame) -> None:
    """Text alternative for a chart: a one-line summary plus the underlying numbers."""
    st.caption(f"**Chart summary:** {summary}")
    with st.expander("View chart data as a table"):
        st.dataframe(data)


def legend_key(fig: go.Figure, name: str, color: str, symbol: str = "circle") -> None:
    """Add a legend-only entry (no data) so a mark's meaning isn't explained by color alone."""
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name=name,
        marker=dict(color=color, size=10, symbol=symbol), hoverinfo="skip",
    ))


def iqr_outliers(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


# ---------- Header & filter ----------
st.title("🍷 Red Wine Quality: Exploratory Data Analysis")

exclude_outliers = st.toggle(
    "Exclude outliers",
    value=False,
    help="Use the cleaned dataset, where any row with a value beyond 1.5×IQR "
    "in any feature column has been removed.",
)
df = load_data(exclude_outliers)
features = [c for c in df.columns if c != TARGET]
cols = df.columns.to_list()
ncols = 4
nrows = math.ceil(len(cols) / ncols)

st.caption(
    f"Showing **{'cleaned' if exclude_outliers else 'raw'}** data: "
    f"{len(df):,} rows × {len(df.columns)} columns."
)

# ---------- Insights ----------
st.header("Insights")
st.markdown(INSIGHTS)
st.divider()

# ---------- Chart 1: Violin plots ----------
st.header("1. Distributions")

fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                    vertical_spacing=0.07, horizontal_spacing=0.07)
n_outliers = {}
for i, col in enumerate(cols):
    n_outliers[col] = int(iqr_outliers(df[col]).sum())
    row, c = i // ncols + 1, i % ncols + 1
    fig.add_trace(
        go.Violin(
            y=df[col],
            name=f"{n_outliers[col]} outliers",
            spanmode="hard",             # don't draw density beyond the observed min/max
            box_visible=True,
            meanline_visible=False,
            points="outliers",           # plotly uses the same 1.5×IQR rule
            fillcolor=BLUE,
            opacity=0.75,
            line_color=BLUE_DARK,
            line_width=1,
            marker=dict(color=RED, size=7, opacity=0.9),
            hoveron="violins+points",
            showlegend=False,
        ),
        row=row,
        col=c,
    )
    fig.update_yaxes(title_text=UNITS[col], title_font=dict(size=11), row=row, col=c)

legend_key(fig, "Distribution (box shows median and middle 50%)", BLUE, "square")
legend_key(fig, "Outlier (beyond 1.5×IQR from the box)", RED)
fig.update_layout(
    title="Distribution of each variable, in its own units",
    height=300 * nrows,
    margin=dict(t=110, b=20, l=20, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
)
fig.update_xaxes(tickfont=dict(size=11))
st.plotly_chart(fig)
st.caption("Each panel has its own y-axis because the variables use different units and "
           "scales. The label under each violin is its outlier count.")
most = sorted(n_outliers.items(), key=lambda kv: kv[1], reverse=True)[:3]
chart_text_alternative(
    f"The columns with the most outliers are "
    + ", ".join(f"{k} ({v})" for k, v in most) + ".",
    pd.DataFrame({
        "unit": [UNITS[c] for c in cols],
        "min": df.min(), "Q1": df.quantile(0.25), "median": df.median(),
        "Q3": df.quantile(0.75), "max": df.max(), "outliers (1.5×IQR)": n_outliers,
    }, index=cols).round(4),
)
how_to_read("violin")
st.divider()

# ---------- Chart 2: Correlation heatmap ----------
st.header("2. Feature correlations")

corr = df[features].corr()
fig = go.Figure(
    go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale=DIVERGING_SCALE,
        texttemplate="%{z:.2f}",
        textfont=dict(size=11),
        xgap=2,
        ygap=2,
        hovertemplate="%{y} × %{x}<br>Pearson r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r<br>(unitless)", tickvals=[-1, -0.5, 0, 0.5, 1]),
    )
)
fig.update_layout(
    title="Pearson correlation between every pair of features (−1 to +1)",
    height=680,
    margin=dict(t=60, b=20, l=20, r=20),
)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig)
pairs = corr.where(~pd.DataFrame(
    [[i <= j for j in range(len(corr))] for i in range(len(corr))],
    index=corr.index, columns=corr.columns,
)).stack()
pos, neg = pairs.idxmax(), pairs.idxmin()
chart_text_alternative(
    f"Blue cells are positive correlations and red cells are negative. The strongest positive "
    f"pair is {pos[0]} and {pos[1]} (r = {pairs[pos]:+.2f}); the strongest negative pair is "
    f"{neg[0]} and {neg[1]} (r = {pairs[neg]:+.2f}).",
    corr.round(3),
)
how_to_read("heatmap")
st.divider()

# ---------- Chart 3: Summary statistics ----------
st.header("3. Summary statistics")

stat_funcs = {
    "Mean": lambda d: d.mean(),
    "Median": lambda d: d.median(),
    "Std. dev.": lambda d: d.std(),
}
chosen = st.multiselect("Statistics", list(stat_funcs), default=list(stat_funcs))
stats = pd.DataFrame({name: f(df) for name, f in stat_funcs.items()})

if not chosen:
    st.info("Select at least one statistic.")
else:
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                        vertical_spacing=0.09, horizontal_spacing=0.07)
    for i, col in enumerate(cols):
        row, c = i // ncols + 1, i % ncols + 1
        vals = stats.loc[col, chosen]
        fig.add_trace(
            go.Bar(
                x=chosen,
                y=vals.values,
                marker_color=BLUE,
                text=[f"{v:,.3g}" for v in vals.values],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate=f"{col}<br>%{{x}} = %{{y:,.4f}} {UNITS[col]}<extra></extra>",
            ),
            row=row,
            col=c,
        )
        fig.update_yaxes(title_text=UNITS[col], title_font=dict(size=11),
                         range=[0, vals.max() * 1.25], row=row, col=c)
    fig.update_layout(
        title="Selected statistics for each variable, in its own units",
        height=280 * nrows,
        margin=dict(t=80, b=20, l=20, r=20),
        bargap=0.35,
    )
    st.plotly_chart(fig)
    st.caption("Each panel has its own y-axis starting at zero, so compare bars within a "
               "panel, not across panels.")
    # Features only: quality is an integer score, so its median gap is a rounding artifact
    s = stats.loc[features]
    skew = ((s["Mean"] - s["Median"]) / s["Std. dev."]).abs().sort_values()
    chart_text_alternative(
        f"The mean sits furthest from the median (relative to the spread) for {skew.index[-1]} "
        f"and {skew.index[-2]}, a sign of skewed distributions.",
        stats[chosen].round(4).assign(unit=[UNITS[c] for c in stats.index]),
    )
how_to_read("summary")
st.divider()

# ---------- Chart 4: Correlation with quality ----------
st.header("4. What moves with quality")

quality_corr = df[features].corrwith(df[TARGET]).sort_values()
fig = go.Figure()
for name, mask, color in [
    ("Positive: higher value, higher quality", quality_corr >= 0, BLUE),
    ("Negative: higher value, lower quality", quality_corr < 0, RED),
]:
    sub = quality_corr[mask]
    fig.add_trace(go.Bar(
        x=sub.values,
        y=sub.index,
        name=name,
        orientation="h",
        marker_color=color,
        text=[f"{v:+.2f}" for v in sub.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Pearson r with quality = %{x:.3f}<extra></extra>",
    ))
limit = max(abs(quality_corr).max() * 1.25, 0.1)
fig.update_layout(
    title="Pearson correlation of each feature with quality",
    height=520,
    margin=dict(t=90, b=20, l=20, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(title="Pearson r with quality (unitless, −1 to +1)", range=[-limit, limit],
               zeroline=True, zerolinewidth=1),
    yaxis=dict(categoryorder="array", categoryarray=quality_corr.index.to_list()),
)
st.plotly_chart(fig)
chart_text_alternative(
    f"{quality_corr.idxmax()} has the strongest positive correlation with quality "
    f"(r = {quality_corr.max():+.2f}) and {quality_corr.idxmin()} the strongest negative "
    f"(r = {quality_corr.min():+.2f}).",
    quality_corr.sort_values(ascending=False).round(3).rename("Pearson r with quality").to_frame(),
)
how_to_read("quality_corr")
st.divider()


