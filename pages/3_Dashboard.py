# Prompt used to generate this page:
#
# Now create a dashboard to make predictions on the dataset. It should still have the filter
# for removing outliers at the top. Also include a row at the top showing the average, median
# and std of quality
# Visuals:
# 1. Predicted vs actual linear regression plot. User can select which variables are included.
#    Quality is the target variable.
# 2. SHAP value for each variable in a bar chart, static
# 3. Confusion matrix for good/bad prediction using logistic regression, user can set a cutoff
#    for "good" from 0-10. No variable exclusion.
#
# At the bottom of the page, include a citation and link to the source:
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wine Quality Predictions", page_icon="🍷", layout="wide")

# ---------- TODO: Fill in your own text ----------
INSIGHTS = """
- The r-squared increases from .403 to .423 when outliers are excluded on linear regression
- The model overestimates quality on actual scores below 5, and underestimates on scores over 7
- Alcohol content has the greatest contribution to model quality
- Residual sugar contributes nothing to the model's predictions
- A cutoff of >=6 for "good" predictions gives the best balance of accuracy (75.7%), precision (78%), and recall (76.7%)
"""

# ---------- Colors (validated color-blind safe on light and dark backgrounds) ----------
BLUE = "#2a78d6"
BLUE_SCALE = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#0d366b"]]
GRAY = "#8a8984"          # neutral reference line, dashed and named in the legend

TARGET = "quality"
QUALITY_UNIT = "score, 0–10"
RANDOM_STATE = 42
TEST_SIZE = 0.2
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RAW_PATH = ASSETS_DIR / "winequality-red.csv"
CLEAN_PATH = ASSETS_DIR / "cleaned_winequality_red.csv"


@st.cache_data
def load_data(exclude_outliers: bool) -> pd.DataFrame:
    if exclude_outliers:
        # The cleaned file was saved with its pandas index as the first column
        return pd.read_csv(CLEAN_PATH, index_col=0)
    return pd.read_csv(RAW_PATH)


def chart_text_alternative(summary: str, data: pd.DataFrame) -> None:
    """Text alternative for a chart: a one-line summary plus the underlying numbers."""
    st.caption(f"**Chart summary:** {summary}")
    with st.expander("View chart data as a table"):
        st.dataframe(data)


# ---------- Header & filter ----------
st.title("🍷 Predicting Red Wine Quality")

exclude_outliers = st.toggle(
    "Exclude outliers",
    value=False,
    help="Use the cleaned dataset, where any row with a value beyond 1.5×IQR "
    "in any feature column has been removed.",
)
df = load_data(exclude_outliers)
features = [c for c in df.columns if c != TARGET]
X_all, y = df[features], df[TARGET]

st.caption(
    f"Showing **{'cleaned' if exclude_outliers else 'raw'}** data: {len(df):,} wines. "
    f"Models are trained on {1 - TEST_SIZE:.0%} of rows and evaluated on the held-out "
    f"{TEST_SIZE:.0%}. Quality is a sensory score from 0 (very bad) to 10 (excellent)."
)

# ---------- Insights ----------
st.header("Insights")
st.markdown(INSIGHTS)
st.divider()

# ---------- Quality summary row ----------
m1, m2, m3 = st.columns(3)
m1.metric(f"Average quality ({QUALITY_UNIT})", f"{y.mean():.2f}")
m2.metric(f"Median quality ({QUALITY_UNIT})", f"{y.median():.1f}")
m3.metric("Std. dev. of quality (score points)", f"{y.std():.2f}")
st.divider()

# ---------- 1. Linear regression: predicted vs actual ----------
st.header("1. Linear regression")

selected = st.multiselect("Variables included in the model", features, default=features)

if not selected:
    st.info("Select at least one variable to fit the model.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        df[selected], y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    lin = LinearRegression().fit(X_train, y_train)
    y_pred = lin.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("R² on test set (unitless, 0–1)", f"{r2:.3f}")
    c2.metric("RMSE on test set (score points)", f"{rmse:.3f}")
    c3.metric("Variables used", f"{len(selected)} of {len(features)}")

    # Quality is an integer, so add a small horizontal jitter to separate overlapping points
    jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.15, 0.15, len(y_test))
    lo = min(y_test.min(), y_pred.min()) - 0.5
    hi = max(y_test.max(), y_pred.max()) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (predicted = actual)",
        line=dict(color=GRAY, width=2, dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=y_test.values + jitter, y=y_pred, mode="markers", name="Test-set wine",
        marker=dict(color=BLUE, size=8, opacity=0.5),
        customdata=y_test.values,
        hovertemplate="Actual: %{customdata}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Predicted vs. actual quality on the held-out test set",
        height=580,
        margin=dict(t=90, b=20, l=20, r=20),
        xaxis=dict(title=f"Actual quality ({QUALITY_UNIT})", range=[lo, hi], dtick=1),
        yaxis=dict(title=f"Predicted quality ({QUALITY_UNIT})", range=[lo, hi], dtick=1,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig)
    st.caption("Both axes use the same scale. Actual quality values are jittered slightly "
               "left/right so overlapping wines stay visible.")
    chart_text_alternative(
        f"Using {len(selected)} variable(s), the model explains {r2:.0%} of the variance in "
        f"quality, and its predictions are off by about {rmse:.2f} score points on average "
        f"(RMSE). Predictions bunch toward the middle of the scale, so the best and worst "
        f"wines are under- and over-predicted.",
        pd.DataFrame({"Actual quality": y_test.values,
                      "Predicted quality": y_pred.round(2),
                      "Error (predicted − actual)": (y_pred - y_test.values).round(2)}),
    )
st.divider()

# ---------- 2. SHAP values ----------
st.header("2. Feature importance")

# Uses every variable, independent of the selection above.
# For a linear model with independent features, the SHAP value of feature j for row i is
# exactly coef_j * (x_ij - mean(x_j)) — the same result shap.LinearExplainer returns.
X_train_all, X_test_all, y_train_all, _ = train_test_split(
    X_all, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
lin_all = LinearRegression().fit(X_train_all, y_train_all)
shap_values = lin_all.coef_ * (X_test_all - X_train_all.mean())
mean_abs_shap = shap_values.abs().mean().sort_values()

fig = go.Figure(go.Bar(
    x=mean_abs_shap.values,
    y=mean_abs_shap.index,
    orientation="h",
    marker_color=BLUE,
    text=[f"{v:.3f}" for v in mean_abs_shap.values],
    textposition="outside",
    cliponaxis=False,
    hovertemplate="%{y}<br>Mean |SHAP| = %{x:.4f} score points<extra></extra>",
))
fig.update_layout(
    title="Average impact of each feature on predicted quality (mean |SHAP value|)",
    height=520,
    margin=dict(t=60, b=20, l=20, r=60),
    xaxis_title="Mean |SHAP value| (score points of predicted quality)",
)
st.plotly_chart(fig)
st.caption("Computed from a linear regression on all variables, evaluated on the test set. "
           "This chart does not change with the variable selection above.")
top3 = mean_abs_shap.sort_values(ascending=False)
chart_text_alternative(
    f"The three most influential features are {top3.index[0]} ({top3.iloc[0]:.3f}), "
    f"{top3.index[1]} ({top3.iloc[1]:.3f}) and {top3.index[2]} ({top3.iloc[2]:.3f}) "
    f"score points of average impact.",
    top3.round(4).rename("Mean |SHAP| (score points)").to_frame(),
)
st.divider()

# ---------- 3. Logistic regression confusion matrix ----------
st.header("3. Good vs. bad wine classifier")

cutoff = st.slider(f"A wine is \"good\" if quality ({QUALITY_UNIT}) ≥",
                   min_value=0, max_value=10, value=7)
y_good = (y >= cutoff).astype(int)
n_good = int(y_good.sum())

if n_good == 0 or n_good == len(y_good):
    label = "good" if n_good else "bad"
    st.warning(
        f"With a cutoff of {cutoff}, every wine is \"{label}\" (quality ranges from "
        f"{y.min()} to {y.max()}), so there is nothing to classify. Choose another cutoff."
    )
else:
    # Stratify so both classes appear in the test set; fall back if a class is too small
    stratify = y_good if y_good.value_counts().min() >= 2 else None
    X_train, X_test, yg_train, yg_test = train_test_split(
        X_all, y_good, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logit.fit(X_train, yg_train)
    yg_pred = logit.predict(X_test)
    acc = accuracy_score(yg_test, yg_pred)
    rec = recall_score(yg_test, yg_pred, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Good wines in data", f"{n_good:,} ({n_good / len(y_good):.0%})")
    c2.metric("Accuracy on test set", f"{acc:.1%}")
    c3.metric("Precision for good", f"{precision_score(yg_test, yg_pred, zero_division=0):.1%}")
    c4.metric("Recall for good", f"{rec:.1%}")

    cm = confusion_matrix(yg_test, yg_pred, labels=[0, 1])
    names = [f"Bad (< {cutoff})", f"Good (≥ {cutoff})"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=names,
        y=names,
        colorscale=BLUE_SCALE,
        texttemplate="%{z} wines",
        textfont=dict(size=20),
        xgap=2,
        ygap=2,
        colorbar=dict(title="Number of<br>wines"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>%{z} wines<extra></extra>",
    ))
    fig.update_layout(
        title=f"Confusion matrix: logistic regression, good = quality ≥ {cutoff}",
        height=470,
        margin=dict(t=60, b=20, l=20, r=20),
        xaxis=dict(title="Predicted class", side="bottom"),
        yaxis=dict(title="Actual class", autorange="reversed"),
    )
    left, _ = st.columns([2, 1])
    with left:
        st.plotly_chart(fig)
    st.caption("Uses all variables. The diagonal (top-left, bottom-right) counts correct "
               "predictions on the held-out test set.")
    tn, fp, fn, tp = cm.ravel()
    chart_text_alternative(
        f"Of {cm.sum()} test wines, {tn + tp} were classified correctly ({acc:.0%}). "
        f"The model found {tp} of the {tp + fn} truly good wines (recall {rec:.0%}) and "
        f"wrongly labeled {fp} bad wines as good.",
        pd.DataFrame(cm, index=[f"Actual {n}" for n in names],
                     columns=[f"Predicted {n}" for n in names]),
    )
st.divider()

# ---------- Citation ----------
st.subheader("Data source")
st.markdown(
    """
    Yasser H. *Wine Quality Dataset*. Kaggle (11/18/2026).
    [https://www.kaggle.com/datasets/yasserh/wine-quality-dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

    Originally from: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. *Modeling wine
    preferences by data mining from physicochemical properties.* Decision Support Systems,
    47(4):547–553, 2009.
    """
)
