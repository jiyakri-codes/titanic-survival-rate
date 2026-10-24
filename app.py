"""
🚢 Titanic Survival 3D Dashboard
================================
Interactive ML-powered Titanic dashboard
Built with Streamlit · Plotly · Scikit-Learn

Version : 3.0.0
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Titanic 3D Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* HERO */

.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(99,102,241,.35);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e0e7ff;
    letter-spacing: 2px;
    margin: 0 0 .4rem 0;
}

.hero-sub {
    font-size: 1rem;
    color: #a5b4fc;
    margin: 0;
}


/* KPI */

.kpi-card {
    background: linear-gradient(145deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(67,56,202,.3);
    transition: transform .2s;
}

.kpi-card:hover {
    transform: translateY(-4px);
}

.kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #c7d2fe;
}

.kpi-label {
    font-size: .82rem;
    color: #818cf8;
    letter-spacing: 1px;
    text-transform: uppercase;
}


/* SECTION */

.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e7ff;
    border-left: 4px solid #6366f1;
    padding-left: .75rem;
    margin: 1.5rem 0 1rem 0;
}


/* DIVIDER */

.glow-divider {
    height: 2px;
    background: linear-gradient(
        90deg,
        transparent,
        #6366f1,
        #a78bfa,
        transparent
    );
    border: none;
    margin: 2rem 0;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1744);
    border-right: 1px solid #312e81;
}


/* PREDICTION */

.pred-box {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #059669;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #6ee7b7;
}

.pred-box.died {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-color: #dc2626;
    color: #fca5a5;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME
# ============================================================

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(15,12,41,0)",
        plot_bgcolor="rgba(30,27,75,0.4)",
        font=dict(color="#c7d2fe", family="Inter"),
        title_font=dict(color="#e0e7ff", size=16),
        legend=dict(
            bgcolor="rgba(30,27,75,0.7)",
            bordercolor="#4338ca",
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor="#312e81",
            zerolinecolor="#4338ca",
            linecolor="#4338ca"
        ),
        yaxis=dict(
            gridcolor="#312e81",
            zerolinecolor="#4338ca",
            linecolor="#4338ca"
        ),
        margin=dict(t=50, b=40, l=40, r=20)
    )
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=86400)
def load_and_engineer(path="train.csv"):

    df = pd.read_csv(path)

    # Basic cleaning
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna("S")

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")
    else:
        df["Cabin"] = "Unknown"

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (
        df["FamilySize"] == 1
    ).astype(int)

    # Title
    if "Name" in df.columns:

        df["Title"] = (
            df["Name"]
            .str.extract(r" ([A-Za-z]+)\.", expand=False)
            .fillna("Rare")
        )

        df["Title"] = df["Title"].replace({
            "Mme": "Mrs",
            "Ms": "Miss",
            "Mlle": "Miss"
        })

        title_counts = df["Title"].value_counts()

        rare_titles = title_counts[
            title_counts < 10
        ].index

        df["Title"] = df["Title"].replace(
            rare_titles,
            "Rare"
        )

    else:
        df["Title"] = "Rare"

    # Fare per person
    df["FarePerPerson"] = (
        df["Fare"] /
        df["FamilySize"].replace(0, 1)
    )

    # Age band
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 55, 100],
        labels=[
            "Child",
            "Teen",
            "YoungAdult",
            "Adult",
            "Senior"
        ],
        include_lowest=True
    ).astype(str)

    # Fare band
    try:
        df["FareBand"] = pd.qcut(
            df["Fare"],
            q=4,
            labels=[
                "Low",
                "Medium",
                "High",
                "VeryHigh"
            ],
            duplicates="drop"
        ).astype(str)
    except Exception:
        df["FareBand"] = "Medium"

    # Deck
    df["Deck"] = df["Cabin"].apply(
        lambda x: x[0]
        if x != "Unknown"
        else "U"
    )

    return df


# ============================================================
# ML FEATURES
# ============================================================

FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "Title",
    "Embarked",
    "FarePerPerson",
    "AgeBand",
    "FareBand"
]

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "FarePerPerson"
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Title",
    "Embarked",
    "AgeBand",
    "FareBand"
]


def create_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )


def build_pipeline(model):

    return Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            model
        )
    ])


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = load_and_engineer("train.csv")

    DATA_LOADED = True

except FileNotFoundError:

    DATA_LOADED = False


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-banner">

<p class="hero-title">
🚢 TITANIC 3D DASHBOARD
</p>

<p class="hero-sub">
Interactive · ML-Powered · Visually Stunning — v3.0
</p>

</div>
""", unsafe_allow_html=True)


if not DATA_LOADED:

    st.error(
        "⚠️ `train.csv` not found. "
        "Place train.csv in the same folder as this app."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🗺️ Navigation")

    page = st.radio(
        "Go to section:",
        [
            "📊 Overview & KPIs",
            "🌐 3D Visualizations",
            "🔬 Deep-Dive Analysis",
            "🤖 ML Model Lab",
            "🎯 Live Predictor",
            "👤 Passenger Explorer"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        "<hr style='border-color:#312e81'>",
        unsafe_allow_html=True
    )

    st.markdown("## 🔍 Global Filters")

    selected_pclass = st.multiselect(
        "Passenger Class",
        sorted(data["Pclass"].unique()),
        default=sorted(data["Pclass"].unique())
    )

    selected_sex = st.multiselect(
        "Sex",
        list(data["Sex"].unique()),
        default=list(data["Sex"].unique())
    )

    age_range = st.slider(
        "Age Range",
        int(data["Age"].min()),
        int(data["Age"].max()),
        (
            int(data["Age"].min()),
            int(data["Age"].max())
        )
    )

    fare_range = st.slider(
        "Fare Range",
        float(data["Fare"].min()),
        float(data["Fare"].max()),
        (
            float(data["Fare"].min()),
            float(data["Fare"].max())
        )
    )

    selected_embarked = st.multiselect(
        "Port of Embarkation",
        list(data["Embarked"].unique()),
        default=list(data["Embarked"].unique())
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered = data[
    data["Pclass"].isin(selected_pclass)
    &
    data["Sex"].isin(selected_sex)
    &
    data["Age"].between(*age_range)
    &
    data["Fare"].between(*fare_range)
    &
    data["Embarked"].isin(selected_embarked)
].copy()


# ============================================================
# EMPTY FILTER PROTECTION
# ============================================================

if filtered.empty:

    st.warning(
        "⚠️ No passengers match the selected filters. "
        "Please adjust the sidebar filters."
    )

    st.stop()


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview & KPIs":

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)

    survived_count = int(
        filtered["Survived"].sum()
    )

    died_count = int(
        len(filtered) - survived_count
    )

    survival_rate = (
        filtered["Survived"].mean() * 100
    )

    kpis = [
        (
            "👥 Passengers",
            f"{len(filtered):,}"
        ),
        (
            "✅ Survived",
            f"{survived_count:,}"
        ),
        (
            "💀 Perished",
            f"{died_count:,}"
        ),
        (
            "📈 Survival Rate",
            f"{survival_rate:.1f}%"
        ),
        (
            "💰 Median Fare",
            f"${filtered['Fare'].median():.1f}"
        )
    ]

    for col, (label, value) in zip(
        [k1, k2, k3, k4, k5],
        kpis
    ):

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">

                <div class="kpi-value">
                {value}
                </div>

                <div class="kpi-label">
                {label}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        "<div class='glow-divider'></div>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Automatic Insights
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>💡 Automatic Insights</div>",
        unsafe_allow_html=True
    )

    insight1 = (
        f"📊 The selected passengers have a "
        f"**{survival_rate:.1f}% survival rate**."
    )

    sex_rates = (
        filtered.groupby("Sex")["Survived"]
        .mean()
        .sort_values(ascending=False)
    )

    class_rates = (
        filtered.groupby("Pclass")["Survived"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(sex_rates) > 0:

        best_sex = sex_rates.index[0]

        insight2 = (
            f"👤 **{best_sex.title()}** passengers "
            f"have the highest survival rate "
            f"({sex_rates.iloc[0] * 100:.1f}%)."
        )

    else:

        insight2 = "No gender insight available."


    if len(class_rates) > 0:

        best_class = int(class_rates.index[0])

        insight3 = (
            f"🎫 Class **{best_class}** has the "
            f"highest survival rate "
            f"({class_rates.iloc[0] * 100:.1f}%)."
        )

    else:

        insight3 = "No class insight available."


    i1, i2, i3 = st.columns(3)

    i1.info(insight1)
    i2.success(insight2)
    i3.warning(insight3)


    # --------------------------------------------------------
    # Survival Breakdown
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>📊 Survival Breakdown</div>",
        unsafe_allow_html=True
    )

    fig_grid = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Survival by Class",
            "Survival by Sex",
            "Survival by Embarkation",
            "Survival by Title",
            "Survival by Family Size",
            "Survival by Deck"
        ]
    )


    def add_bar(
        column,
        row,
        col
    ):

        grp = (
            filtered
            .groupby(column)["Survived"]
            .agg(["mean", "count"])
            .reset_index()
        )

        fig_grid.add_trace(
            go.Bar(
                x=grp[column].astype(str),
                y=grp["mean"],
                text=[
                    f"{v:.0%}"
                    for v in grp["mean"]
                ],
                textposition="outside",
                marker=dict(
                    color=grp["mean"],
                    colorscale=[
                        [0, "#f87171"],
                        [0.5, "#fbbf24"],
                        [1, "#34d399"]
                    ],
                    showscale=False
                ),
                customdata=grp["count"],
                hovertemplate=(
                    "%{x}<br>"
                    "Survival: %{y:.1%}<br>"
                    "Passengers: %{customdata}"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig_grid.update_yaxes(
            tickformat="%",
            range=[0, 1.15],
            row=row,
            col=col
        )


    add_bar("Pclass", 1, 1)
    add_bar("Sex", 1, 2)
    add_bar("Embarked", 1, 3)
    add_bar("Title", 2, 1)
    add_bar("FamilySize", 2, 2)
    add_bar("Deck", 2, 3)

    fig_grid.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=600
    )

    st.plotly_chart(
        fig_grid,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>📉 Distribution Analysis</div>",
        unsafe_allow_html=True
    )

    dc1, dc2 = st.columns(2)

    with dc1:

        fig_age = px.histogram(
            filtered,
            x="Age",
            color="Survived",
            nbins=35,
            marginal="violin",
            title="Age Distribution",
            color_discrete_map={
                0: "#f87171",
                1: "#34d399"
            }
        )

        fig_age.update_layout(
            **PLOTLY_TEMPLATE["layout"]
        )

        st.plotly_chart(
            fig_age,
            use_container_width=True
        )


    with dc2:

        fig_fare = px.box(
            filtered,
            x="Pclass",
            y="Fare",
            color="Survived",
            title="Fare Distribution by Class",
            color_discrete_map={
                0: "#f87171",
                1: "#34d399"
            }
        )

        fig_fare.update_layout(
            **PLOTLY_TEMPLATE["layout"]
        )

        st.plotly_chart(
            fig_fare,
            use_container_width=True
        )


    # Download

    st.download_button(
        "📥 Download Filtered Data",
        filtered.to_csv(index=False),
        "filtered_titanic_data.csv",
        "text/csv"
    )


# ============================================================
# PAGE 2 — 3D
# ============================================================

elif page == "🌐 3D Visualizations":

    st.markdown(
        "<div class='section-header'>🌐 3D Passenger Visualization</div>",
        unsafe_allow_html=True
    )

    fig3d = px.scatter_3d(
        filtered,
        x="Age",
        y="Fare",
        z="Pclass",
        color="Survived",
        symbol="Sex",
        size="FamilySize",
        hover_data=[
            "Name",
            "Title",
            "Embarked",
            "FamilySize"
        ],
        color_discrete_map={
            0: "#f87171",
            1: "#34d399"
        }
    )

    fig3d.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=650,
        scene=dict(
            xaxis_title="Age",
            yaxis_title="Fare",
            zaxis_title="Passenger Class",
            bgcolor="rgba(15,12,41,0.8)"
        )
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )


    # --------------------------------------------------------
    # 3D Survival Surface
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>🏔️ Survival Probability Surface</div>",
        unsafe_allow_html=True
    )

    if filtered["Age"].nunique() > 5 and filtered["Fare"].nunique() > 5:

        age_bins = np.linspace(
            filtered["Age"].min(),
            filtered["Age"].max(),
            20
        )

        fare_bins = np.linspace(
            filtered["Fare"].min(),
            min(filtered["Fare"].max(), 200),
            20
        )

        matrix = np.full(
            (len(fare_bins)-1, len(age_bins)-1),
            np.nan
        )

        for i, (fl, fu) in enumerate(
            zip(fare_bins[:-1], fare_bins[1:])
        ):

            for j, (al, au) in enumerate(
                zip(age_bins[:-1], age_bins[1:])
            ):

                sub = filtered[
                    filtered["Age"].between(al, au)
                    &
                    filtered["Fare"].between(fl, fu)
                ]

                if len(sub) >= 2:

                    matrix[i, j] = (
                        sub["Survived"].mean()
                    )

        matrix = (
            pd.DataFrame(matrix)
            .interpolate(axis=0)
            .interpolate(axis=1)
            .ffill()
            .bfill()
            .values
        )

        fig_surface = go.Figure(
            go.Surface(
                z=matrix,
                x=age_bins[:-1],
                y=fare_bins[:-1],
                colorscale=[
                    [0, "#f87171"],
                    [0.5, "#fbbf24"],
                    [1, "#34d399"]
                ],
                colorbar=dict(
                    title="Survival"
                )
            )
        )

        fig_surface.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=600,
            scene=dict(
                xaxis_title="Age",
                yaxis_title="Fare",
                zaxis_title="Survival Probability",
                bgcolor="rgba(15,12,41,0.8)"
            )
        )

        st.plotly_chart(
            fig_surface,
            use_container_width=True
        )

    else:

        st.info(
            "Not enough variation in the filtered data "
            "to generate the surface."
        )


    # --------------------------------------------------------
    # 3D Bar
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>📦 Class × Sex Survival</div>",
        unsafe_allow_html=True
    )

    grouped = (
        filtered
        .groupby(["Pclass", "Sex"])["Survived"]
        .mean()
        .reset_index()
    )

    fig_bar = px.bar(
        grouped,
        x="Pclass",
        y="Survived",
        color="Sex",
        barmode="group",
        text_auto=".0%",
        title="Survival Rate by Class and Sex",
        color_discrete_sequence=[
            "#818cf8",
            "#f472b6"
        ]
    )

    fig_bar.update_yaxes(
        tickformat=".0%",
        range=[0, 1.1]
    )

    fig_bar.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=450
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TREEMAP
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>🌳 Survival Treemap</div>",
        unsafe_allow_html=True
    )

    tree_df = filtered.copy()

    tree_df["Class"] = tree_df[
        "Pclass"
    ].map({
        1: "1st Class",
        2: "2nd Class",
        3: "3rd Class"
    })

    tree_df["Status"] = tree_df[
        "Survived"
    ].map({
        0: "Perished",
        1: "Survived"
    })

    tree_agg = (
        tree_df
        .groupby(
            ["Class", "Sex", "Status"]
        )
        .size()
        .reset_index(name="Passengers")
    )

    fig_tree = px.treemap(
        tree_agg,
        path=["Class", "Sex", "Status"],
        values="Passengers",
        color="Status",
        color_discrete_map={
            "Survived": "#34d399",
            "Perished": "#f87171"
        }
    )

    fig_tree.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=550
    )

    st.plotly_chart(
        fig_tree,
        use_container_width=True
    )


# ============================================================
# PAGE 3 — DEEP DIVE
# ============================================================

elif page == "🔬 Deep-Dive Analysis":

    st.markdown(
        "<div class='section-header'>🔗 Correlation Matrix</div>",
        unsafe_allow_html=True
    )

    numeric_cols = [
        "Age",
        "Fare",
        "Pclass",
        "SibSp",
        "Parch",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "Survived"
    ]

    corr = filtered[numeric_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[
            "#f87171",
            "#1e1b4b",
            "#34d399"
        ],
        title="Pearson Correlation Matrix"
    )

    fig_corr.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=550
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Parallel Coordinates
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>🌊 Parallel Coordinates</div>",
        unsafe_allow_html=True
    )

    pc_df = filtered[
        [
            "Age",
            "Fare",
            "Pclass",
            "FamilySize",
            "SibSp",
            "Parch",
            "Survived"
        ]
    ].dropna()

    fig_parallel = px.parallel_coordinates(
        pc_df,
        color="Survived",
        dimensions=[
            "Pclass",
            "Age",
            "Fare",
            "FamilySize",
            "SibSp",
            "Parch"
        ],
        color_continuous_scale=[
            "#f87171",
            "#34d399"
        ]
    )

    fig_parallel.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=500
    )

    st.plotly_chart(
        fig_parallel,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Sunburst
    # --------------------------------------------------------

    d1, d2 = st.columns(2)

    with d1:

        temp = filtered.copy()

        temp["Class"] = temp[
            "Pclass"
        ].map({
            1: "1st",
            2: "2nd",
            3: "3rd"
        })

        temp["Status"] = temp[
            "Survived"
        ].map({
            0: "Perished",
            1: "Survived"
        })

        sun = (
            temp
            .groupby(
                ["Class", "Embarked", "Status"]
            )
            .size()
            .reset_index(name="Count")
        )

        fig_sun = px.sunburst(
            sun,
            path=[
                "Class",
                "Embarked",
                "Status"
            ],
            values="Count",
            color="Status",
            color_discrete_map={
                "Survived": "#34d399",
                "Perished": "#f87171"
            }
        )

        fig_sun.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=500
        )

        st.plotly_chart(
            fig_sun,
            use_container_width=True
        )


    with d2:

        fig_violin = px.violin(
            filtered,
            x="Pclass",
            y="Fare",
            color="Survived",
            box=True,
            points="outliers",
            color_discrete_map={
                0: "#f87171",
                1: "#34d399"
            },
            title="Fare Distribution"
        )

        fig_violin.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=500
        )

        st.plotly_chart(
            fig_violin,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Scatter Matrix
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-header'>🔭 Scatter Matrix</div>",
        unsafe_allow_html=True
    )

    sm_cols = [
        "Age",
        "Fare",
        "FamilySize",
        "FarePerPerson",
        "Survived"
    ]

    fig_matrix = px.scatter_matrix(
        filtered[sm_cols].dropna(),
        dimensions=[
            "Age",
            "Fare",
            "FamilySize",
            "FarePerPerson"
        ],
        color="Survived",
        color_discrete_map={
            0: "#f87171",
            1: "#34d399"
        }
    )

    fig_matrix.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=600
    )

    st.plotly_chart(
        fig_matrix,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — ML MODEL LAB
# ============================================================

elif page == "🤖 ML Model Lab":

    st.markdown(
        "<div class='section-header'>🤖 Machine Learning Model Laboratory</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Train and compare multiple machine learning models."
    )


    X = data[FEATURE_COLS].copy()
    y = data["Survived"].copy()


    # --------------------------------------------------------
    # MODEL SETTINGS
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 2])

    with col1:

        models_to_train = st.multiselect(
            "Select Models",
            [
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
                "AdaBoost",
                "SVM"
            ],
            default=[
                "Random Forest",
                "Gradient Boosting"
            ]
        )

        test_size = (
            st.slider(
                "Test Split %",
                10,
                40,
                20
            ) / 100
        )

        cv_folds = st.slider(
            "Cross Validation Folds",
            3,
            10,
            5
        )

        n_estimators = st.slider(
            "Number of Trees",
            50,
            500,
            200,
            step=50
        )

        train_button = st.button(
            "🚀 Train Models",
            use_container_width=True
        )


    # --------------------------------------------------------
    # MODEL MAP
    # --------------------------------------------------------

    MODEL_MAP = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=42
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "AdaBoost":
            AdaBoostClassifier(
                n_estimators=n_estimators,
                random_state=42
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=42
            )
    }


    if train_button:

        if not models_to_train:

            st.warning(
                "Please select at least one model."
            )

            st.stop()


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )


        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )


        results = {}
        trained_models = {}


        progress = st.progress(
            0,
            text="Starting training..."
        )


        for index, model_name in enumerate(
            models_to_train
        ):

            progress.progress(
                index / len(models_to_train),
                text=f"Training {model_name}..."
            )


            pipeline = build_pipeline(
                MODEL_MAP[model_name]
            )


            # Cross validation
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1
            )


            # Train
            pipeline.fit(
                X_train,
                y_train
            )


            # Predictions
            predictions = pipeline.predict(
                X_test
            )

            probabilities = pipeline.predict_proba(
                X_test
            )[:, 1]


            results[model_name] = {

                "accuracy":
                    accuracy_score(
                        y_test,
                        predictions
                    ),

                "precision":
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                "cv_mean":
                    cv_scores.mean(),

                "cv_std":
                    cv_scores.std(),

                "auc":
                    roc_auc_score(
                        y_test,
                        probabilities
                    ),

                "cm":
                    confusion_matrix(
                        y_test,
                        predictions
                    ),

                "probabilities":
                    probabilities
            }


            trained_models[
                model_name
            ] = pipeline


        progress.progress(
            1.0,
            text="✅ Training complete!"
        )


        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        result_table = pd.DataFrame([
            {
                "Model": model,
                "Test Accuracy":
                    round(
                        results[model]["accuracy"],
                        4
                    ),
                "CV Mean":
                    round(
                        results[model]["cv_mean"],
                        4
                    ),
                "Precision":
                    round(
                        results[model]["precision"],
                        4
                    ),
                "Recall":
                    round(
                        results[model]["recall"],
                        4
                    ),
                "F1 Score":
                    round(
                        results[model]["f1"],
                        4
                    ),
                "AUC":
                    round(
                        results[model]["auc"],
                        4
                    )
            }
            for model in models_to_train
        ])


        st.markdown(
            "### 🏆 Model Performance"
        )

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model_name = max(
            results,
            key=lambda x:
            results[x]["cv_mean"]
        )

        best_model = trained_models[
            best_model_name
        ]


        st.success(
            f"🏆 Best model based on cross-validation: "
            f"**{best_model_name}** "
            f"({results[best_model_name]['cv_mean']:.2%})"
        )


        # Save complete pipeline

        joblib.dump(
            best_model,
            "titanic_best_model.pkl"
        )

        joblib.dump(
            FEATURE_COLS,
            "titanic_feature_cols.pkl"
        )


        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        fig_compare = px.bar(
            result_table,
            x="Model",
            y=[
                "Test Accuracy",
                "CV Mean",
                "AUC"
            ],
            barmode="group",
            title="Model Comparison",
            text_auto=".3f"
        )

        fig_compare.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=450
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )


        # ----------------------------------------------------
        # ROC CURVES
        # ----------------------------------------------------

        st.markdown(
            "### 📈 ROC Curves"
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(
                    dash="dash"
                ),
                name="Random"
            )
        )


        for model_name in models_to_train:

            probabilities = results[
                model_name
            ]["probabilities"]

            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities
            )

            auc = results[
                model_name
            ]["auc"]

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} (AUC={auc:.3f})"
                )
            )


        fig_roc.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="ROC-AUC Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450
        )

        st.plotly_chart(
            fig_roc,
            use_container_width=True
        )


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.markdown(
            "### 🔲 Confusion Matrices"
        )

        for model_name in models_to_train:

            cm = results[
                model_name
            ]["cm"]

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=[
                    "#1e1b4b",
                    "#6366f1"
                ],
                x=[
                    "Predicted 0",
                    "Predicted 1"
                ],
                y=[
                    "Actual 0",
                    "Actual 1"
                ],
                title=model_name
            )

            fig_cm.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=350
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        if best_model_name in [
            "Random Forest",
            "Gradient Boosting",
            "AdaBoost"
        ]:

            estimator = best_model.named_steps[
                "model"
            ]

            preprocessor = best_model.named_steps[
                "preprocessor"
            ]


            # Get transformed feature names
            feature_names = (
                preprocessor
                .get_feature_names_out()
            )


            importance = (
                estimator.feature_importances_
            )


            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            })


            fi = fi.sort_values(
                "Importance",
                ascending=False
            ).head(20)


            fig_fi = px.bar(
                fi.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title=(
                    f"Top Features — "
                    f"{best_model_name}"
                ),
                color="Importance",
                color_continuous_scale="Viridis"
            )


            fig_fi.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=600
            )


            st.plotly_chart(
                fig_fi,
                use_container_width=True
            )


        # Download results

        st.download_button(
            "📥 Download Model Results",
            result_table.to_csv(
                index=False
            ),
            "model_results.csv",
            "text/csv"
        )


# ============================================================
# PAGE 5 — LIVE PREDICTOR
# ============================================================

elif page == "🎯 Live Predictor":

    st.markdown(
        "<div class='section-header'>🎯 Live Survival Predictor</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Enter passenger information and predict "
        "their survival probability."
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = joblib.load(
            "titanic_best_model.pkl"
        )

        model_ready = True

    except FileNotFoundError:

        model_ready = False

        st.warning(
            "⚠️ No trained model found. "
            "Go to ML Model Lab and train a model first."
        )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            index=2
        )

        sex = st.radio(
            "Sex",
            ["male", "female"]
        )

        age = st.slider(
            "Age",
            1,
            80,
            30
        )


    with c2:

        sibsp = st.number_input(
            "Siblings / Spouses",
            0,
            8,
            0
        )

        parch = st.number_input(
            "Parents / Children",
            0,
            6,
            0
        )

        fare = st.number_input(
            "Ticket Fare ($)",
            min_value=0.0,
            max_value=520.0,
            value=30.0
        )


    with c3:

        embarked = st.selectbox(
            "Port",
            ["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown"
            }[x]
        )

        title = st.selectbox(
            "Title",
            [
                "Mr",
                "Mrs",
                "Miss",
                "Master",
                "Rare"
            ]
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if model_ready:

        family_size = (
            sibsp +
            parch +
            1
        )

        is_alone = (
            1
            if family_size == 1
            else 0
        )


        if age <= 12:
            age_band = "Child"
        elif age <= 18:
            age_band = "Teen"
        elif age <= 35:
            age_band = "YoungAdult"
        elif age <= 55:
            age_band = "Adult"
        else:
            age_band = "Senior"


        # Approximate fare bands
        if fare <= 7.91:
            fare_band = "Low"
        elif fare <= 14.45:
            fare_band = "Medium"
        elif fare <= 31:
            fare_band = "High"
        else:
            fare_band = "VeryHigh"


        passenger = pd.DataFrame([{

            "Pclass": pclass,

            "Sex": sex,

            "Age": age,

            "Fare": fare,

            "SibSp": sibsp,

            "Parch": parch,

            "FamilySize": family_size,

            "IsAlone": is_alone,

            "Title": title,

            "Embarked": embarked,

            "FarePerPerson":
                fare /
                max(family_size, 1),

            "AgeBand": age_band,

            "FareBand": fare_band

        }])


        probability = model.predict_proba(
            passenger
        )[0][1]

        prediction = model.predict(
            passenger
        )[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result_class = (
            ""
            if prediction == 1
            else "died"
        )

        result_text = (
            "🟢 SURVIVED"
            if prediction == 1
            else "🔴 PERISHED"
        )


        st.markdown(
            f"""
            <div class="pred-box {result_class}">
            {result_text}
            <br>
            Survival Probability:
            {probability * 100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # GAUGE
        # ----------------------------------------------------

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number=dict(
                    suffix="%"
                ),
                title=dict(
                    text="Survival Probability"
                ),
                gauge=dict(

                    axis=dict(
                        range=[0, 100]
                    ),

                    bar=dict(
                        color=(
                            "#34d399"
                            if probability >= 0.5
                            else "#f87171"
                        )
                    ),

                    steps=[
                        dict(
                            range=[0, 30],
                            color="rgba(248,113,113,.25)"
                        ),

                        dict(
                            range=[30, 70],
                            color="rgba(251,191,36,.15)"
                        ),

                        dict(
                            range=[70, 100],
                            color="rgba(52,211,153,.25)"
                        )
                    ]
                )
            )
        )


        fig_gauge.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=350
        )


        st.plotly_chart(
            fig_gauge,
            use_container_width=True
        )


        # ----------------------------------------------------
        # PASSENGER PROFILE
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-header'>👤 Passenger Profile</div>",
            unsafe_allow_html=True
        )


        profile = pd.DataFrame({
            "Feature": [
                "Class",
                "Sex",
                "Age",
                "Family Size",
                "Fare",
                "Embarkation",
                "Title"
            ],

            "Value": [
                pclass,
                sex,
                age,
                family_size,
                f"${fare:.2f}",
                embarked,
                title
            ]
        })


        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


        st.info(
            "ℹ️ The prediction is based on the trained "
            "machine-learning model. It is a statistical "
            "prediction, not a historical certainty."
        )


    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    st.markdown(
        "<div class='glow-divider'></div>",
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='section-header'>📂 Batch Prediction</div>",
        unsafe_allow_html=True
    )


    if model_ready:

        if os.path.exists("test.csv"):

            test_raw = pd.read_csv(
                "test.csv"
            )


            # Reuse EXACT same feature engineering
            test_engineered = load_and_engineer(
                "test.csv"
            )


            # If test has no target, that's fine
            for column in FEATURE_COLS:

                if column not in test_engineered.columns:

                    test_engineered[column] = np.nan


            predictions = model.predict(
                test_engineered[
                    FEATURE_COLS
                ]
            )


            probabilities = model.predict_proba(
                test_engineered[
                    FEATURE_COLS
                ]
            )[:, 1]


            if "PassengerId" in test_raw.columns:

                passenger_ids = (
                    test_raw["PassengerId"]
                )

            else:

                passenger_ids = (
                    np.arange(
                        1,
                        len(predictions) + 1
                    )
                )


            prediction_df = pd.DataFrame({

                "PassengerId":
                    passenger_ids,

                "Survived":
                    predictions,

                "SurvivalProbability":
                    np.round(
                        probabilities,
                        4
                    )
            })


            st.dataframe(
                prediction_df.head(20),
                use_container_width=True
            )


            st.download_button(
                "📥 Download Predictions",
                prediction_df.to_csv(
                    index=False
                ),
                "submission.csv",
                "text/csv"
            )


            # Prediction distribution

            fig_hist = px.histogram(
                prediction_df,
                x="SurvivalProbability",
                color="Survived",
                nbins=30,
                color_discrete_map={
                    0: "#f87171",
                    1: "#34d399"
                },
                title=(
                    "Predicted Survival "
                    "Probability Distribution"
                )
            )


            fig_hist.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=400
            )


            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )


        else:

            st.info(
                "📁 Add `test.csv` to the same folder "
                "to enable batch predictions."
            )


# ============================================================
# PAGE 6 — PASSENGER EXPLORER
# ============================================================

elif page == "👤 Passenger Explorer":

    st.markdown(
        "<div class='section-header'>👤 Passenger Explorer</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Search and inspect individual Titanic passengers."
    )


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "🔎 Search by passenger name"
    )


    passenger_data = data.copy()


    if search.strip():

        passenger_data = passenger_data[
            passenger_data["Name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]


    # --------------------------------------------------------
    # STATUS FILTER
    # --------------------------------------------------------

    status_filter = st.multiselect(
        "Survival Status",
        ["Survived", "Perished"],
        default=[
            "Survived",
            "Perished"
        ]
    )


    status_map = {
        "Survived": 1,
        "Perished": 0
    }


    allowed_status = [
        status_map[x]
        for x in status_filter
    ]


    passenger_data = passenger_data[
        passenger_data["Survived"].isin(
            allowed_status
        )
    ]


    st.write(
        f"Showing **{len(passenger_data)}** passengers."
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    display_cols = [
        "PassengerId",
        "Name",
        "Sex",
        "Age",
        "Pclass",
        "Fare",
        "FamilySize",
        "Embarked",
        "Survived"
    ]


    available_cols = [
        c
        for c in display_cols
        if c in passenger_data.columns
    ]


    st.dataframe(
        passenger_data[
            available_cols
        ],
        use_container_width=True,
        height=450
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "📥 Download Passenger Results",
        passenger_data.to_csv(
            index=False
        ),
        "passenger_search_results.csv",
        "text/csv"
    )


    # --------------------------------------------------------
    # SELECT PASSENGER
    # --------------------------------------------------------

    if len(passenger_data) > 0:

        selected_index = st.selectbox(
            "Select a passenger to inspect",
            passenger_data.index,
            format_func=lambda idx:
                passenger_data.loc[
                    idx,
                    "Name"
                ]
        )


        passenger = passenger_data.loc[
            selected_index
        ]


        st.markdown(
            "<div class='section-header'>🎫 Passenger Details</div>",
            unsafe_allow_html=True
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Passenger Class",
            passenger["Pclass"]
        )

        c2.metric(
            "Age",
            f"{passenger['Age']:.0f}"
        )

        c3.metric(
            "Fare",
            f"${passenger['Fare']:.2f}"
        )

        c4.metric(
            "Survival",
            "✅ Survived"
            if passenger["Survived"] == 1
            else "💀 Perished"
        )


        st.write(
            f"**Name:** {passenger['Name']}"
        )

        st.write(
            f"**Sex:** {passenger['Sex']}"
        )

        st.write(
            f"**Family Size:** "
            f"{passenger['FamilySize']}"
        )

        st.write(
            f"**Embarked:** "
            f"{passenger['Embarked']}"
        )

    else:

        st.warning(
            "No passenger found."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<div class='glow-divider'></div>",
    unsafe_allow_html=True
)

st.caption(
    "🚢 Titanic Survival 3D Dashboard · "
    "Built with Streamlit, Plotly & Scikit-Learn"
)