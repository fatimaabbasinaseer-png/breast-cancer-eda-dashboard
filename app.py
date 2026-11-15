import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Breast Cancer | EDA Studio",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- PREMIUM THEME --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #fff4fb 0%, #f1edff 48%, #eaf6ff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #38205f 0%, #673b91 52%, #9b5ca8 100%);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-style: italic !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.17);
    border-radius: 12px;
    padding: 9px 12px;
    margin: 4px 0;
    transition: .2s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.25);
    transform: translateX(4px);
}
.hero {
    padding: 28px 30px;
    border-radius: 24px;
    color: white;
    background: linear-gradient(110deg, #56317e, #a34f9c, #e68ab5);
    box-shadow: 0 12px 30px rgba(95, 53, 125, .20);
    margin-bottom: 22px;
}
.hero h1 {font-size: 34px; font-weight: 800; font-style: italic; margin: 0;}
.hero p {font-size: 15px; font-style: italic; margin: 8px 0 0 0; opacity: .95;}
.section-title {
    color: #56317e;
    font-size: 25px;
    font-weight: 800;
    font-style: italic;
    margin: 10px 0 15px 0;
}
.kpi {
    border-radius: 18px;
    padding: 18px 20px;
    min-height: 118px;
    color: white;
    box-shadow: 0 8px 22px rgba(67, 40, 100, .12);
}
.kpi .label {font-size: 13px; font-style: italic; opacity: .95;}
.kpi .value {font-size: 30px; font-weight: 800; margin-top: 5px;}
.kpi.pink {background: linear-gradient(135deg,#e66eaa,#bd4e91);}
.kpi.purple {background: linear-gradient(135deg,#8c62c7,#5a3b91);}
.kpi.blue {background: linear-gradient(135deg,#55a9d8,#3978bd);}
.kpi.teal {background: linear-gradient(135deg,#43b9b1,#278b9a);}
.soft-card {
    background: rgba(255,255,255,.82);
    border: 1px solid rgba(130, 87, 160, .13);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 7px 20px rgba(70, 45, 100, .07);
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #eadcf4;
    border-radius: 16px;
    padding: 14px;
}
h1, h2, h3, h4, p, label, .stMarkdown {
    font-style: italic;
}
.stButton button, .stDownloadButton button {
    border: 0;
    border-radius: 12px;
    color: white;
    background: linear-gradient(90deg,#70469c,#d16da7);
    font-weight: 600;
}
.stButton button:hover, .stDownloadButton button:hover {
    color: white;
    box-shadow: 0 5px 16px rgba(112,70,156,.25);
}
</style>
""", unsafe_allow_html=True)

# -------------------- DATA LOADING --------------------
@st.cache_data
def read_csv_file(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0) if hasattr(file, "seek") else None
        return pd.read_csv(file, encoding="latin-1")

def find_local_csv():
    preferred = [
        "Breast_Cancer (7).csv", "Breast_Cancer (6).csv",
        "Breast_Cancer.csv", "Breast_Cancer (2).csv",
        "breast_cancer.csv", "data.csv"
    ]
    for name in preferred:
        if os.path.exists(name):
            return name
    candidates = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    return candidates[0] if candidates else None

st.sidebar.markdown("# 🎀 EDA Studio")
st.sidebar.caption("✨ Interactive Dataset Dashboard")
uploaded = st.sidebar.file_uploader("📁 Upload Breast Cancer CSV", type=["csv"])

try:
    if uploaded is not None:
        raw_df = read_csv_file(uploaded)
        source_name = uploaded.name
    else:
        local_csv = find_local_csv()
        if local_csv:
            raw_df = read_csv_file(local_csv)
            source_name = local_csv
        else:
            st.markdown('<div class="hero"><h1>🎀 Breast Cancer EDA Studio</h1><p>Please upload your CSV file from the sidebar to begin.</p></div>', unsafe_allow_html=True)
            st.info("Upload the Breast Cancer CSV file using the left sidebar.")
            st.stop()
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

raw_df.columns = [str(c).strip() for c in raw_df.columns]
df = raw_df.copy()

# Identify likely ID columns; retain them but exclude from numeric analysis if clearly IDs
id_cols = [c for c in df.columns if c.lower() in ["id", "unnamed: 0", "index"]]
numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols_all if c not in id_cols]
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Detect common outcome/diagnosis columns without assuming a particular dataset schema
outcome_candidates = ["diagnosis", "target", "class", "label", "outcome"]
outcome_col = next((c for c in df.columns if c.lower().strip() in outcome_candidates), None)
status_col = next((c for c in df.columns if c.lower().strip() == "status"), None)

st.sidebar.success(f"Loaded: {source_name}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Dashboard Menu")
pages = [
    "🏠 Home & Data Overview",
    "📥 Load Data",
    "🔍 Understand Dataset",
    "🧹 Data Cleaning",
    "🧩 Feature Engineering",
    "🧑‍⚕️ Patient Records",
    "📊 Univariate Analysis",
    "📈 Bivariate Analysis",
    "🌐 Multivariate Analysis",
    "⏳ Time-Series Analysis",
    "💡 Business Questions",
    "🤖 Smart Dataset Assistant",
    "📌 Insights & Recommendations"
]
page = st.sidebar.radio("Choose a section", pages, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("🎈 Learn • Explore • Visualize")
st.sidebar.caption("Academic EDA project only")

# Optional interactive filters
filtered = df.copy()
with st.sidebar.expander("🔎 Data Filters", expanded=False):
    filter_cols = categorical_cols[:4]
    for col in filter_cols:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        if vals and len(vals) <= 40:
            chosen = st.multiselect(f"{col}", vals, default=[])
            if chosen:
                filtered = filtered[filtered[col].astype(str).isin(chosen)]
    if numeric_cols:
        col = st.selectbox("Numeric filter (optional)", ["None"] + numeric_cols)
        if col != "None" and filtered[col].notna().any():
            low, high = float(filtered[col].min()), float(filtered[col].max())
            if low < high:
                selected_range = st.slider("Select range", min_value=low, max_value=high, value=(low, high))
                filtered = filtered[filtered[col].between(selected_range[0], selected_range[1])]

# Shared chart palette
palette = ["#E76FA9", "#8061B5", "#55A7D5", "#48B8AE", "#F2B45B", "#8B78D0", "#E58C82", "#5A8FCE"]

def heading(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def kpi_card(label, value, color):
    st.markdown(
        f'<div class="kpi {color}"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True
    )

def show_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        font=dict(family="Poppins", color="#49345F"),
        title_font=dict(size=18, color="#56317e"),
        margin=dict(l=20, r=20, t=55, b=25),
        legend_title_text=""
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- HOME --------------------
if page == "🏠 Home & Data Overview":
    st.markdown('<div class="hero"><h1>🎀 Breast Cancer EDA Dashboard</h1><p>✨ Exploratory Data Analysis & Smart Dataset Assistant</p><p>Academic EDA project only — not a medical diagnostic tool.</p></div>', unsafe_allow_html=True)
    heading("📚 EDA Assignment Workflow")
    workflow = [
        "1. Raw Dataset & Load Data", "2. Understand Dataset", "3. Clean Data",
        "4. Transform / Feature Engineering", "5. Univariate Analysis", "6. Bivariate Analysis",
        "7. Multivariate Analysis", "8. Time-Series Analysis", "9. Business Questions",
        "10. Visualization", "11. Insights & Recommendations"
    ]
    st.markdown("<div class='soft-card'>" + "　 ➜　 ".join(workflow) + "</div>", unsafe_allow_html=True)
    st.markdown("")
    heading("📊 Data Overview")
    a, b, c, d = st.columns(4)
    with a: kpi_card("Total Patient Records", f"{len(filtered):,}", "pink")
    with b: kpi_card("Total Features", f"{filtered.shape[1]:,}", "purple")
    with c: kpi_card("Missing Values", f"{int(filtered.isna().sum().sum()):,}", "blue")
    with d: kpi_card("Duplicate Records", f"{int(filtered.duplicated().sum()):,}", "teal")

    # Show diagnosis categories only when the uploaded dataset actually contains them.
    if outcome_col:
        st.markdown("#### 🎗️ Diagnosis Category Summary")
        outcome_counts = filtered[outcome_col].astype(str).value_counts(dropna=False)
        diagnosis_cols = st.columns(min(3, max(1, len(outcome_counts))))
        for i, (category, count) in enumerate(outcome_counts.items()):
            diagnosis_cols[i % len(diagnosis_cols)].metric(str(category), f"{count:,}")
        fig_outcome = px.pie(
            names=outcome_counts.index.astype(str), values=outcome_counts.values,
            hole=0.48, title=f"{outcome_col} Distribution",
            color_discrete_sequence=palette
        )
        show_chart(fig_outcome)
    elif status_col:
        st.markdown("#### 📊 Patient Status Summary")
        status_counts = filtered[status_col].astype(str).value_counts(dropna=False)
        status_cols = st.columns(min(3, max(1, len(status_counts))))
        for i, (category, count) in enumerate(status_counts.items()):
            status_cols[i % len(status_cols)].metric(str(category), f"{count:,}")
        show_chart(px.pie(names=status_counts.index, values=status_counts.values,
                          hole=0.48, title=f"{status_col} Distribution",
                          color_discrete_sequence=palette))

    st.markdown("")
    left, right = st.columns([1.2, 1])
    with left:
        heading("🗂️ Dataset Preview")
        st.dataframe(filtered.head(12), use_container_width=True)
    with right:
        heading("🧾 Dataset Information")
        info = pd.DataFrame({
            "Column": filtered.columns,
            "Data Type": filtered.dtypes.astype(str).values,
            "Non-Null": filtered.notna().sum().values,
            "Missing": filtered.isna().sum().values,
            "Unique": filtered.nunique(dropna=True).values
        })
        st.dataframe(info, use_container_width=True, height=420)
    st.download_button("⬇️ Download Current Dataset", filtered.to_csv(index=False).encode("utf-8"),
                       "breast_cancer_eda_data.csv", "text/csv")

# -------------------- LOAD DATA --------------------
elif page == "📥 Load Data":
    heading("📥 Load Dataset")
    st.write("Upload the assignment CSV from the sidebar, or use the CSV file already saved in the project folder.")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Loaded file", source_name)
    with c2: st.metric("Rows loaded", f"{raw_df.shape[0]:,}")
    with c3: st.metric("Columns loaded", f"{raw_df.shape[1]:,}")
    st.markdown("#### First five rows of the raw dataset")
    st.dataframe(raw_df.head(), use_container_width=True)
    st.markdown("#### CSV loading status")
    st.success("Dataset loaded successfully and is ready for exploratory data analysis.")
    st.download_button("⬇️ Download Raw Dataset Copy", raw_df.to_csv(index=False).encode("utf-8"),
                       "breast_cancer_raw_copy.csv", "text/csv")

# -------------------- UNDERSTAND DATASET --------------------
elif page == "🔍 Understand Dataset":
    heading("🔍 Understand the Dataset")
    st.write("Review the dataset structure, column types, summary statistics, and unique values before cleaning and analysis.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rows", f"{df.shape[0]:,}")
    with c2: st.metric("Columns", f"{df.shape[1]:,}")
    with c3: st.metric("Numeric columns", len(numeric_cols))
    with c4: st.metric("Categorical columns", len(categorical_cols))
    st.markdown("#### Column information")
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Values": df.notna().sum().values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique(dropna=True).values
    })
    st.dataframe(column_info, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Numeric summary statistics")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns were detected.")
    with right:
        st.markdown("#### Categorical summary")
        if categorical_cols:
            cat_summary = pd.DataFrame({
                "Column": categorical_cols,
                "Unique Categories": [df[c].nunique(dropna=True) for c in categorical_cols],
                "Most Frequent Value": [df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else "—" for c in categorical_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No categorical columns were detected.")

# -------------------- CLEANING --------------------
elif page == "🧹 Data Cleaning":
    heading("🧹 Data Cleaning & Data Quality")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Missing Values by Column")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found.")
        else:
            fig = px.bar(x=missing.index, y=missing.values, labels={"x":"Column","y":"Missing values"},
                         color=missing.values, color_continuous_scale=["#8B78D0","#E76FA9"])
            show_chart(fig)
    with c2:
        st.markdown("#### Duplicate Records")
        dup = int(df.duplicated().sum())
        st.metric("Duplicate rows", dup)
        st.write("Duplicate rows are counted using all columns.")
        if dup == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning("Duplicate rows are present in the raw dataset.")
    st.markdown("#### Data Types")
    st.dataframe(pd.DataFrame({"Column":df.columns, "Data Type":df.dtypes.astype(str).values}), use_container_width=True)
    st.markdown("#### Cleaned Dataset Preview")
    clean = df.drop_duplicates().copy()
    for col in clean.select_dtypes(include=["object"]).columns:
        clean[col] = clean[col].astype("string").str.strip()
    st.caption("Preview applies duplicate removal and trims extra spaces in text columns. Original CSV remains unchanged.")
    st.dataframe(clean.head(20), use_container_width=True)
    st.download_button("⬇️ Download Cleaned CSV", clean.to_csv(index=False).encode("utf-8"),
                       "breast_cancer_cleaned.csv", "text/csv")

# -------------------- FEATURE ENGINEERING --------------------
elif page == "🧩 Feature Engineering":
    heading("🧩 Feature Engineering")
    engineered = df.copy()
    if "Age" in engineered.columns:
        engineered["Age Group"] = pd.cut(engineered["Age"], bins=[0, 30, 45, 60, 100],
                                           labels=["Under 31", "31–45", "46–60", "Above 60"])
        st.success("Created Age Group from Age.")
    if "Tumor Size" in engineered.columns:
        engineered["Tumor Size Group"] = pd.cut(engineered["Tumor Size"], bins=[-1, 20, 50, np.inf],
                                                  labels=["Small (≤20)", "Medium (21–50)", "Large (>50)"])
        st.success("Created Tumor Size Group from Tumor Size.")
    if "Survival Months" in engineered.columns:
        engineered["Survival Period"] = pd.cut(engineered["Survival Months"], bins=[-1, 12, 60, np.inf],
                                                 labels=["Up to 12 months", "13–60 months", "Over 60 months"])
        st.success("Created Survival Period from Survival Months.")
    if len(engineered.columns) == len(df.columns):
        st.info("No recognized columns for the optional example features were found. Your original columns are preserved.")
    st.dataframe(engineered.head(20), use_container_width=True)
    st.download_button("⬇️ Download Engineered Dataset", engineered.to_csv(index=False).encode("utf-8"),
                       "breast_cancer_feature_engineered.csv", "text/csv")

# -------------------- PATIENT RECORDS --------------------
elif page == "🧑‍⚕️ Patient Records":
    heading("🧑‍⚕️ Patient Records")
    st.write("Search, filter, inspect, and download records from the uploaded CSV.")
    st.metric("Total Patient Records in Original Dataset", f"{len(df):,}")
    search_text = st.text_input("🔎 Search patient records", placeholder="Type a value, category, ID, or keyword...")
    records = filtered.copy()
    if search_text.strip():
        match_mask = records.astype(str).apply(
            lambda column: column.str.contains(search_text.strip(), case=False, na=False, regex=False)
        ).any(axis=1)
        records = records.loc[match_mask]
    st.caption(f"Records matching current filters/search: {len(records):,}")
    st.dataframe(records, use_container_width=True, height=430)
    if not records.empty:
        st.markdown("#### View one complete record")
        selected_index = st.selectbox(
            "Choose record row",
            options=records.index.tolist(),
            format_func=lambda idx: f"Record {idx + 1}"
        )
        record_data = records.loc[selected_index]
        detail = pd.DataFrame({"Field": record_data.index, "Value": record_data.values})
        st.dataframe(detail, use_container_width=True, hide_index=True)
    else:
        st.info("No records match the current search and filters.")
    st.download_button(
        "⬇️ Download Patient Records",
        records.to_csv(index=False).encode("utf-8"),
        "breast_cancer_patient_records.csv",
        "text/csv"
    )

# -------------------- UNIVARIATE --------------------
elif page == "📊 Univariate Analysis":
    heading("📊 Univariate Analysis")
    if not numeric_cols and not categorical_cols:
        st.info("No analyzable columns found.")
    else:
        col = st.selectbox("Choose a feature", numeric_cols + categorical_cols)
        if col in numeric_cols:
            fig = px.histogram(filtered, x=col, nbins=30, marginal="box",
                               color_discrete_sequence=[palette[0]], title=f"Distribution of {col}")
            show_chart(fig)
            st.write(filtered[col].describe().to_frame("Summary"))
        else:
            counts = filtered[col].astype(str).value_counts().reset_index()
            counts.columns = [col, "Count"]
            fig = px.bar(counts, x=col, y="Count", color=col, color_discrete_sequence=palette,
                         title=f"Category Count: {col}")
            show_chart(fig)
            st.dataframe(counts, use_container_width=True)

# -------------------- BIVARIATE --------------------
elif page == "📈 Bivariate Analysis":
    heading("📈 Bivariate Analysis")
    x_options = numeric_cols + categorical_cols
    if len(x_options) < 2:
        st.info("At least two usable columns are needed.")
    else:
        x = st.selectbox("Select X feature", x_options, index=0)
        y_choices = [c for c in x_options if c != x]
        y = st.selectbox("Select Y feature", y_choices)
        if x in numeric_cols and y in numeric_cols:
            fig = px.scatter(filtered, x=x, y=y, color=categorical_cols[0] if categorical_cols else None,
                             color_discrete_sequence=palette, title=f"{x} vs {y}")
        elif x in categorical_cols and y in numeric_cols:
            fig = px.box(filtered, x=x, y=y, color=x, color_discrete_sequence=palette, title=f"{y} by {x}")
        elif x in numeric_cols and y in categorical_cols:
            fig = px.box(filtered, x=y, y=x, color=y, color_discrete_sequence=palette, title=f"{x} by {y}")
        else:
            ct = pd.crosstab(filtered[x].astype(str), filtered[y].astype(str))
            fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                            title=f"{x} vs {y}")
        show_chart(fig)

# -------------------- MULTIVARIATE --------------------
elif page == "🌐 Multivariate Analysis":
    heading("🌐 Multivariate Analysis")
    if len(numeric_cols) >= 2:
        corr = filtered[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="PuRd",
                        title="Correlation Heatmap")
        show_chart(fig)
        st.caption("Correlation shows association, not cause and effect.")
    else:
        st.info("At least two numeric columns are needed for a correlation heatmap.")
    if len(numeric_cols) >= 3:
        st.markdown("#### Explore Three Numeric Features")
        x = st.selectbox("X axis", numeric_cols, key="multi_x")
        y = st.selectbox("Y axis", [c for c in numeric_cols if c != x], key="multi_y")
        size_opts = [c for c in numeric_cols if c not in [x, y]]
        size = st.selectbox("Bubble size", ["None"] + size_opts, key="multi_size")
        color = st.selectbox("Color grouping", ["None"] + categorical_cols, key="multi_color")
        fig = px.scatter(filtered, x=x, y=y, size=size if size != "None" else None,
                         color=color if color != "None" else None,
                         color_discrete_sequence=palette, title="Multivariable Relationship")
        show_chart(fig)

# -------------------- TIME SERIES --------------------
elif page == "⏳ Time-Series Analysis":
    heading("⏳ Time-Series Analysis")
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "time" in c.lower()]
    if date_cols:
        date_col = st.selectbox("Select date/time column", date_cols)
        try:
            ts = df.copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col]).sort_values(date_col)
            if ts.empty:
                st.warning("The selected column could not be interpreted as dates.")
            elif not numeric_cols:
                st.info("No numeric measure is available for a time-series chart.")
            else:
                value_col = st.selectbox("Select numeric value", numeric_cols)
                trend = ts.set_index(date_col)[value_col].resample("M").mean().reset_index()
                fig = px.line(trend, x=date_col, y=value_col, markers=True, title=f"{value_col} over time")
                show_chart(fig)
        except Exception as e:
            st.warning(f"Could not create time series: {e}")
    else:
        st.info("No calendar date or year column was found in this dataset. A true time-series trend cannot be calculated from the available fields.")
        st.write("Survival Months is a duration field, not a calendar date. It can be explored in Univariate or Bivariate Analysis, but it should not be presented as a calendar time series.")

# -------------------- BUSINESS QUESTIONS --------------------
elif page == "💡 Business Questions":
    heading("💡 Dataset Questions & Answers")
    questions = {
        "How many records and features are in the dataset?": f"The currently loaded dataset has {df.shape[0]:,} records and {df.shape[1]:,} features.",
        "How many missing values are present?": f"There are {int(df.isna().sum().sum()):,} missing cells in the raw dataset.",
        "How many duplicate records are present?": f"There are {int(df.duplicated().sum()):,} fully duplicated rows.",
    }
    if status_col:
        status_counts = df[status_col].astype(str).value_counts()
        questions[f"What are the counts for each {status_col} category?"] = "; ".join([f"{k}: {v}" for k, v in status_counts.items()])
    if outcome_col:
        outcome_counts = df[outcome_col].astype(str).value_counts()
        questions[f"What are the counts for each {outcome_col} category?"] = "; ".join([f"{k}: {v}" for k, v in outcome_counts.items()])
    if "Age" in df.columns:
        questions["What is the average Age?"] = f"The average Age is {df['Age'].mean():.1f} years (where available)."
    if "Tumor Size" in df.columns:
        questions["What is the average Tumor Size?"] = f"The average Tumor Size is {df['Tumor Size'].mean():.1f} (dataset units)."
    if "Survival Months" in df.columns:
        questions["What is the average Survival Months?"] = f"The average recorded Survival Months is {df['Survival Months'].mean():.1f} months."
    for q, ans in questions.items():
        with st.expander("❓ " + q):
            st.write(ans)
    st.caption("These are descriptive dataset summaries, not medical advice or predictions.")

# -------------------- SMART ASSISTANT --------------------
elif page == "🤖 Smart Dataset Assistant":
    heading("🤖 Smart Dataset Assistant")
    st.write("Ask a question about the loaded dataset. Answers are calculated from the CSV columns.")
    question = st.selectbox("Choose a question", [
        "How many rows and columns are there?",
        "Which columns have missing values?",
        "How many duplicate rows are there?",
        "Show summary statistics for numeric columns.",
        "Which category appears most often in the outcome/status column?",
        "What is the average Age?",
        "What is the average Tumor Size?",
        "What is the average Survival Months?"
    ])
    if st.button("✨ Generate Answer"):
        if question == "How many rows and columns are there?":
            st.success(f"The dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
        elif question == "Which columns have missing values?":
            m = df.isna().sum()
            m = m[m > 0]
            if m.empty: st.success("There are no missing values.")
            else: st.dataframe(m.rename("Missing Values").to_frame())
        elif question == "How many duplicate rows are there?":
            st.success(f"{int(df.duplicated().sum())} duplicate rows were found.")
        elif question == "Show summary statistics for numeric columns.":
            st.dataframe(df.describe().T, use_container_width=True)
        elif question == "Which category appears most often in the outcome/status column?":
            selected_outcome = outcome_col or status_col
            if selected_outcome and not df[selected_outcome].mode().empty:
                st.success(f"Most frequent {selected_outcome}: {df[selected_outcome].mode().iloc[0]}")
            else:
                st.info("No diagnosis or status column was detected in this dataset.")
        elif question == "What is the average Age?":
            if "Age" in df.columns: st.success(f"Average Age: {df['Age'].mean():.2f}")
            else: st.info("An Age column was not found.")
        elif question == "What is the average Tumor Size?":
            if "Tumor Size" in df.columns: st.success(f"Average Tumor Size: {df['Tumor Size'].mean():.2f}")
            else: st.info("A Tumor Size column was not found.")
        elif question == "What is the average Survival Months?":
            if "Survival Months" in df.columns: st.success(f"Average Survival Months: {df['Survival Months'].mean():.2f}")
            else: st.info("A Survival Months column was not found.")
    st.info("Assistant answers describe this dataset only. They do not diagnose, predict, or recommend treatment.")

# -------------------- INSIGHTS --------------------
elif page == "📌 Insights & Recommendations":
    heading("📌 Insights & Recommendations")
    st.markdown("#### Dataset Quality")
    st.write(f"• Records: **{len(df):,}**")
    st.write(f"• Features: **{df.shape[1]:,}**")
    st.write(f"• Missing cells: **{int(df.isna().sum().sum()):,}**")
    st.write(f"• Duplicate rows: **{int(df.duplicated().sum()):,}**")
    if outcome_col:
        st.markdown(f"#### {outcome_col} Category Summary")
        st.dataframe(df[outcome_col].astype(str).value_counts().rename("Records").to_frame())
        st.caption("Categories are shown exactly as stored in the dataset; no medical classification is inferred.")
    elif status_col:
        st.markdown(f"#### {status_col} Summary")
        st.dataframe(df[status_col].astype(str).value_counts().rename("Records").to_frame())
    st.markdown("#### Numeric Feature Summary")
    if numeric_cols:
        summary = df[numeric_cols].describe().T
        st.dataframe(summary, use_container_width=True)
        st.markdown("#### Suggested EDA Recommendations")
        st.write("• Review missing values and duplicate records before analysis.")
        st.write("• Compare numeric distributions and category counts.")
        st.write("• Use correlation and grouped comparisons to explore relationships.")
        st.write("• Avoid interpreting association as causation.")
        st.write("• Do not use this educational dashboard for individual medical decisions.")
    else:
        st.info("No numeric columns were detected for numeric summary.")

st.markdown("---")
st.markdown("<center><i>🎀 Breast Cancer EDA Studio | Academic project only — not a medical diagnostic tool.</i></center>", unsafe_allow_html=True)
