import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer EDA Dashboard", layout="wide")
st.title("🎀 Breast Cancer EDA Dashboard")
st.write("By Fatima Abbas | Exploratory Data Analysis of Wisconsin Breast Cancer Dataset")

# Load Data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

# Sidebar
st.sidebar.header("Filters")
diagnosis_filter = st.sidebar.multiselect(
    "Select Diagnosis",
    options=df['diagnosis'].unique(),
    default=df['diagnosis'].unique()
)
filtered_df = df[df['diagnosis'].isin(diagnosis_filter)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(filtered_df))
col2.metric("Benign Cases", len(filtered_df[filtered_df['diagnosis']=='Benign']))
col3.metric("Malignant Cases", len(filtered_df[filtered_df['diagnosis']=='Malignant']))
col4.metric("Total Features", 30)

# Charts
st.subheader("Diagnosis Distribution")
fig1 = px.pie(filtered_df, names='diagnosis', title='Benign vs Malignant', color='diagnosis', color_discrete_map={'Benign':'#2ecc71','Malignant':'#e74c3c'})
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Mean Radius by Diagnosis")
fig2 = px.box(filtered_df, x='diagnosis', y='mean radius', color='diagnosis', title='Mean Radius Distribution')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Mean Texture vs Mean Area")
fig3 = px.scatter(filtered_df, x='mean texture', y='mean area', color='diagnosis', title='Texture vs Area')
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Correlation - Top Features")
top_features = ['mean radius','mean texture','mean perimeter','mean area','mean smoothness']
fig4 = px.imshow(filtered_df[top_features].corr(), text_auto=True, title='Correlation Heatmap')
st.plotly_chart(fig4, use_container_width=True)

st.subheader("Raw Data Preview")
st.dataframe(filtered_df.head(20))
