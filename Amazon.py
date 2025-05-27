import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter

# Streamlit page config (Must be the first Streamlit command)
st.set_page_config(page_title="Amazon Products Analysis", layout="wide")

st.title("📊 Amazon Products Data Analysis")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload your Amazon Products dataset for analysis.")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.subheader("🔍 Raw Data Preview")
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(), use_container_width=True)
    
    # Data Cleaning
    df.fillna(df.median(numeric_only=True), inplace=True)
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna("Unknown")
    
    df['discount_price'] = df['discount_price'].astype(str).str.replace("₹", "").str.replace(",", "").str.strip()
    df['actual_price'] = df['actual_price'].astype(str).str.replace("₹", "").str.replace(",", "").str.strip()
    df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce')
    df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
    
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')
    
    df['discount_percentage'] = ((df['actual_price'] - df['discount_price']) / df['actual_price']) * 100
    df['discount_percentage'] = df['discount_percentage'].fillna(0).clip(lower=0)  # Ensures no negative values
    df['discount_size'] = np.sqrt(df['discount_percentage'].clip(lower=0))  # Scale for better visualization
    
    df.dropna(subset=['discount_percentage', 'ratings'], inplace=True)
    
    st.subheader("🧹 Cleaned Data Preview")
    st.write(f"After cleaning: {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("📊 Countplot for Main Category vs Sub Category")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, y='main_category', palette='dark', ax=ax)
    ax.set_title("Countplot of Main Categories")
    st.pyplot(fig)
    
    st.subheader("⭐ Distribution of Ratings")
    fig = px.histogram(df, x='ratings', nbins=30, title="Distribution of Ratings", color_discrete_sequence=['green'])
    st.plotly_chart(fig)
    
    st.subheader("💰 Discount Percentage vs Ratings")
    fig = px.scatter(df, x='discount_percentage', y='ratings', title="Discount Percentage vs Ratings", color='ratings', size='discount_size', color_continuous_scale='plasma')
    st.plotly_chart(fig, use_container_width=True, key="discount_vs_ratings")
    
    st.subheader("🔢 Log Discounted Price vs Log Number of Ratings")
    df['log_discount_price'] = np.log1p(df['discount_price'])
    df['log_no_of_ratings'] = np.log1p(df['no_of_ratings'])
    fig = px.scatter(df, x='log_discount_price', y='log_no_of_ratings', title="Log Discounted Price vs Log Number of Ratings", color='log_no_of_ratings', color_continuous_scale='reds')
    st.plotly_chart(fig)
    
    st.subheader("🥧 Product Distribution by Main Category")
    fig = px.pie(df, names='main_category', title="Product Distribution by Category", hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig)
    
    st.subheader("📊 Main Category vs Number of Ratings vs Prices (Bar Plot)")
    category_group = df.groupby('main_category')['no_of_ratings'].agg(['mean', 'std']).reset_index()
    category_group.fillna(0, inplace=True)  # Ensure no NaN values in std
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=category_group, x='main_category', y='mean', palette='dark', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Bar Plot of Main Category vs Number of Ratings")
    st.pyplot(fig)
    
    st.subheader("📉 Scatter Plot: Discounted Price vs Number of Ratings")
    fig = px.scatter(df, x='discount_price', y='no_of_ratings', title="Discounted Price vs Number of Ratings", color='no_of_ratings', color_continuous_scale='viridis', opacity=0.6)
    st.plotly_chart(fig)
    
    st.subheader("🔍 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig = px.imshow(numeric_df.corr(), text_auto=True, aspect='auto', title="Correlation Heatmap", color_continuous_scale='rdylbu')
    st.plotly_chart(fig)
    
    
