# eda_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Job EDA Dashboard", layout="wide")

# === Load Data ===
@st.cache_data
def load_data():
    return pd.read_csv("data/data_output.csv")

df = load_data()

st.title("📊 Job Market EDA Dashboard")
st.markdown("Analisis eksploratif interaktif dari data pasar kerja — meliputi distribusi gaji, tingkat pendidikan, dan salary competitiveness index.")

# === Sidebar Filter ===
st.sidebar.header("🔍 Filter Data")
exp_filter = st.sidebar.multiselect("Filter berdasarkan Pengalaman Kerja:", sorted(df['experience_group'].dropna().unique()))
edu_filter = st.sidebar.multiselect("Filter berdasarkan Tingkat Pendidikan:", sorted(df['education_level'].dropna().unique()))
if exp_filter:
    df = df[df['experience_group'].isin(exp_filter)]
if edu_filter:
    df = df[df['education_level'].isin(edu_filter)]

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Salary Distribution",
    "🎓 Education & Experience",
    "🏢 Job Titles",
    "📈 Competitiveness Index",
    "🔥 Top Competitive Jobs"
])

# --- TAB 1: Salary Distribution ---
with tab1:
    st.header("💰 Distribusi Gaji dan Rentangnya")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Gaji (Histogram)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['salary'], bins=50, color='skyblue', edgecolor='black')
        ax.set_xlabel("Salary (Rp)")
        ax.set_ylabel("Frequency")
        ax.set_title("Salary Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot Gaji")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(df['salary'], vert=False)
        ax.set_xlabel("Gaji (Rp)")
        ax.set_title("Sebaran & Outlier Gaji")
        st.pyplot(fig)

    # Rentang Gaji
    st.subheader("Sebaran Gaji Berdasarkan Rentang")
    bins = [0, 2e6, 5e6, 10e6, 20e6, float('inf')]
    labels = ['< 2 juta', '2–5 juta', '5–10 juta', '10–20 juta', '> 20 juta']
    df['range_gaji'] = pd.cut(df['salary'], bins=bins, labels=labels, right=False)
    gaji_count = df['range_gaji'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(gaji_count.index, gaji_count.values, color='cornflowerblue', edgecolor='black')
    ax.set_xlabel("Rentang Gaji")
    ax.set_ylabel("Jumlah")
    ax.set_title("Sebaran Gaji Berdasarkan Rentang")
    st.pyplot(fig)

# --- TAB 2: Education & Experience ---
with tab2:
    st.header("🎓 Analisis Berdasarkan Pendidikan & Pengalaman")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Level Pendidikan")
        emp_dist = df['education_level'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(emp_dist.values, labels=emp_dist.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Educational Level Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Rata-rata Gaji Berdasarkan Pendidikan")
        avg_salary_edu = df.groupby('education_level')['salary'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(avg_salary_edu.index, avg_salary_edu.values, color='skyblue')
        ax.set_xticklabels(avg_salary_edu.index, rotation=45, ha='right')
        ax.set_ylabel("Rata-rata Gaji (Rp)")
        st.pyplot(fig)

    st.subheader("Rata-rata Gaji Berdasarkan Pengalaman")
    avg_salary_exp = df.groupby('experience_group')['salary'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(avg_salary_exp.index, avg_salary_exp.values, color='maroon')
    ax.set_xticklabels(avg_salary_exp.index, rotation=45)
    ax.set_ylabel("Rata-rata Gaji (Rp)")
    st.pyplot(fig)

# --- TAB 3: Job Titles ---
with tab3:
    st.header("🏢 Analisis Job Title")
    top_jobs = df['job_title'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='viridis', ax=ax)
    ax.set_xlabel("Jumlah")
    ax.set_ylabel("Job Title")
    ax.set_title("Top 10 Job Title Terbanyak")
    st.pyplot(fig)

# --- TAB 4: Salary Competitiveness Index ---
with tab4:
    st.header("📈 Salary Competitiveness Index Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Index")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['salary_competitiveness_index'], bins=30, edgecolor='black', kde=True, ax=ax)
        ax.axvline(100, color='red', linestyle='--', label='Market Rate')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot per Experience Group")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='experience_group', y='salary_competitiveness_index', palette='Set3', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Heatmap Salary Competitiveness per Lokasi")
    pivot_data = df.pivot_table(values='salary_competitiveness_index',
                                index='experience_group',
                                columns='location',
                                aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_data.iloc[:, :10], cmap='RdYlGn', annot=True, fmt=".1f", center=100, ax=ax)
    ax.set_title("Mean Competitiveness Index (Top 10 Locations)")
    st.pyplot(fig)

# --- TAB 5: Top Competitive Jobs ---
with tab5:
    st.header("🔥 Top Competitive & Non-Competitive Jobs")

    col1, col2 = st.columns(2)
    top_jobs = df.nlargest(15, 'salary_competitiveness_index')[['job_title', 'salary_competitiveness_index']]
    bottom_jobs = df.nsmallest(15, 'salary_competitiveness_index')[['job_title', 'salary_competitiveness_index']]

    with col1:
        st.subheader("Top 15 Most Competitive Jobs")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x='salary_competitiveness_index', y='job_title', data=top_jobs, color='green', ax=ax)
        ax.axvline(100, color='red', linestyle='--')
        st.pyplot(fig)

    with col2:
        st.subheader("Bottom 15 Least Competitive Jobs")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x='salary_competitiveness_index', y='job_title', data=bottom_jobs, color='red', ax=ax)
        ax.axvline(100, color='black', linestyle='--')
        st.pyplot(fig)
