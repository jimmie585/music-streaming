import os
import subprocess

# Ensure scikit-learn is installed
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "--no-cache-dir", "scikit-learn"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# ---- 🎵 Project Overview ----
st.title("🎵 Spotify Stream Prediction Dashboard")
st.markdown("""
## 📌 Overview  
This **interactive dashboard** analyzes **Spotify's most streamed songs** and uses **Machine Learning** to **predict the number of streams** for new songs based on different features.  

## 🚨 Problem Statement  
**Music industry professionals** (artists, producers, and streaming platforms) often struggle to estimate a song's potential popularity.  
This dashboard **helps analyze song trends** and makes **data-driven predictions** for better decision-making.
""")

# ---- 📂 Load Dataset ----
st.markdown("## 📂 Spotify Dataset")
st.write("""
This dataset contains **song attributes** such as **danceability, energy, acousticness, playlists, and chart rankings**.  
The app uses these features to **predict the number of streams** a song is likely to get.
""")

DATA_PATH = "https://raw.githubusercontent.com/jimmie585/music-streaming/main/Spotify%20Most%20Streamed%20Songs.csv"
df = pd.read_csv(DATA_PATH)

# Ensure "streams" column is numeric
if "streams" in df.columns:
    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df = df.dropna(subset=["streams"])  # Remove NaN values
    df["streams"] = df["streams"].astype(int)

# ---- 📊 Feature Selection ----
st.markdown("## 📊 Features Used in the Prediction Model")
st.write("""
The model analyzes the following **song features** to predict the number of streams:
- **Danceability (%)** – How danceable a track is.
- **Valence (%)** – How positive/happy a song sounds.
- **Energy (%)** – The intensity and activity level of a track.
- **Acousticness (%)** – The likelihood of a song being acoustic.
- **Instrumentalness (%)** – Presence of vocals in a track.
- **Liveness (%)** – Detects audience sounds in live performances.
- **Speechiness (%)** – Detects spoken words in songs (e.g., rap).
- **BPM (Beats Per Minute)** – Tempo of the track.
- **Number of Playlists & Charts Appearances** – Measures popularity across streaming platforms.
""")

# Feature Selection
features = [
    'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
    'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm',
    'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
    'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts',
    'in_shazam_charts', 'artist(s)_name'
]
target = 'streams'

# Filter the dataset
df = df[features + [target]]

# Clean numerical columns: Remove commas and convert to numeric
numerical_features = [col for col in features if col != 'artist(s)_name']
for col in numerical_features:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(",", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
df = df.dropna()

# ---- 📊 Data Visualizations ----
st.markdown("## 📊 Data Visualizations")

# Top 10 Most Streamed Songs
st.subheader("🔥 Top 10 Most Streamed Songs")
top_songs = df.nlargest(10, "streams")[["artist(s)_name", "streams"]]
st.bar_chart(top_songs.set_index("artist(s)_name"))

# Feature Correlation with Streams (Fixed)
st.subheader("📈 Feature Correlation with Streams")
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns

if "streams" in numeric_df.columns:
    correlation = numeric_df.corr()["streams"].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=correlation.index, y=correlation.values, ax=ax, palette="coolwarm")
    plt.xticks(rotation=90)
    st.pyplot(fig)
else:
    st.warning("⚠️ No numeric features available for correlation.")

# Distribution of Streams
st.subheader("📊 Distribution of Streams")
fig, ax = plt.subplots()
sns.histplot(df["streams"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# ---- 🚀 Machine Learning Model (Random Forest) ----
st.markdown("## 🚀 How the Prediction Model Works")
st.write("""
We use **Random Forest Regression**, a machine learning algorithm that:
- **Learns from past hit songs** and their features.
- **Predicts the number of streams** for new songs based on their attributes.
""")

# Preprocessing pipeline
categorical_features = ['artist(s)_name']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---- 📊 Model Evaluation ----
st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R2):** {r2:.2f}")

# ---- 🎯 Make Predictions ----
st.subheader("🎯 Predict Streams for a New Song")
st.write("**Adjust the song features below** and click **Predict Streams** to estimate its popularity.")

# Input fields for user prediction
danceability = st.slider("Danceability (%)", 0, 100, 50)
valence = st.slider("Valence (%)", 0, 100, 50)
energy = st.slider("Energy (%)", 0, 100, 50)
acousticness = st.slider("Acousticness (%)", 0, 100, 50)

if st.button("🎶 Predict Streams"):
    new_data = pd.DataFrame({'danceability_%': [danceability], 'valence_%': [valence], 'energy_%': [energy], 'acousticness_%': [acousticness]})
    prediction = model.predict(new_data)
    st.write(f"### 🔥 Predicted Streams: **{int(prediction[0])}**")

st.markdown("---")
st.markdown("👨‍💻 **Created by James Ndungu** | 📧 [Email](mailto:jamesndungu.dev@gmail.com) | 🔗 [GitHub](https://github.com/jimmie585)")
