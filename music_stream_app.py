import os
import subprocess

# Force install scikit-learn if missing
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "--no-cache-dir", "scikit-learn"])


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# ---- Project Overview ----
st.title("🎵 Spotify Stream Prediction Dashboard")
st.markdown("""
## 📌 Project Overview  
This application predicts the number of **streams for songs** based on various features such as **danceability, energy, playlists, and charts rankings**.  

## 🚨 Problem Statement  
Music industry professionals struggle to estimate the potential popularity of songs.  
This dashboard helps **artists, producers, and streaming platforms** analyze song trends and make data-driven decisions.
""")

# Load dataset from CSV
DATA_PATH= "https://raw.githubusercontent.com/jimmie585/music-streaming/refs/heads/main/Spotify%20Most%20Streamed%20Songs.csv"
df=pd.read_csv(DATA_PATH)
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split


# Ensure "streams" column is numeric
if "streams" in df.columns:
    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df = df.dropna(subset=["streams"])  # Remove NaN values
    df["streams"] = df["streams"].astype(int)

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

# Encode categorical variables
categorical_features = ['artist(s)_name']

# Preprocessing pipeline
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

# ---- Model Evaluation ----
st.subheader("📊 Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R2):** {r2:.2f}")

# ---- Prediction Section ----
st.subheader("🎯 Predict Streams for a New Song")

st.write("Enter the features for prediction:")
danceability = st.number_input("Danceability (%)", min_value=0, max_value=100, value=50)
valence = st.number_input("Valence (%)", min_value=0, max_value=100, value=50)
energy = st.number_input("Energy (%)", min_value=0, max_value=100, value=50)
acousticness = st.number_input("Acousticness (%)", min_value=0, max_value=100, value=50)
instrumentalness = st.number_input("Instrumentalness (%)", min_value=0, max_value=100, value=50)
liveness = st.number_input("Liveness (%)", min_value=0, max_value=100, value=50)
speechiness = st.number_input("Speechiness (%)", min_value=0, max_value=100, value=50)
bpm = st.number_input("Beats Per Minute (BPM)", min_value=0, max_value=200, value=120)
spotify_playlists = st.number_input("Spotify Playlists", min_value=0, value=10)
spotify_charts = st.number_input("Spotify Charts", min_value=0, value=10)
apple_playlists = st.number_input("Apple Playlists", min_value=0, value=10)
apple_charts = st.number_input("Apple Charts", min_value=0, value=10)
deezer_playlists = st.number_input("Deezer Playlists", min_value=0, value=10)
deezer_charts = st.number_input("Deezer Charts", min_value=0, value=10)
shazam_charts = st.number_input("Shazam Charts", min_value=0, value=10)
artist = st.selectbox("Artist", df["artist(s)_name"].unique())

# Create a DataFrame for the new input
new_data = pd.DataFrame({
    'danceability_%': [danceability],
    'valence_%': [valence],
    'energy_%': [energy],
    'acousticness_%': [acousticness],
    'instrumentalness_%': [instrumentalness],
    'liveness_%': [liveness],
    'speechiness_%': [speechiness],
    'bpm': [bpm],
    'in_spotify_playlists': [spotify_playlists],
    'in_spotify_charts': [spotify_charts],
    'in_apple_playlists': [apple_playlists],
    'in_apple_charts': [apple_charts],
    'in_deezer_playlists': [deezer_playlists],
    'in_deezer_charts': [deezer_charts],
    'in_shazam_charts': [shazam_charts],
    'artist(s)_name': [artist]
})

# Predict streams for the new data
if st.button("🎶 Predict Streams"):
    prediction = model.predict(new_data)
    st.write(f"### 🔥 Predicted Streams: **{int(prediction[0])}**")

# ---- Footer ----
st.markdown("---")
st.markdown("""
**👨‍💻 Created by James Ndungu**  
📧 Email: [jamesndungu.dev@gmail.com](mailto:jamesndungu.dev@gmail.com)  
📞 Phone: +254796593045  
🔗 GitHub: [James' GitHub](https://github.com/jimmie585)
""")




