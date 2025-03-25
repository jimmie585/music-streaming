if st.button("🎶 Predict Streams"):
    # Ensure new_data has all required columns
    new_data = pd.DataFrame(columns=X_train.columns)

    # Fill numerical features with input values
    new_data.loc[0, 'danceability_%'] = danceability
    new_data.loc[0, 'valence_%'] = valence
    new_data.loc[0, 'energy_%'] = energy
    new_data.loc[0, 'acousticness_%'] = acousticness

    # Fill other numerical features with default values
    for col in X_train.columns:
        if col not in ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'artist(s)_name']:
            new_data.loc[0, col] = 0  # Default for missing numerical features

    # Handle categorical feature (artist(s)_name)
    known_artists = df['artist(s)_name'].unique()
    if artist in known_artists:
        new_data.loc[0, 'artist(s)_name'] = artist
    else:
        new_data.loc[0, 'artist(s)_name'] = known_artists[0]  # Use first known artist to avoid error

    # Make prediction
    prediction = model.predict(new_data)
    st.write(f"### 🔥 Predicted Streams: **{int(prediction[0])}**")
