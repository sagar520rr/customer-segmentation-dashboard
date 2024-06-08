import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from flask import Flask, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/api/segment-profiles')
def get_segment_profiles():
    try:
        # Load the data
        data = pd.read_csv('customer_data.csv', delimiter='\t')
        data.columns = data.columns.str.strip()

        # Clean and convert numeric features
        for feature in data.columns:
            if feature not in ['Education', 'Marital_Status']:
                data[feature] = data[feature].astype(str).str.extract(r'(\d+)', expand=False)
                data[feature] = pd.to_numeric(data[feature], errors='coerce')

        # Fill missing values for all columns with appropriate defaults
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col].fillna('Unknown', inplace=True)
            else:
                data[col].fillna(0, inplace=True)

        # One-hot encode categorical features
        data_encoded = pd.get_dummies(data, columns=['Education', 'Marital_Status'])

        # Standardize numeric features
        numeric_features = [col for col in data.columns if col not in ['Education', 'Marital_Status']]
        scaler = StandardScaler()
        data_encoded[numeric_features] = scaler.fit_transform(data_encoded[numeric_features])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        data['cluster'] = kmeans.fit_predict(data_encoded)

        # Prepare profiles for API response, focusing only on specified features
        response_features = ['Year_Birth', 'Education', 'Marital_Status', 'Income']
        profiles = data.groupby('cluster').mean(numeric_only=True).to_dict(orient='index')

        # Add back categorical data (mode)
        for feature in ['Education', 'Marital_Status']:
            mode_data = data.groupby('cluster')[feature].agg(lambda x: x.mode()[0])
            for cluster, value in mode_data.items():
                profiles[cluster][feature] = value

        # Filter profiles to include only the response features
        profiles_filtered = {}
        for cluster, profile in profiles.items():
            profiles_filtered[cluster] = {feature: profile.get(feature, 'Unknown') for feature in response_features}

        # Convert cluster keys to strings for JSON serialization
        profiles_filtered = {str(k): v for k, v in profiles_filtered.items()}

        logging.debug("Profiles: %s", profiles_filtered)
        return jsonify(profiles_filtered)
    except Exception as e:
        logging.error("Error: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
