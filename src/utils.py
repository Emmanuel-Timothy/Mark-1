import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Convert 'Lateness_Frequency' to numeric values
    data['Lateness_Frequency'] = data['Lateness_Frequency'].map({
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
    })
    
    # Fill missing values
    data.fillna(0, inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    features = data.drop(columns=['Lateness_Frequency'])
    scaled_features = scaler.fit_transform(features)
    
    return pd.DataFrame(scaled_features, columns=features.columns), data['Lateness_Frequency']

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def get_prediction_probabilities(model, input_data):
    probabilities = model.predict_proba(input_data)
    return probabilities.max(axis=1) * 100  # Return the highest probability as a percentage

def load_model(model_path):
    import joblib
    return joblib.load(model_path)