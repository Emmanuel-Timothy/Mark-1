import os
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model and preprocessing pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/lateness_model.pkl')
model = load(MODEL_PATH)

PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '../models/preprocessor.pkl')
preprocessor = load(PREPROCESSOR_PATH)

# Validate the loaded model
if not hasattr(model, 'predict'):
    raise ValueError("Loaded object is not a valid scikit-learn model.")

# Map numeric predictions back to categorical labels
lateness_mapping = {
    0: "Never",
    1: "Rarely",
    2: "Sometimes",
    3: "Often",
    4: "Always"
}

def predict_lateness(features):
    """
    Predict lateness frequency and confidence percentage based on input features.
    """
    # Convert features to DataFrame
    input_data = pd.DataFrame([features])

    # Apply preprocessing
    input_data = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(input_data)
        confidence_percentage = np.max(prediction_proba[0]) * 100
    else:
        confidence_percentage = 100.0  # Assume 100% confidence if probabilities are unavailable

    # Get the predicted class and map it to the categorical label
    predicted_class = lateness_mapping[prediction[0]]

    return predicted_class, confidence_percentage

if __name__ == "__main__":
    try:
        # Example input features with validation
        user_input = {}
        for feature in ['Gaming_Min', 'Social_Media_Min', 'Sleep_Min', 'Tutoring_Min']:
            while True:
                try:
                    value = float(input(f"Enter {feature.replace('_', ' ')} (in minutes): "))
                    user_input[feature] = value
                    break
                except ValueError:
                    print(f"Invalid input for {feature}. Please enter a numeric value.")

        # Ask for Has_Checklist input
        while True:
            has_checklist = input("Do you have a checklist? (yes/no): ").strip().lower()
            if has_checklist in ['yes', 'no']:
                user_input['Has_Checklist'] = 1 if has_checklist == 'yes' else 0
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        # Ask for reasons
        reasons = ['Reason_Lazyness', 'Reason_Forgetfulness', 'Reason_Extracurriculars', 'Reason_TightDeadlines']
        for reason in reasons:
            while True:
                reason_input = input(f"Is {reason.replace('_', ' ')} a factor? (yes/no): ").strip().lower()
                if reason_input in ['yes', 'no']:
                    user_input[reason] = 1 if reason_input == 'yes' else 0
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")

        # Ask for External_Factors input
        external_factors_options = [
            "None", "Busy", "Dislikes school", "Eating", "Sports", "Health issues",
            "Late night activities", "Laziness", "College preparation",
            "Environmental distractions", "Socializing", "Unclear", "Philosophical",
            "Stable", "Activities", "Community", "Life motivation"
        ]
        print("\nSelect External Factors (you can choose multiple, separated by commas):")
        for i, factor in enumerate(external_factors_options):
            print(f"{i}. {factor}")
        
        while True:
            try:
                selected_factors = input("Enter the numbers corresponding to the factors (e.g., 1,3,5): ")
                selected_indices = [int(x.strip()) for x in selected_factors.split(",")]
                if 0 in selected_indices:  # If "None" is selected, set External_Factors to "None"
                    user_input['External_Factors'] = "None"
                else:
                    user_input['External_Factors'] = ", ".join(
                        [external_factors_options[i] for i in selected_indices if i > 0]
                    )
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid numbers separated by commas.")

        predicted_class, confidence = predict_lateness(user_input)
        print(f"\nPredicted Lateness Frequency: {predicted_class}, with confidence of {confidence:.2f}%")
    except Exception as e:
        print(f"An error occurred: {e}")