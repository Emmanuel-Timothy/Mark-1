from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Define preprocessing pipeline
categorical_features = ['External_Factors']
numerical_features = ['Gaming_Min', 'Social_Media_Min', 'Sleep_Min', 'Tutoring_Min']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Load the dataset
data = pd.read_csv('data/data.csv')

# Map 'Lateness_Frequency' to numeric values
data['Lateness_Frequency'] = data['Lateness_Frequency'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
})

# Define features and target variable
X = data.drop(columns=['Lateness_Frequency', 'Timestamp', 'Name', 'Grade'])
y = data['Lateness_Frequency']

# Apply preprocessing to features
X = preprocessor.fit_transform(X)

# Save the preprocessor
joblib.dump(preprocessor, './models/preprocessor.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, './models/lateness_model.pkl')