# src/model_train.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from clean_data import clean_vc_data

def train_model():
    # Load and clean data
    df = clean_vc_data('../data/cleaned_investments_VC.csv')
    print("🧠 Using model_train.py from:", __file__)


    # Filter for binary classification
    df = df[df['status'].isin(['operating', 'closed'])]

    # Encode target
    label_encoder = LabelEncoder()
    df['status_encoded'] = label_encoder.fit_transform(df['status'])

    # Feature selection
    features = ['funding_total_usd', 'funding_rounds', 'funding_duration_days', 'is_in_us']
    X = df[features].fillna(0)
    y = df['status_encoded']

    # Train/test split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoder
    joblib.dump(model, 'output/model.pkl')
    joblib.dump(label_encoder, 'output/label_encoder.pkl')

    print("✅ Model trained and saved!")

# If run directly from CLI
if __name__ == "__main__":
    train_model()
