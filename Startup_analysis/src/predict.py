import joblib
import pandas as pd

def load_model():
    model = joblib.load('output/model.pkl')
    encoder = joblib.load('output/label_encoder.pkl')
    return model, encoder

def predict_status(input_dict):
    model, encoder = load_model()
    
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Fill missing values
    df_input = df_input.fillna(0)

    # Predict
    pred = model.predict(df_input)[0]
    label = encoder.inverse_transform([pred])[0]
    return label
