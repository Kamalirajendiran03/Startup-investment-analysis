import pandas as pd

def clean_vc_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean funding amount
    df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(',', '')
    df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')

    # Date parsing
    date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Feature engineering
    df['funding_duration_days'] = (df['last_funding_at'] - df['first_funding_at']).dt.days
    df['is_in_us'] = df['country_code'].apply(lambda x: 1 if x == 'USA' else 0)

    # Drop rows with missing target 'status'
    df = df.dropna(subset=['status'])

    return df
