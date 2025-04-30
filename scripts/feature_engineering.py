import pandas as pd
import numpy as np

def add_features(df):
    # Age calculation
    if 'DOB' in df.columns:
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        df['Age'] = 2016 - df['DOB'].dt.year
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Chronic condition count
    chronic_cols = [col for col in df.columns if 'ChronicCond_' in col]
    df['ChronicCount'] = df[chronic_cols].apply(lambda row: sum(row == 1), axis=1)

    # Gender (encode if not already numeric)
    if df['Gender'].dtype == object:
        df['Gender'] = df['Gender'].astype('category').cat.codes

    # Fill missing numeric values with 0
    df.fillna(0, inplace=True)

    # Drop unused columns
    df.drop(['ClaimID', 'BeneID', 'DOB', 'DOD'], axis=1, inplace=True, errors='ignore')

    return df

def load_and_engineer(input_path='data/processed_train.csv', output_path='data/model_ready.csv'):
    df = pd.read_csv(input_path)
    df_fe = add_features(df)
    df_fe.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Output saved to {output_path}")
    return df_fe

if __name__ == "__main__":
    load_and_engineer()
