import pandas as pd
import numpy as np

def add_features(df):
    # Convert DOB to Age if exists
    if 'DOB' in df.columns:
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        df['Age'] = 2016 - df['DOB'].dt.year
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df.drop(columns=['DOB'], inplace=True)

    # Convert ClaimStartDt and ClaimEndDt to numeric parts
    for date_col in ['ClaimStartDt', 'ClaimEndDt']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[date_col + '_year'] = df[date_col].dt.year.fillna(0).astype(int)
            df[date_col + '_month'] = df[date_col].dt.month.fillna(0).astype(int)
            df[date_col + '_day'] = df[date_col].dt.day.fillna(0).astype(int)
            df.drop(columns=[date_col], inplace=True)

    # Vectorized chronic condition count
    chronic_cols = [col for col in df.columns if 'ChronicCond_' in col]
    if chronic_cols:
        df['ChronicCount'] = df[chronic_cols].eq(1).sum(axis=1)
    else:
        df['ChronicCount'] = 0

    # Encode Gender if needed
    if 'Gender' in df.columns and df['Gender'].dtype == object:
        df['Gender'] = df['Gender'].astype('category').cat.codes

    # Drop heavy or irrelevant columns if present
    drop_cols = ['ClaimID', 'BeneID', 'DOD', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Encode remaining categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [col for col in cat_cols if col != 'PotentialFraud']
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df.fillna(0, inplace=True)
    return df

import time  # Add this at the top of your file if not already there

def load_and_engineer(input_path='data/processed_train.csv', output_path='data/model_ready.csv'):
    chunk_size = 25000  # Smaller chunk for smoother run
    chunk_count = 0
    first_chunk = True

    print("⏳ Loading and processing in chunks...")

    for chunk in pd.read_csv(input_path, low_memory=False, chunksize=chunk_size):
        chunk_count += 1
        print(f"\n👉 Processing chunk {chunk_count}...")
        
        start_time = time.time()  # START TIMER

        chunk_fe = add_features(chunk)

        # Save processed chunk directly to disk
        if first_chunk:
            chunk_fe.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk_fe.to_csv(output_path, index=False, mode='a', header=False)

        elapsed = time.time() - start_time  # END TIMER
        print(f"✅ Chunk {chunk_count} processed in {elapsed:.2f} seconds")

    print(f"\n🎯 Feature engineering complete. Output saved to {output_path}")


if __name__ == "__main__":
    load_and_engineer()
