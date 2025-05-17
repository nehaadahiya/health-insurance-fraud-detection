import pandas as pd
import sqlite3
import os
from pathlib import Path

def extract_csvs(base_path):
    files = {
        "train": "Train-1542865627584.csv",
        "train_beneficiary": "Train_Beneficiarydata-1542865627584.csv",
        "train_inpatient": "Train_Inpatientdata-1542865627584.csv",
        "train_outpatient": "Train_Outpatientdata-1542865627584.csv"
    }
    return {key: pd.read_csv(base_path / fname) for key, fname in files.items()}

def transform(df, key):
    if "beneficiary" in key:
        df["DOB"] = pd.to_datetime(df["DOB"], errors='coerce')
        df["DOD"] = pd.to_datetime(df["DOD"], errors='coerce')

    # Safe fillna for all column types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(pd.NaT)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")
    return df


def load_to_sqlite(dfs, db_path):
    conn = sqlite3.connect(db_path)
    for name, df in dfs.items():
        # Fix the warning properly
        df = df.copy()
        df.fillna(value=pd.NA, inplace=True)
        
        # Convert datetime columns to string
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"⏳ Converting datetime column '{col}' in '{name}' to string")
                df[col] = df[col].astype(str)
        
        # Save to SQLite
        try:
            df.to_sql(name, conn, if_exists='replace', index=False)
            print(f"✅ Loaded '{name}' into database")
        except Exception as e:
            print(f"❌ Failed to load '{name}': {e}")
    conn.close()


def save_merged(df_merged, output_path):
    df_merged.to_csv(output_path, index=False)

def run_etl():
    base_path = Path("data")
    db_path = Path("db/healthcare_fraud.db")
    output_path = base_path / "processed_train.csv"

    os.makedirs("db", exist_ok=True)

    dfs = extract_csvs(base_path)
    dfs = {k: transform(v, k) for k, v in dfs.items()}

    load_to_sqlite(dfs, db_path)

    # Normalize column names
    for k in dfs:
        dfs[k].columns = dfs[k].columns.str.strip().str.replace('\uFEFF', '')

    out = dfs['train_outpatient'].copy()
    inpt = dfs['train_inpatient'].copy()
    claims_df = pd.concat([out, inpt], axis=0, ignore_index=True)

    claims_df = claims_df.merge(dfs['train_beneficiary'], on='BeneID', how='left')

    if 'Provider' in claims_df.columns and 'Provider' in dfs['train'].columns:
        final_df = claims_df.merge(dfs['train'], on='Provider', how='left')
    else:
        print("❌ 'Provider' column missing — check your data.")
        exit(1)

    save_merged(final_df, output_path)
    print("✅ ETL complete: merged file saved to processed_train.csv")

if __name__ == "__main__":
    run_etl()
