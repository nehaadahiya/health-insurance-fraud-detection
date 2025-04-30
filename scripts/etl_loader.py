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
    df.fillna(0, inplace=True)
    return df

def load_to_sqlite(dfs, db_path):
    conn = sqlite3.connect(db_path)
    for name, df in dfs.items():
        df.to_sql(name, conn, if_exists='replace', index=False)
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

    df = dfs['train']
    df = df.merge(dfs['train_beneficiary'], on='BeneID', how='left')
    df = df.merge(dfs['train_outpatient'], on='ClaimID', how='left')
    df = df.merge(dfs['train_inpatient'], on='ClaimID', how='left')

    save_merged(df, output_path)
    print("ETL complete: merged file saved to processed_train.csv")

if __name__ == "__main__":
    run_etl()
