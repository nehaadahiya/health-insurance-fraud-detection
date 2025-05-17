import pandas as pd
import sqlite3
import os

# File paths for the uploaded files
file_paths = {
    "train_outpatient": "/data/Train_Outpatientdata-1542865627584.csv",
    "test_beneficiary": "/data/Test_Beneficiarydata-1542969243754.csv",
    "test_inpatient": "/data/Test_Inpatientdata-1542969243754.csv",
    "test_outpatient": "data/Test_Outpatientdata-1542969243754.csv",
    "test": "data/Test-1542969243754.csv",
    "train_beneficiary": "data/Train_Beneficiarydata-1542865627584.csv",
    "train_inpatient": "data/Train_Inpatientdata-1542865627584.csv",
    "train": "data/Train-1542865627584.csv"
}

# SQLite database name
db_name = "healthcare_fraud_multiple_files.db"

# Step 1: Extract
def extract_data(file_path):
    """
    Extract data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Extracted data from {file_path}. Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        return data
    except Exception as e:
        print(f"Error extracting data from {file_path}: {e}")
        return None

# Step 2: Transform
def transform_data(data, file_key):
    """
    Apply specific transformations based on the file key.
    """
    try:
        # Handle missing values
        data.fillna(method='ffill', inplace=True)

        # Transformations based on file type (example)
        if "beneficiary" in file_key:
            # Example transformation: Ensure date columns are datetime
            date_cols = [col for col in data.columns if "Date" in col]
            for col in date_cols:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        elif "outpatient" in file_key or "inpatient" in file_key:
            # Example transformation: Convert numeric columns
            numeric_cols = data.select_dtypes(include=['object']).columns
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Transformed data for {file_key}.")
        return data
    except Exception as e:
        print(f"Error transforming data for {file_key}: {e}")
        return None

# Step 3: Load
def load_data(data, db_name, table_name):
    """
    Load transformed data into a SQLite database, converting datetime columns to strings.
    """
    try:
        # DEBUG: Print column types before conversion
        print(f"Converting datetime columns in {table_name} before loading...")
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                print(f"Converting column: {col} from datetime to string")
                data[col] = data[col].astype(str)

        conn = sqlite3.connect(db_name)
        data.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"✅ Loaded data into table '{table_name}' in database '{db_name}'.")
    except Exception as e:
        print(f"❌ Error loading data into table '{table_name}': {e}")


# Main ETL Pipeline
def etl_pipeline(file_paths, db_name):
    """
    Orchestrate the ETL process for multiple files.
    """
    print("Starting ETL pipeline for multiple files...")
    for file_key, file_path in file_paths.items():
        print(f"Processing {file_key}...")
        # Extract
        raw_data = extract_data(file_path)
        if raw_data is None:
            continue
        
        # Transform
        transformed_data = transform_data(raw_data, file_key)
        if transformed_data is None:
            continue
        
        # Load
        load_data(transformed_data, db_name, file_key)
    
    print("ETL pipeline completed for all files.")

# Run the ETL pipeline
etl_pipeline(file_paths, db_name)
