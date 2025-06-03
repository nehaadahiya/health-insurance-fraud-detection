import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils.multiclass import unique_labels

def load_data(path="data/model_ready.csv"):
    print("Loading data...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # Drop ID columns (adjust if needed)
    df = df.drop(columns=['BeneID', 'ClaimID'], errors='ignore')

    # Separate target
    y = df.pop('PotentialFraud')

    # Ordinal encode diagnosis codes
    diag_cols = [col for col in df.columns if 'ClmDiagnosisCode_' in col] + ['ClmAdmitDiagnosisCode']
    df[diag_cols] = df[diag_cols].astype(str)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[diag_cols] = encoder.fit_transform(df[diag_cols])

    # Process date columns to derive features
    date_cols = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['ClaimDuration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days
    df['HospitalStayLength'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days
    df['PatientAge'] = (df['ClaimStartDt'] - df['DOB']).dt.days // 365
    df['IsDeceased'] = df['DOD'].notnull().astype(int)
    df = df.drop(columns=date_cols)

    # Map RenalDiseaseIndicator
    df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].map({'Y': 1, 'N': 0}).fillna(0)

    # Frequency encode physician-related columns
    phys_cols = ['Provider', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
    for col in phys_cols:
        freq_map = df[col].value_counts().to_dict()
        df[col + '_Freq'] = df[col].map(freq_map)
    df = df.drop(columns=phys_cols)

    return df, y


def add_iso_score_feature(X_train_scaled, X_test_scaled, contamination=0.05, random_state=42):
    print("Training Isolation Forest for anomaly scores...")
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    iso.fit(X_train_scaled)
    print(" Isolation Forest training complete.")
    train_scores = iso.decision_function(X_train_scaled)
    test_scores = iso.decision_function(X_test_scaled)
    print("Anomaly scores computed for train and test.")
    return train_scores, test_scores


import pandas as pd

def plot_anomaly_score_distribution(train_scores, y_train, test_scores, y_test):
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    y_train_numeric = y_train.map({'No': 0, 'Yes': 1})
    y_test_numeric = y_test.map({'No': 0, 'Yes': 1})

    print("Train class distribution:", y_train_numeric.value_counts())
    print("Test class distribution:", y_test_numeric.value_counts())

    plt.figure(figsize=(12, 6))
    sns.kdeplot(train_scores[y_train_numeric == 0], label='Train Normal', fill=True)
    sns.kdeplot(train_scores[y_train_numeric == 1], label='Train Fraud', fill=True)
    sns.kdeplot(test_scores[y_test_numeric == 0], label='Test Normal', linestyle='--')
    sns.kdeplot(test_scores[y_test_numeric == 1], label='Test Fraud', linestyle='--')
    plt.title("Anomaly Score Distribution (Isolation Forest)")
    plt.xlabel("Anomaly Score (Higher = More Normal)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/anomaly_score_distribution.png")
    plt.close()
    print("Saved anomaly score distribution plot.")


def train_ensemble(X_train, y_train):
    print("Training ensemble models...")

    rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    lgbm = LGBMClassifier(random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)
    print("Random Forest trained.")
    xgb.fit(X_train, y_train)
    print("XGBoost trained.")
    lgbm.fit(X_train, y_train)
    print("LightGBM trained.")

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    print("Ensemble trained with soft voting.")

    return ensemble


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    y_test_numeric = y_test.map({'No': 0, 'Yes': 1})
    print("\n[RESULT] Classification Report:\n", classification_report(y_test_numeric, y_pred))
    auc = roc_auc_score(y_test_numeric, y_prob)
    print(f"[RESULT] ROC AUC Score: {auc:.4f}")

    # Ensure all 4 matrix values show
    cm = confusion_matrix(y_test_numeric, y_pred, labels=[0, 1])
    cm_labels = ["TN", "FP", "FN", "TP"]
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap='coolwarm', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
    plt.title("Confusion Matrix - Isolation Forest")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Custom annotation with labels (TN, FP, etc.)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            label = cm_labels[i][j]
            ax.text(j + 0.5, i + 0.5, f"{count}\n({label})", ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("Saved cleaned confusion matrix.")


    fpr, tpr, _ = roc_curve(y_test_numeric, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve - Isolation Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png")
    plt.close()
    print("Saved ROC curve.")


def plot_feature_importance(model, feature_names):
    print("Plotting feature importances...")
    if hasattr(model, 'estimators_'):
        rf = model.estimators_[0]  # Random Forest is the first in ensemble
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Random Forest in Ensemble)")
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.tight_layout()
        plt.savefig("outputs/feature_importance2.png")
        plt.close()
        print(" Saved feature importance plot.")
    else:
        print("[WARN] Model missing 'estimators_' attribute; skipping feature importance.")


def main():
    print("🚀 Starting fraud detection pipeline...")

    os.makedirs("outputs", exist_ok=True)

    X, y = load_data()

    # Split data with stratification on target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Drop any ID columns if still present (adjust names accordingly)
    id_cols = ['BENE_ID', 'CLAIM_ID', 'Provider_ID']
    X_train = X_train.drop(columns=id_cols, errors='ignore')
    X_test = X_test.drop(columns=id_cols, errors='ignore')

    # Drop non-numeric columns (if any remain)
    non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"[WARN] Dropping non-numeric columns: {list(non_numeric_cols)}")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add Isolation Forest anomaly score feature
    train_scores, test_scores = add_iso_score_feature(X_train_scaled, X_test_scaled)

    # Append anomaly score to scaled dataframes
    X_train_enhanced = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_enhanced = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_train_enhanced['AnomalyScore'] = train_scores
    X_test_enhanced['AnomalyScore'] = test_scores

    # Plot anomaly score distribution
    plot_anomaly_score_distribution(train_scores, y_train.values, test_scores, y_test)

    # Impute missing values (mean strategy)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_enhanced), columns=X_train_enhanced.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_enhanced), columns=X_test_enhanced.columns)

    # Balance the data using SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    y_train_numeric = y_train.map({'No': 0, 'Yes': 1})
    X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train_numeric)
    print(f"After SMOTE, X shape: {X_resampled.shape}, y distribution: {np.bincount(y_resampled)}")

    # Train ensemble model
    ensemble = train_ensemble(X_resampled, y_resampled)

    # Evaluate on test set
    evaluate_model(ensemble, X_test_imputed, y_test)

    # Feature importance plot from Random Forest inside ensemble
    plot_feature_importance(ensemble, X_train_enhanced.columns)

    print("🔥 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
