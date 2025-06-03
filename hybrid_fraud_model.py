import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === Load Data with Feature Engineering ===
def load_processed_data(path="data/model_ready.csv", top_feature_path="models/top_25_features.pkl"):
    df = pd.read_csv(path, low_memory=False)

    # Drop ID-like columns
    drop_cols = ['BeneID', 'ClaimID', 'Provider']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Extract and encode target
    y = df.pop('PotentialFraud').astype('category').cat.codes

    # === Date Parsing ===
    date_cols = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # === Derived Numeric Features ===
    if 'ClaimStartDt' in df.columns and 'ClaimEndDt' in df.columns:
        df['ClaimDuration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days

    if 'AdmissionDt' in df.columns and 'DischargeDt' in df.columns:
        df['HospitalStayLength'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days

    if 'ClaimStartDt' in df.columns and 'DOB' in df.columns:
        df['PatientAge'] = (df['ClaimStartDt'] - df['DOB']).dt.days // 365

    if 'DOD' in df.columns:
        df['IsDeceased'] = df['DOD'].notnull().astype(int)

    df.drop(columns=date_cols, inplace=True, errors='ignore')  # Drop raw date cols

    # === Frequency Encoding of Physicians ===
    for col in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df[col + 'Freq'] = df[col].map(freq_map)
            df.drop(columns=col, inplace=True)

    # === Encode Categorical ===
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # === Fill Missing ===
    df.fillna(0, inplace=True)

    # ✅ Load top 25 features from .pkl
    top_features = joblib.load(top_feature_path)
    for col in top_features:
        if col not in df.columns:
            print(f"🧩 Adding missing feature: {col}")
            df[col] = 0
    df = df[top_features]

    return df, y

# === Load Trained Supervised Model ===
def load_supervised_model(path="models/stacked_model.pkl"):
    return joblib.load(path)

# === Generate Isolation Forest Scores ===
def compute_anomaly_scores(X, contamination=0.05):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    scores = iso.decision_function(X_scaled)  # Higher = more normal
    return 1 - (scores - scores.min()) / (scores.max() - scores.min())  # Normalize and invert

# === Compute Hybrid Score ===
def compute_hybrid_scores(classifier_probs, anomaly_scores, alpha=0.7):
    return alpha * classifier_probs + (1 - alpha) * anomaly_scores

# === Evaluate Hybrid Model ===
def evaluate_hybrid_model(y_true, hybrid_scores, threshold=0.5):
    y_pred = (hybrid_scores >= threshold).astype(int)
    print("\n📊 Classification Report:\n", classification_report(y_true, y_pred))
    print(f"🔍 ROC AUC Score: {roc_auc_score(y_true, hybrid_scores):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)
    plot_roc_curve(y_true, hybrid_scores)

# === Plot Functions ===
def plot_confusion_matrix(cm):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Hybrid Model - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/hybrid_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Hybrid Model - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/hybrid_roc_curve.png")
    plt.close()

# === Alpha Optimization ===
def optimize_alpha(y_true, classifier_probs, anomaly_scores):
    print("\n🔧 Optimizing alpha for best ROC AUC...")
    alphas = np.linspace(0, 1, 21)
    scores = []

    for alpha in alphas:
        hybrid_scores = compute_hybrid_scores(classifier_probs, anomaly_scores, alpha)
        auc = roc_auc_score(y_true, hybrid_scores)
        scores.append(auc)

    best_alpha = alphas[np.argmax(scores)]
    best_auc = max(scores)

    print(f"✅ Best alpha: {best_alpha:.2f} with ROC AUC: {best_auc:.4f}")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(alphas, scores, marker='o', color='purple')
    plt.xlabel('Alpha (Weight for Classifier)')
    plt.ylabel('ROC AUC')
    plt.title('Hybrid Model - Alpha Tuning')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/hybrid_alpha_tuning.png")
    plt.close()

    return best_alpha

# === Main Function ===
def main():
    print("📥 Loading data and model...")
    X, y = load_processed_data()
    model = load_supervised_model()

    print("🔮 Getting classifier probabilities...")
    classifier_probs = model.predict_proba(X)[:, 1]

    print("🧪 Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(X)

    best_alpha = optimize_alpha(y, classifier_probs, anomaly_scores)

    print("⚖️  Calculating hybrid scores...")
    hybrid_scores = compute_hybrid_scores(classifier_probs, anomaly_scores, alpha=best_alpha)

    print("📊 Evaluating hybrid model...")
    evaluate_hybrid_model(y, hybrid_scores, threshold=0.5)

if __name__ == "__main__":
    main()
