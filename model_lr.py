# logistic_regression_pipeline.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.feature_selection import RFECV
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

# Directories
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path="data/model_ready.csv"):
    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)

    # Drop irrelevant identifiers
    drop_cols = ['BeneID', 'ClaimID', 'Provider']
    X = df.drop(columns=['PotentialFraud'] + drop_cols, errors='ignore')

    # Encode 'PotentialFraud' into 0/1
    y = df['PotentialFraud'].astype('category').cat.codes

    return X, y

def preprocess_data(X, y):
    # Drop non-numeric columns (like dates or object types)
    X_numeric = X.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    print("🕵️ Non-numeric columns:", X.select_dtypes(exclude=[np.number]).columns.tolist())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled



def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def select_features(X_train, y_train, X_test):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    selector = RFECV(estimator=lr, step=1, cv=5, scoring='roc_auc', n_jobs=-1)
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test), selector


def tune_logistic_regression(X, y):
    param_dist = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'C': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.0, 0.5, 1.0]  # For elasticnet
    }

    base_model = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_


def calibrate_and_predict(model, X_train, y_train, X_test):
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    y_probs = calibrated_model.predict_proba(X_test)[:, 1]
    return calibrated_model, y_probs


def find_best_threshold(y_test, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], precision, recall


def evaluate_model(y_test, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    return report, conf_mat, auc, y_pred


def plot_results(conf_mat, auc, threshold, y_test, y_probs, precision, recall):
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/LR_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/LR_roc_curve.png", dpi=300)
    plt.close()

    # Precision-recall
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/LR_precision_recall_curve.png", dpi=300)
    plt.close()


def print_final_metrics(conf_mat, auc, threshold):
    tn, fp, fn, tp = conf_mat.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"\n🔍 Final Logistic Regression Metrics:")
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall (Sensitivity): {recall:.2%}")
    print(f"Specificity: {specificity:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"AUC: {auc:.2%}")
    print(f"Optimal Threshold: {threshold:.2f}")


def main():
    X, y = load_data()
    X_resampled, y_resampled = preprocess_data(X, y)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X_resampled, y_resampled)
    X_train_sel, X_test_sel, selector = select_features(X_train, y_train, X_test)
    best_model = tune_logistic_regression(X_train_sel, y_train)
    calibrated_model, y_probs = calibrate_and_predict(best_model, X_train_sel, y_train, X_test_sel)
    threshold, precision, recall = find_best_threshold(y_test, y_probs)
    report, conf_mat, auc, _ = evaluate_model(y_test, y_probs, threshold)

    print("📊 Classification Report:\n", pd.DataFrame(report).transpose())
    plot_results(conf_mat, auc, threshold, y_test, y_probs, precision, recall)
    print_final_metrics(conf_mat, auc, threshold)

    joblib.dump(calibrated_model, f"{MODEL_DIR}/logistic_model.pkl")
    joblib.dump(selector, f"{MODEL_DIR}/feature_selector.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")


if __name__ == "__main__":
    main()
