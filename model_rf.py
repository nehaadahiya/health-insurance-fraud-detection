import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import joblib
import optuna
import os
import time
import numpy as np

def preprocess_data():
    df = pd.read_csv("data/model_ready.csv", on_bad_lines='skip', low_memory=False)
    drop_cols = ['BeneID', 'ClaimID', 'Provider']
    X = df.drop(columns=['PotentialFraud'] + drop_cols, errors='ignore')
    y = df['PotentialFraud'].astype('category').cat.codes

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def optimize_model(X, y):
    def objective(trial):
        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 400),
            max_depth=trial.suggest_int("max_depth", 10, 30),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        smt = SMOTETomek(random_state=42)
        X_res, y_res = smt.fit_resample(X, y)
        scores = cross_val_score(rf, X_res, y_res, cv=skf, scoring='f1_weighted')
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    return study.best_params

def train_final_model(X, y, best_rf_params):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    rf = RandomForestClassifier(**best_rf_params, class_weight='balanced', random_state=42, n_jobs=-1)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        learning_rate=0.05, max_depth=8, n_estimators=300, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=1, random_state=42)
    lgbm = LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                          subsample=0.8, class_weight='balanced', random_state=42)

    ensemble = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )

    print("🚀 Training ensemble model...")
    start = time.time()
    ensemble.fit(X_train, y_train)
    print(f"✅ Training completed in {(time.time() - start) / 60:.2f} minutes")

    joblib.dump(ensemble, "models/ensemble_model.pkl")

    y_probs = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.45).astype(int)

    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print(f"🔍 ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    specificity = confusion_matrix(y_test, y_pred)[0, 0] / sum(confusion_matrix(y_test, y_pred)[0])
    print(f"🧠 Specificity: {specificity:.4f}")

    # Save confusion matrix and ROC plot
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Optimized Ensemble - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/optimized_confusion_matrix.png")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Optimized Ensemble - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/optimized_roc_curve.png")
    plt.show()

def main():
    print("📥 Loading and preprocessing data...")
    X, y = preprocess_data()

    print("\n🔍 Hyperparameter tuning using Optuna...")
    best_rf_params = optimize_model(X, y)
    print(f"✅ Best Random Forest params: {best_rf_params}")

    print("\n🎯 Training final model with stacking and SMOTETomek...")
    train_final_model(X, y, best_rf_params)

if __name__ == "__main__":
    main()
