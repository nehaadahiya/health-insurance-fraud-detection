import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, os, joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTETomek

# === STEP 1: Load & Preprocess Data ===
def preprocess_data():
    df = pd.read_csv("data/model_ready.csv", on_bad_lines='skip', low_memory=False)
    drop_cols = ['BeneID', 'ClaimID', 'Provider']
    X = df.drop(columns=['PotentialFraud'] + drop_cols, errors='ignore')
    y = df['PotentialFraud'].astype('category').cat.codes

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns

# === STEP 2: Resample using SMOTETomek ===
def resample_data(X, y):
    print("🌀 Applying SMOTETomek...")
    smt = SMOTETomek(random_state=42)
    return smt.fit_resample(X, y)

# === STEP 3: Hyperparameter Tuning with StratifiedKFold ===
def tune_random_forest(X_train, y_train):
    print("⚙️ Tuning Random Forest hyperparameters with StratifiedKFold...")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=50,
        scoring='roc_auc',
        cv=skf,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    print(f"Best RF Params: {random_search.best_params_}")
    return random_search.best_estimator_

# === STEP 4: Feature Selection ===
def select_features(X_train, y_train, model, feature_names, top_n=25):
    print("📊 Running Permutation Feature Importance...")
    results = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': results.importances_mean
    }).sort_values(by='importance', ascending=False)
    top_features = importances.head(top_n)['feature'].values
    plot_top_features(importances.head(top_n))
    return top_features

def plot_top_features(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(y='feature', x='importance', data=df, palette='viridis')
    plt.title("Top Permutation Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/top_feature_importance.png")
    plt.close()

# === STEP 5: Train & Evaluate Ensemble with Stacking ===
def train_and_evaluate(X, y, feature_names):
    X_res, y_res = resample_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # Tune Random Forest base model with StratifiedKFold
    rf_best = tune_random_forest(X_train, y_train)

    # Select top features based on permutation importance of tuned RF
    top_features = select_features(X_train, y_train, rf_best, feature_names)
    top_idx = [list(feature_names).index(f) for f in top_features]

    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    # Define models for ensemble
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    lgbm = LGBMClassifier(class_weight='balanced', random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('rf', rf_best), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1
    )

    stack = StackingClassifier(
        estimators=[('voting', voting_clf)],
        final_estimator=LogisticRegression(),
        passthrough=True,
        n_jobs=-1
    )

    print("🚀 Training stacked model...")
    start = time.time()
    stack.fit(X_train_sel, y_train)
    print(f"✅ Training completed in {(time.time() - start) / 60:.2f} mins")

    os.makedirs("models", exist_ok=True)
    joblib.dump(stack, "models/stacked_model.pkl")

    y_probs = stack.predict_proba(X_test_sel)[:, 1]

    # Custom threshold to push recall higher (you can tune this!)
    threshold = 0.40
    y_pred = (y_probs >= threshold).astype(int)

    # === Evaluation ===
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print(f"🔍 ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0, 0] / sum(cm[0])
    print(f"🧠 Specificity: {specificity:.4f}")

    # === Plot Confusion Matrix ===
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/stacked_confusion_matrix.png")
    plt.close()

    # === Plot ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/stacked_roc_curve.png")
    plt.close()

# === MAIN ===
def main():
    print("📥 Loading data...")
    X, y, feature_names = preprocess_data()

    print("⚙️ Training and evaluating model...")
    train_and_evaluate(X, y, feature_names)

if __name__ == "__main__":
    main()
