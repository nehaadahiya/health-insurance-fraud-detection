import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, os, joblib, gc

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

# === Clear joblib cache to avoid worker crashes ===
def clear_joblib_cache():
    try:
        import joblib.externals.loky.backend.context
        joblib.externals.loky.backend.context._force_kill_worker = True
        gc.collect()
        print("🧹 Cleared joblib cache to avoid worker crashes.")
    except Exception as e:
        print(f"⚠️ Could not clear joblib cache: {e}")

# === STEP 3: Use Best RF Directly ===
def tune_random_forest(X_train, y_train):
    model_path = "models/rf_best_model.pkl"
    
    if os.path.exists(model_path):
        print("✅ Loading cached best Random Forest model...")
        return joblib.load(model_path)

    print("⚙️ Instantiating Random Forest with known best parameters...")
    best_rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    best_rf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, model_path)

    clear_joblib_cache()  # Optional: still good to clear memory

    return best_rf


# === STEP 3: Hyperparameter Tuning with StratifiedKFold ===
# def tune_random_forest(X_train, y_train):
#     model_path = "models/rf_best_model.pkl"
    
#     if os.path.exists(model_path):
#         print("✅ Loading cached best Random Forest model...")
#         return joblib.load(model_path)

#     print("⚙️ Tuning Random Forest hyperparameters with StratifiedKFold...")
#     rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

#     param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [20, 25, 30, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['sqrt', 'log2']
#     }

#     skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

#     random_search = RandomizedSearchCV(
#         rf,
#         param_distributions=param_dist,
#         n_iter=30,
#         scoring='roc_auc',
#         cv=skf,
#         verbose=2,
#         n_jobs=-1,
#         random_state=42
#     )

#     random_search.fit(X_train, y_train)
#     best_rf = random_search.best_estimator_

#     print(f"✅ Best RF Params: {random_search.best_params_}")
    
#     os.makedirs("models", exist_ok=True)
#     joblib.dump(best_rf, model_path)

#     clear_joblib_cache()  # Clear cache after tuning

#     return best_rf

# === Plot Feature Importance ===
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("🌟 Feature Importance - Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/feature_importance.png")
    plt.clf()
    print("✅ Feature importance plot saved to outputs/feature_importance.png")

# === STEP 4: Feature Selection using RF Importances (lightweight alternative) ===
def select_features(X_train, y_train, model, feature_names, top_n=25):
    print("📊 Extracting Feature Importances from RF model (Gini-based)...")
    importances = model.feature_importances_
    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    plot_top_features(importances_df.head(top_n))
    top_features = importances_df.head(top_n)['feature'].values
    return top_features

def plot_top_features(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(y='feature', x='importance', data=df, palette='viridis')
    plt.title("Top 25 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/top_feature_importance_rf.png")
    plt.close()

# # === STEP 4: Feature Selection ===
# def select_features(X_train, y_train, model, feature_names, top_n=25):
#     print("📊 Running Permutation Feature Importance...")
#     results = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=42, n_jobs=1)
#     importances = pd.DataFrame({
#         'feature': feature_names,
#         'importance': results.importances_mean
#     }).sort_values(by='importance', ascending=False)
    
#     plot_top_features(importances.head(top_n))
#     top_features = importances.head(top_n)['feature'].values
#     return top_features

# def plot_top_features(df):
#     os.makedirs("outputs", exist_ok=True)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(y='feature', x='importance', data=df, palette='viridis')
#     plt.title("Top 25 Permutation Feature Importances")
#     plt.tight_layout()
#     plt.savefig("outputs/top_feature_importance.png")
#     plt.close()

# === STEP 5: Train & Evaluate Ensemble with Stacking ===
def train_and_evaluate(X, y, feature_names):
    X_res, y_res = resample_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    rf_best = tune_random_forest(X_train, y_train)

    # Plot feature importance from RF
    plot_feature_importance(rf_best, feature_names)

    top_features = select_features(X_train, y_train, rf_best, feature_names)
    top_idx = [list(feature_names).index(f) for f in top_features]

    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    # Ensemble models
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    lgbm = LGBMClassifier(class_weight='balanced', random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('rf', rf_best), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1
    )

    stack = StackingClassifier(
        estimators=[('voting', voting_clf)],
        final_estimator=LogisticRegression(max_iter=1000),
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

    threshold = 0.40
    y_pred = (y_probs >= threshold).astype(int)

    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print(f"🔍 ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0, 0] / sum(cm[0])
    print(f"🧠 Specificity: {specificity:.4f}")

    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_probs)

def plot_confusion_matrix(cm):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/stacked_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_test, y_probs):
    os.makedirs("outputs", exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='darkorange')
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
