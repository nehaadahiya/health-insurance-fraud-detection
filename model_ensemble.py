import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

def main():
    # Load dataset
    df = pd.read_csv("data/model_ready.csv")
    X = df.drop(columns=['PotentialFraud'])
    y = df['PotentialFraud'].astype('category').cat.codes

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define individual models
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('mlp', mlp), ('lr', lr)],
        voting='soft',
        n_jobs=-1
    )

    # Train ensemble
    ensemble.fit(X_train, y_train)

    # Save ensemble
    joblib.dump(ensemble, "models/ensemble_model.pkl")

    # Predict and evaluate
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.title("Voting Classifier - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/ensemble_confusion_matrix.png")
    plt.show()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Voting Classifier - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/ensemble_roc_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier, plot_importance
from skopt import BayesSearchCV

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Step 0: Load dataset
df = pd.read_csv("your_dataset.csv")  # <-- Replace with your dataset path
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Step 1: Subsample for tuning
X_sample, _, y_sample, _ = train_test_split(X, y, stratify=y, train_size=500000, random_state=42)

# Step 2: Validation split
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, stratify=y_sample, test_size=0.2, random_state=42)

# Step 3: SMOTE
smote = SMOTE(random_state=42, n_jobs=-1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 4: Feature selection
lgb_temp = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lgb_temp.fit(X_train_res, y_train_res)
selector = SelectFromModel(lgb_temp, prefit=True, threshold='median')
X_train_res_fs = selector.transform(X_train_res)
X_val_fs = selector.transform(X_val)

# Step 5: Bayesian tuning
param_space = {
    'num_leaves': (20, 150),
    'max_depth': (5, 30),
    'learning_rate': (1e-3, 0.3, 'log-uniform'),
    'min_child_samples': (20, 100),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'reg_alpha': (0, 5),
    'reg_lambda': (0, 5),
}

lgb_est = LGBMClassifier(n_estimators=1000, random_state=42, n_jobs=-1)

opt = BayesSearchCV(
    lgb_est,
    param_space,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=42
)

opt.fit(X_train_res_fs, y_train_res,
        eval_set=[(X_val_fs, y_val)],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=False)

# Step 6: Final training on full data
X_full_fs = selector.transform(X)
final_model = LGBMClassifier(n_estimators=1000, random_state=42, n_jobs=-1, **opt.best_params_)
final_model.fit(X_full_fs, y)

# Save model
joblib.dump(final_model, "lgbm_fraud_model.pkl")

# Predict and report
y_pred = final_model.predict(X_full_fs)
y_proba = final_model.predict_proba(X_full_fs)[:, 1]

report = classification_report(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)
conf_mat = confusion_matrix(y, y_pred)

print("=== Classification Report ===\n", report)
print("ROC AUC Score:", roc_auc)
print("Best Parameters:\n", opt.best_params_)

# Step 7: Plotting

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y, y_proba)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Precision-Recall Curve
PrecisionRecallDisplay.from_predictions(y, y_proba)
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.close()

# Feature Importance
plt.figure(figsize=(12, 8))
plot_importance(final_model, max_num_features=20, importance_type="gain", title="Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("All plots saved: confusion_matrix.png, roc_curve.png, pr_curve.png, feature_importance.png")
