import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os 

def main():
    import time

    # Load dataset
    df = pd.read_csv("data/model_ready.csv", on_bad_lines='skip', low_memory=False)
    
    # Drop high-cardinality columns
    drop_cols = ['BeneID', 'ClaimID', 'Provider']
    X = df.drop(columns=['PotentialFraud'] + drop_cols, errors='ignore')
    y = df['PotentialFraud'].astype('category').cat.codes
    
    #label encode all object-type columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }


    # Model and hyperparameter search
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,           # 🚀 Was 30
        scoring='f1',
        n_jobs=-1,
        cv=3,
        verbose=2,           # 📢 Gives you progress updates
        random_state=42
    )

    print("\n🔥 Starting model training...")
    start = time.time()
    search.fit(X_train, y_train)
    end = time.time()
    print(f"⏱️ Training completed in {(end - start)/60:.2f} minutes.\n")

    # Best model
    best_rf = search.best_estimator_

    # Save the model
    joblib.dump(best_rf, "models/random_forest_fraud.pkl")

    # Predict and evaluate
    y_pred = best_rf.predict(X_test)
    y_probs = best_rf.predict_proba(X_test)[:, 1]

    print("Best RF parameters:", search.best_params_)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_probs))

    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("f{output_dir}/rf_confusion_matrix.png")
    plt.show()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Random Forest - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("f{output_dir}/rf_roc_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
