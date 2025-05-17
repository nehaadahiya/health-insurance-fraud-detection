import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Load dataset
    df = pd.read_csv("data/model_ready.csv")
    X = df.drop(columns=['PotentialFraud'])
    y = df['PotentialFraud'].astype('category').cat.codes

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Hyperparameter grid for MLP
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['adaptive'],
        'max_iter': [300, 500],
        'early_stopping': [True]
    }

    mlp = MLPClassifier(random_state=42)
    grid = GridSearchCV(
        mlp,
        param_grid,
        scoring='f1',
        n_jobs=-1,
        cv=3,
        verbose=2
    )
    grid.fit(X_train, y_train)

    # Best model
    best_mlp = grid.best_estimator_

    # Save the model
    joblib.dump(best_mlp, "models/brute_force_mlp.pkl")

    # Predict and evaluate
    y_pred = best_mlp.predict(X_test)
    y_prob = best_mlp.predict_proba(X_test)[:, 1]

    print("Best MLP parameters:", grid.best_params_)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.title("Brute-Force MLP - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/brute_force_mlp_confusion.png")
    plt.show()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Brute-Force MLP - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/brute_force_mlp_roc.png")
    plt.show()

if __name__ == "__main__":
    main()
