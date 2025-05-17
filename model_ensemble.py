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
