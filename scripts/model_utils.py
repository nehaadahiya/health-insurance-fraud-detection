import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path='data/model_ready.csv', label_col='PotentialFraud'):
    df = pd.read_csv(file_path)

    # Ensure binary target
    if df[label_col].dtype != int:
        df[label_col] = df[label_col].astype('category').cat.codes

    X = df.drop(columns=[label_col])
    y = df[label_col]

    return X, y

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def train_models(X_train, y_train):
    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr

    # ANN
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    ann.fit(X_train, y_train)
    models['ANN'] = ann

    return models

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"🔎 Evaluation for {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "specificity": specificity
    }

def full_pipeline():
    X, y = load_data()
    X_res, y_res = balance_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train)

    scores = {}
    for name, model in models.items():
        print(f"\n{'='*30}\n{name} Results:")
        result = evaluate_model(model, X_test, y_test, model_name=name)
        scores[name] = result

    return scores

if __name__ == "__main__":
    full_pipeline()
