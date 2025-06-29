# Health Insurance Fraud Detection 🚨

## Project Story

Health insurance fraud isn’t just a line item loss — it’s a silent heist that affects millions. This project aims to detect fraudulent claims using advanced machine learning techniques and anomaly detection methods. By analyzing patterns and catching outliers, we help insurers identify suspicious activities before they drain resources.

## Project Overview

This project focuses on identifying fraudulent insurance claims using both supervised and unsupervised models. The final system includes an end-to-end data pipeline, fraud detection model, and an interactive Streamlit app for demo purposes.

## Project Structure

```
📁 Health Insurance Fraud Detection/
│── 📂 data/
│   ├── model_ready.csv                # Preprocessed dataset
│── 📂 scripts/
│   ├── data_preprocessing.py         # Cleaning and feature engineering
│   ├── fraud_detection_model.py      # Training and evaluation
│   ├── streamlit_app.py              # Streamlit dashboard
│── 📂 models/                        # Saved trained models
│── 📂 outputs/                       # Visualizations and evaluation metrics
│── README.md
```

## Dataset Information

* **Source**: Multiple health claim datasets (India-based)
* **Segments**:

  * Generic treatment and costing data
  * Patient-specific treatment and claim history
  * Insurance claim data including fraud labels
* **Goal**: Detect anomalies and classify potential frauds

## Approach & Models

* **Supervised Models**:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * LightGBM
  * Ensemble VotingClassifier

* **Unsupervised / Anomaly Detection**:

  * Isolation Forest
  * One-Class SVM

* **Techniques Used**:

  * SMOTE for class imbalance
  * Feature selection via SelectFromModel
  * Hyperparameter tuning (RandomizedSearchCV)
  * Performance metrics: Accuracy, Precision, Recall, F1, AUC

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repo-url>
cd Health Insurance Fraud Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

```bash
python scripts/data_preprocessing.py
```

### 4. Train the Model

```bash
python scripts/fraud_detection_model.py
```

### 5. Launch Streamlit App

```bash
streamlit run scripts/streamlit_app.py
```

## Streamlit Dashboard

### Screenshots

#### Model Performance Overview

![Performance Screenshot](/outputs/streamlit_perf_overview.png)

#### Feature Importance

![Feature Importance Screenshot](/outputs/streamlit_feature_importance.png)

#### Fraud Probability Distribution

![Probability Screenshot](/outputs/streamlit_probability_dist.png)

### App Features

* Upload new claim datasets and check predictions
* Visualize fraud probability and key metrics
* Explore feature importance and decision thresholds interactively

## Future Improvements

* Integrate real-time claim processing pipeline
* Add Power BI visualization for executive-level summaries
* Incorporate more advanced explainability modules (e.g., SHAP)
* Expand to multi-country datasets for broader fraud scenarios

## Contact

For questions or collaboration, feel free to reach out.

---

🔥 Let’s turn data into defense — one claim at a time.
