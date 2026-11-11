# Financial Fraud Detection with Explainable AI

This project implements a machine learning system for detecting fraudulent financial transactions using advanced ensemble methods and explainable AI techniques.

## Project Overview

The system processes synthetic financial transaction data to build models that can:
1. Accurately detect fraudulent transactions with high precision and recall
2. Provide explainable results using SHAP and LIME
3. Visualize insights through an interactive dashboard

## Features

- **Data Preprocessing**: Cleaning, normalization, and feature engineering
- **Imbalance Handling**: SMOTE, SMOTE-Tomek, and other resampling techniques
- **Model Development**: Random Forest, XGBoost, LightGBM with hyperparameter tuning
- **Explainable AI**: SHAP and LIME integration for model interpretability
- **Interactive Dashboard**: Streamlit-based visualization for monitoring and explanation

## Project Structure

```
ffd_project/
├── data/                      # Data storage
│   └── processed/             # Processed datasets
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   ├── features/              # Feature engineering
│   ├── models/                # Model training and evaluation
│   └── visualization/         # Visualization and dashboard
├── models/                    # Saved model files
├── dashboard/                 # Dashboard application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-fraud-detection.git
cd financial-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**:
   ```bash
   python src/data/preprocess.py
   ```

2. **Model Training**:
   ```bash
   python src/models/train_model.py
   ```

3. **Run Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## Performance Metrics

The system aims to achieve:
- F1-score: ~0.92
- AUC-ROC: ~0.96
- High recall to catch more fraud cases
- Balanced precision to reduce false alarms

## Dashboard Preview

The dashboard provides:
- Real-time fraud detection monitoring
- Model explanation for individual transactions
- Feature importance visualization
- Performance metrics tracking
