# Financial Fraud Detection Dashboard

This dashboard provides an interactive interface for exploring and explaining financial fraud detection models.

## Features

- **Model Performance Evaluation**: Compare different models using metrics like precision, recall, F1-score, and AUC-ROC
- **Transaction Explanation**: Understand why specific transactions were flagged as fraudulent using SHAP and LIME
- **Live Monitoring**: Simulate a stream of transactions and monitor fraud alerts in real-time
- **Visualizations**: Interactive plots for model performance and explanations

## Running the Dashboard

To run the dashboard, execute the following command from the project root directory:

```bash
streamlit run dashboard/app.py
```

## Dashboard Pages

1. **Home**: Overview of the project and dataset
2. **Model Performance**: Detailed evaluation of model performance with interactive visualizations
3. **Transaction Explanation**: Explainable AI tools to understand individual predictions
4. **Live Monitoring**: Simulated transaction stream with real-time fraud detection

## Usage Instructions

1. First, load the data using the button in the sidebar
2. Load the trained models using the button in the sidebar
3. Navigate between pages using the sidebar menu
4. Explore different models and their explanations

## Requirements

All required packages are listed in the project's `requirements.txt` file.
