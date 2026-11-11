"""
Explainable AI module for financial fraud detection.
This module implements SHAP and LIME for model interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline
import os
import joblib
from IPython.display import display, HTML

def explain_with_shap(model, X, feature_names=None, sample_idx=None, output_dir=None):
    """
    Explain model predictions using SHAP (SHapley Additive exPlanations).
    
    Args:
        model: Trained model
        X: Feature matrix to explain
        feature_names: Names of the features
        sample_idx: Index of a specific sample to explain (if None, global explanation)
        output_dir: Directory to save the visualizations
        
    Returns:
        shap.Explainer: SHAP explainer object
    """
    print("Generating SHAP explanations...")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Create SHAP explainer based on model type
    if hasattr(model, 'predict_proba'):
        # For most scikit-learn models
        explainer = shap.Explainer(model, X[:100])  # Use a subset for background
    elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
        # For XGBoost models
        explainer = shap.TreeExplainer(model)
    elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
        # For LightGBM models
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer
        def predict_fn(x):
            return model.predict_proba(x)[:, 1]
        
        # Use a subset of data as background
        background = shap.sample(X, 100)
        explainer = shap.KernelExplainer(predict_fn, background)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Global feature importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # Bar plot of mean absolute SHAP values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Mean Impact")
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'shap_mean_impact.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # Individual explanation if sample_idx is provided
    if sample_idx is not None:
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        plt.title(f"SHAP Explanation for Sample {sample_idx}")
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'shap_sample_{sample_idx}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    return explainer, shap_values

def explain_with_lime(model, X, feature_names=None, class_names=None, sample_idx=None, output_dir=None):
    """
    Explain model predictions using LIME (Local Interpretable Model-agnostic Explanations).
    
    Args:
        model: Trained model
        X: Feature matrix to explain
        feature_names: Names of the features
        class_names: Names of the classes
        sample_idx: Index of a specific sample to explain (if None, a random sample is chosen)
        output_dir: Directory to save the visualizations
        
    Returns:
        lime.lime_tabular.LimeTabularExplainer: LIME explainer object
    """
    print("Generating LIME explanations...")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Create class names if not provided
    if class_names is None:
        class_names = ['Normal', 'Fraud']
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # Choose a sample to explain
    if sample_idx is None:
        sample_idx = np.random.randint(0, X.shape[0])
    
    # Get the sample to explain
    sample = X[sample_idx]
    
    # Define prediction function
    def predict_fn(x):
        return model.predict_proba(x)
    
    # Generate explanation
    explanation = explainer.explain_instance(
        sample, 
        predict_fn, 
        num_features=10
    )
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save explanation as HTML
        explanation_html = explanation.as_html()
        with open(os.path.join(output_dir, f'lime_sample_{sample_idx}.html'), 'w') as f:
            f.write(explanation_html)
        
        # Save explanation as image
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for Sample {sample_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lime_sample_{sample_idx}.png'), bbox_inches='tight')
        plt.close()
    else:
        # Display explanation
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for Sample {sample_idx}")
        plt.tight_layout()
        plt.show()
    
    return explainer, explanation

def explain_prediction(model, X, sample, feature_names=None, class_names=None, output_dir=None):
    """
    Explain a specific prediction using both SHAP and LIME.
    
    Args:
        model: Trained model
        X: Feature matrix (for background distribution)
        sample: Sample to explain (1D array or DataFrame row)
        feature_names: Names of the features
        class_names: Names of the classes
        output_dir: Directory to save the visualizations
        
    Returns:
        dict: Dictionary containing explanations
    """
    print("Explaining prediction...")
    
    # Convert sample to numpy array if it's a DataFrame
    if hasattr(sample, 'values'):
        sample_values = sample.values
    else:
        sample_values = sample
    
    # Reshape sample if needed
    if sample_values.ndim == 1:
        sample_values = sample_values.reshape(1, -1)
    
    # Get prediction and probability
    prediction = model.predict(sample_values)[0]
    probability = model.predict_proba(sample_values)[0, prediction]
    
    print(f"Prediction: {class_names[prediction] if class_names else prediction}")
    print(f"Probability: {probability:.4f}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # SHAP explanation
    shap_explainer, shap_values = explain_with_shap(
        model, X, feature_names, sample_idx=0, output_dir=None
    )
    
    # Calculate SHAP values for the sample
    sample_shap_values = shap_explainer(sample_values)[0]
    
    # SHAP waterfall plot
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(sample_shap_values, show=False)
    plt.title("SHAP Explanation")
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sample_shap_explanation.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # LIME explanation
    lime_explainer, lime_explanation = explain_with_lime(
        model, X, feature_names, class_names, sample_idx=None, output_dir=None
    )
    
    # Generate LIME explanation for the sample
    def predict_fn(x):
        return model.predict_proba(x)
    
    sample_lime_explanation = lime_explainer.explain_instance(
        sample_values[0], 
        predict_fn, 
        num_features=10
    )
    
    # LIME plot
    fig = sample_lime_explanation.as_pyplot_figure()
    plt.title("LIME Explanation")
    if output_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_lime_explanation.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # Return explanations
    return {
        'prediction': prediction,
        'probability': probability,
        'shap_values': sample_shap_values,
        'lime_explanation': sample_lime_explanation
    }

def create_explanation_report(model, X, y, feature_names=None, class_names=None, 
                             n_samples=5, output_dir=None):
    """
    Create a comprehensive explanation report for a model.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: Names of the features
        class_names: Names of the classes
        n_samples: Number of individual samples to explain
        output_dir: Directory to save the report
        
    Returns:
        dict: Dictionary containing explanation components
    """
    print("Creating explanation report...")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Create class names if not provided
    if class_names is None:
        class_names = ['Normal', 'Fraud']
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Global SHAP explanation
    shap_explainer, shap_values = explain_with_shap(
        model, X, feature_names, output_dir=output_dir
    )
    
    # Select samples for individual explanations
    # Include both classes and focus on interesting cases
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Find correct and incorrect predictions for each class
    correct_normal = np.where((y == 0) & (y_pred == 0))[0]
    incorrect_normal = np.where((y == 0) & (y_pred == 1))[0]  # False positives
    correct_fraud = np.where((y == 1) & (y_pred == 1))[0]
    incorrect_fraud = np.where((y == 1) & (y_pred == 0))[0]  # False negatives
    
    # Select samples from each category
    samples_to_explain = []
    
    # Add false positives (normal transactions predicted as fraud)
    if len(incorrect_normal) > 0:
        # Sort by prediction probability (most confident mistakes first)
        sorted_idx = incorrect_normal[np.argsort(-y_pred_proba[incorrect_normal])]
        samples_to_explain.extend(sorted_idx[:min(2, len(sorted_idx))])
    
    # Add false negatives (fraud transactions predicted as normal)
    if len(incorrect_fraud) > 0:
        # Sort by prediction probability (least confident mistakes first)
        sorted_idx = incorrect_fraud[np.argsort(y_pred_proba[incorrect_fraud])]
        samples_to_explain.extend(sorted_idx[:min(2, len(sorted_idx))])
    
    # Add correct predictions
    if len(correct_normal) > 0:
        samples_to_explain.append(np.random.choice(correct_normal))
    
    if len(correct_fraud) > 0:
        samples_to_explain.append(np.random.choice(correct_fraud))
    
    # Limit to n_samples
    samples_to_explain = samples_to_explain[:n_samples]
    
    # Individual explanations
    individual_explanations = []
    
    for i, sample_idx in enumerate(samples_to_explain):
        print(f"Explaining sample {i+1}/{len(samples_to_explain)} (index {sample_idx})...")
        
        # Get sample
        sample = X[sample_idx]
        true_label = y[sample_idx]
        pred_label = y_pred[sample_idx]
        pred_proba = y_pred_proba[sample_idx]
        
        # Sample-specific output directory
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}') if output_dir else None
        if sample_output_dir:
            os.makedirs(sample_output_dir, exist_ok=True)
        
        # SHAP explanation
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        plt.title(f"SHAP Explanation for Sample {sample_idx}")
        if sample_output_dir:
            plt.savefig(os.path.join(sample_output_dir, 'shap_explanation.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        # LIME explanation
        lime_explainer, _ = explain_with_lime(
            model, X, feature_names, class_names, sample_idx=sample_idx, 
            output_dir=sample_output_dir
        )
        
        # Store explanation metadata
        explanation_meta = {
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': pred_label,
            'prediction_probability': pred_proba,
            'correct_prediction': true_label == pred_label,
            'explanation_type': 'False Positive' if true_label == 0 and pred_label == 1 else
                               'False Negative' if true_label == 1 and pred_label == 0 else
                               'True Positive' if true_label == 1 and pred_label == 1 else
                               'True Negative'
        }
        
        individual_explanations.append(explanation_meta)
    
    # Create summary table
    if output_dir:
        summary_df = pd.DataFrame(individual_explanations)
        summary_df.to_csv(os.path.join(output_dir, 'explanation_summary.csv'), index=False)
        
        # Create HTML report
        html_report = create_html_report(model, X, y, feature_names, class_names, 
                                        individual_explanations, output_dir)
        
        with open(os.path.join(output_dir, 'explanation_report.html'), 'w') as f:
            f.write(html_report)
    
    return {
        'shap_explainer': shap_explainer,
        'shap_values': shap_values,
        'individual_explanations': individual_explanations
    }

def create_html_report(model, X, y, feature_names, class_names, individual_explanations, output_dir):
    """
    Create an HTML report for model explanations.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: Names of the features
        class_names: Names of the classes
        individual_explanations: List of individual explanation metadata
        output_dir: Directory where explanation files are saved
        
    Returns:
        str: HTML report content
    """
    # Create HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Explanation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            .summary { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .explanation { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            .explanation-header { display: flex; justify-content: space-between; margin-bottom: 20px; }
            .explanation-images { display: flex; flex-wrap: wrap; gap: 20px; }
            .explanation-image { max-width: 100%; }
            .correct { color: green; }
            .incorrect { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Explanation Report</h1>
            
            <div class="summary">
                <h2>Global Feature Importance</h2>
                <img src="shap_feature_importance.png" alt="SHAP Feature Importance" style="max-width: 100%;">
                <img src="shap_mean_impact.png" alt="SHAP Mean Impact" style="max-width: 100%;">
            </div>
            
            <h2>Individual Explanations</h2>
            <table>
                <tr>
                    <th>Sample ID</th>
                    <th>True Label</th>
                    <th>Predicted Label</th>
                    <th>Probability</th>
                    <th>Correct</th>
                    <th>Type</th>
                </tr>
    """
    
    # Add rows for each individual explanation
    for exp in individual_explanations:
        correct_class = "correct" if exp['correct_prediction'] else "incorrect"
        html += f"""
                <tr>
                    <td>{exp['sample_idx']}</td>
                    <td>{class_names[exp['true_label']]}</td>
                    <td>{class_names[exp['predicted_label']]}</td>
                    <td>{exp['prediction_probability']:.4f}</td>
                    <td class="{correct_class}">{exp['correct_prediction']}</td>
                    <td>{exp['explanation_type']}</td>
                </tr>
        """
    
    html += """
            </table>
    """
    
    # Add individual explanation sections
    for exp in individual_explanations:
        sample_idx = exp['sample_idx']
        html += f"""
            <div class="explanation">
                <div class="explanation-header">
                    <h3>Sample {sample_idx} ({exp['explanation_type']})</h3>
                    <div>
                        <p>True Label: {class_names[exp['true_label']]}</p>
                        <p>Predicted Label: {class_names[exp['predicted_label']]} ({exp['prediction_probability']:.4f})</p>
                    </div>
                </div>
                
                <div class="explanation-images">
                    <div>
                        <h4>SHAP Explanation</h4>
                        <img src="sample_{sample_idx}/shap_explanation.png" alt="SHAP Explanation" class="explanation-image">
                    </div>
                    <div>
                        <h4>LIME Explanation</h4>
                        <img src="sample_{sample_idx}/lime_sample_{sample_idx}.png" alt="LIME Explanation" class="explanation-image">
                    </div>
                </div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def compare_explanations(models, model_names, X, feature_names=None, sample_idx=None, output_dir=None):
    """
    Compare explanations across different models.
    
    Args:
        models: List of trained models
        model_names: List of model names
        X: Feature matrix
        feature_names: Names of the features
        sample_idx: Index of a specific sample to explain (if None, a random sample is chosen)
        output_dir: Directory to save the visualizations
        
    Returns:
        dict: Dictionary containing explanations for each model
    """
    print("Comparing explanations across models...")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Choose a sample to explain
    if sample_idx is None:
        sample_idx = np.random.randint(0, X.shape[0])
    
    # Get the sample to explain
    sample = X[sample_idx]
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate explanations for each model
    explanations = {}
    
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        print(f"Generating explanations for {model_name}...")
        
        # Model-specific output directory
        model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
        if model_output_dir:
            os.makedirs(model_output_dir, exist_ok=True)
        
        # SHAP explanation
        shap_explainer, shap_values = explain_with_shap(
            model, X, feature_names, sample_idx=sample_idx, output_dir=model_output_dir
        )
        
        # LIME explanation
        lime_explainer, lime_explanation = explain_with_lime(
            model, X, feature_names, sample_idx=sample_idx, output_dir=model_output_dir
        )
        
        # Store explanations
        explanations[model_name] = {
            'shap_explainer': shap_explainer,
            'shap_values': shap_values,
            'lime_explainer': lime_explainer,
            'lime_explanation': lime_explanation
        }
    
    # Create comparison visualizations
    if len(models) > 1 and output_dir:
        # Compare SHAP values across models
        plt.figure(figsize=(12, 8))
        
        for i, model_name in enumerate(model_names):
            shap_values = explanations[model_name]['shap_values']
            
            plt.subplot(len(models), 1, i+1)
            shap.plots.waterfall(shap_values[sample_idx], show=False)
            plt.title(f"SHAP Explanation - {model_name}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison_shap.png'), bbox_inches='tight')
        plt.close()
    
    return explanations

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.preprocess import preprocess_data
    from models.train_model import train_and_evaluate_all_models, load_model
    
    # Load preprocessed data
    processed_dir = "../../data/processed"
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # Load feature names
    with open(os.path.join(processed_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().splitlines()
    
    # Load trained model
    model_dir = "../../models"
    model_dict = load_model(os.path.join(model_dir, 'models'), 'xgboost')
    model = model_dict['model']
    
    # Create explanation report
    explanation_dir = os.path.join(model_dir, 'explanations')
    explanations = create_explanation_report(
        model, X_test, y_test, feature_names, 
        class_names=['Normal', 'Fraud'], 
        output_dir=explanation_dir
    )
    
    print("Explanation report created successfully.")
