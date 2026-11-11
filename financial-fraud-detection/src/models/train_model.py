"""
Model training module for financial fraud detection.
This module implements various machine learning models and ensemble techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import time
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

def train_random_forest(X_train, y_train, params=None, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Dictionary of model parameters
        random_state: Random seed for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("Training Random Forest classifier...")
    
    # Calculate class weights based on class distribution
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    weight_for_0 = total_samples / (2 * class_counts[0])
    weight_for_1 = total_samples / (2 * class_counts[1])
    class_weight_dict = {0: weight_for_0, 1: weight_for_1}
    print(f"Using class weights: {class_weight_dict}")
    
    # Default parameters optimized for imbalanced data
    if params is None:
        params = {
            'n_estimators': 200,  # More trees for better performance
            'max_depth': 15,      # Allow deeper trees to capture rare patterns
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': class_weight_dict,  # Custom class weights
            'random_state': random_state,
            'n_jobs': -1,         # Use all available cores
            'bootstrap': True,
            'criterion': 'gini'    # Gini often works better for imbalanced data
        }
    
    # Create and train the model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(X_train, y_train, params=None, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Dictionary of model parameters
        random_state: Random seed for reproducibility
        
    Returns:
        xgb.XGBClassifier: Trained model
    """
    print("Training XGBoost classifier...")
    
    # Calculate positive class weight for imbalanced data
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    print(f"Using scale_pos_weight: {scale_pos_weight}")
    
    # Default parameters optimized for imbalanced data
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,  # Lower learning rate for better convergence
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            'random_state': random_state,
            'n_jobs': -1,          # Use all available cores
            'min_child_weight': 1,
            'gamma': 0.1,          # Regularization parameter
            'reg_alpha': 0.1,      # L1 regularization
            'reg_lambda': 1.0,     # L2 regularization
            'eval_metric': 'auc'   # Use AUC as evaluation metric
        }
    
    # Create and train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def train_lightgbm(X_train, y_train, params=None, random_state=42):
    """
    Train a LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Dictionary of model parameters
        random_state: Random seed for reproducibility
        
    Returns:
        lgb.LGBMClassifier: Trained model
    """
    print("Training LightGBM classifier...")
    
    # Calculate class weights based on class distribution
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    weight_for_0 = total_samples / (2 * class_counts[0])
    weight_for_1 = total_samples / (2 * class_counts[1])
    class_weight_dict = {0: weight_for_0, 1: weight_for_1}
    print(f"Using class weights: {class_weight_dict}")
    
    # Default parameters optimized for imbalanced data
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,  # Lower learning rate for better convergence
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'class_weight': class_weight_dict,  # Custom class weights
            'random_state': random_state,
            'n_jobs': -1,          # Use all available cores
            'num_leaves': 31,      # Number of leaves in each tree
            'min_child_samples': 20,  # Minimum number of samples in a leaf
            'reg_alpha': 0.1,      # L1 regularization
            'reg_lambda': 1.0,     # L2 regularization
            'metric': 'auc',       # Use AUC as evaluation metric
            'boost_from_average': True,  # Helps with imbalanced data
            'is_unbalance': True   # Specifically for imbalanced datasets
        }
    
    # Create and train the model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def train_logistic_regression(X_train, y_train, params=None, random_state=42):
    """
    Train a Logistic Regression classifier as a baseline.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Dictionary of model parameters
        random_state: Random seed for reproducibility
        
    Returns:
        LogisticRegression: Trained model
    """
    print("Training Logistic Regression classifier...")
    
    # Default parameters
    if params is None:
        params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'class_weight': 'balanced',
            'random_state': random_state
        }
    
    # Create and train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    return model

def find_optimal_threshold(y_test, y_pred_proba, metric='f1'):
    """
    Find the optimal threshold for classification based on a specific metric.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', or 'balanced')
        
    Returns:
        float: Optimal threshold
    """
    thresholds = np.linspace(0.001, 0.2, 100)  # Focus on lower thresholds for imbalanced data
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_test, y_pred)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred)
        elif metric == 'balanced':
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        scores.append(score)
    
    # Find threshold with highest score
    best_score_idx = np.argmax(scores)
    best_threshold = thresholds[best_score_idx]
    best_score = scores[best_score_idx]
    
    print(f"Optimal threshold for {metric}: {best_threshold:.4f} (score: {best_score:.4f})")
    return best_threshold

def evaluate_model(model, X_test, y_test, model_name="Model", threshold=None, optimize_threshold=True):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for reporting
        threshold: Decision threshold for classification (if None, use 0.01 for imbalanced data)
        optimize_threshold: Whether to find the optimal threshold
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"Evaluating {model_name}...")
    
    # Get predictions and probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Set default threshold for imbalanced data if not provided
    if threshold is None:
        threshold = 0.01  # Much lower than the default 0.5 for imbalanced data
    
    # Find optimal threshold if requested
    if optimize_threshold:
        threshold = find_optimal_threshold(y_test, y_pred_proba, metric='f1')
    
    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Print results
    print(f"{model_name} Performance (threshold={threshold:.4f}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Return metrics
    return {
        'model_name': model_name,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

def visualize_model_performance(evaluation_results, output_dir=None):
    """
    Visualize model performance metrics.
    
    Args:
        evaluation_results: Dictionary of evaluation metrics
        output_dir: Directory to save the visualizations
    """
    print("Visualizing model performance...")
    
    # Extract data
    model_name = evaluation_results['model_name']
    y_test = evaluation_results.get('y_test', None)
    y_pred = evaluation_results['y_pred']
    y_pred_proba = evaluation_results['y_pred_proba']
    cm = evaluation_results['confusion_matrix']
    
    # Create directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    if y_test is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {evaluation_results["auc_roc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
        plt.close()
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'AP = {evaluation_results["avg_precision"]:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc='lower left')
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'))
        plt.close()

def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type='xgboost', n_trials=100, random_state=42):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_type: Type of model to optimize ('random_forest', 'xgboost', 'lightgbm')
        n_trials: Number of optimization trials
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Best hyperparameters
    """
    print(f"Optimizing hyperparameters for {model_type}...")
    
    def objective(trial):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': 'balanced',
                'random_state': random_state
            }
            model = RandomForestClassifier(**params)
            
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
                'objective': 'binary:logistic',
                'random_state': random_state
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'class_weight': 'balanced',
                'objective': 'binary',
                'random_state': random_state
            }
            model = lgb.LGBMClassifier(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate F1 score (primary metric)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f1 = f1_score(y_val, y_pred)
        
        # Also calculate AUC-ROC
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        # Store additional metrics for pruning decisions
        trial.set_user_attr('precision', precision_score(y_val, y_pred))
        trial.set_user_attr('recall', recall_score(y_val, y_pred))
        trial.set_user_attr('auc_roc', auc_roc)
        
        return f1
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name=f"{model_type}_optimization")
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"Best trial:")
    trial = study.best_trial
    print(f"  Value (F1 Score): {trial.value:.4f}")
    print(f"  AUC-ROC: {trial.user_attrs['auc_roc']:.4f}")
    print(f"  Precision: {trial.user_attrs['precision']:.4f}")
    print(f"  Recall: {trial.user_attrs['recall']:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params

def create_ensemble(models, X_train, y_train, weights=None):
    """
    Create a weighted ensemble of models.
    
    Args:
        models: List of trained models
        X_train: Training features
        y_train: Training target
        weights: List of weights for each model (if None, equal weights are used)
        
    Returns:
        dict: Ensemble model dictionary
    """
    print("Creating ensemble model...")
    
    if weights is None:
        # Equal weights
        weights = [1.0 / len(models)] * len(models)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Create ensemble dictionary
    ensemble = {
        'models': models,
        'weights': weights
    }
    
    return ensemble

def predict_ensemble(ensemble, X):
    """
    Make predictions using the ensemble model.
    
    Args:
        ensemble: Ensemble model dictionary
        X: Features
        
    Returns:
        tuple: (y_pred, y_pred_proba) - Predictions and probabilities
    """
    # Get models and weights
    models = ensemble['models']
    weights = ensemble['weights']
    
    # Get predictions from each model
    y_pred_probas = []
    for model in models:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred_probas.append(y_pred_proba)
    
    # Weighted average of probabilities
    y_pred_proba = np.zeros_like(y_pred_probas[0])
    for i, weight in enumerate(weights):
        y_pred_proba += weight * y_pred_probas[i]
    
    # Convert to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return y_pred, y_pred_proba

def save_model(model, model_path, model_name, metadata=None):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_path: Directory to save the model
        model_name: Name of the model
        metadata: Additional metadata to save with the model
    """
    print(f"Saving model {model_name}...")
    
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Create model dictionary with metadata
    model_dict = {
        'model': model,
        'metadata': metadata or {},
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save the model
    joblib.dump(model_dict, os.path.join(model_path, f"{model_name}.pkl"))
    print(f"Model saved to {os.path.join(model_path, model_name)}.pkl")

def load_model(model_path, model_name):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Directory where the model is saved
        model_name: Name of the model
        
    Returns:
        dict: Model dictionary with model and metadata
    """
    print(f"Loading model {model_name}...")
    
    # Load the model
    model_dict = joblib.load(os.path.join(model_path, f"{model_name}.pkl"))
    print(f"Model loaded from {os.path.join(model_path, model_name)}.pkl")
    
    return model_dict

def train_and_evaluate_all_models(X_train, y_train, X_test, y_test, optimize=False, output_dir=None):
    """
    Train and evaluate multiple models on the same dataset.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        optimize: Whether to optimize hyperparameters
        output_dir: Directory to save models and results
        
    Returns:
        tuple: (models, results) - Trained models and evaluation results
    """
    print("Training and evaluating all models...")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Split training data for validation (used in hyperparameter optimization)
    if optimize:
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        X_train_opt, X_val, y_train_opt, y_val = X_train, None, y_train, None
    
    # Train models
    models = {}
    results = {}
    
    # Logistic Regression (baseline)
    lr_model = train_logistic_regression(X_train, y_train)
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    if output_dir:
        visualize_model_performance(results['logistic_regression'], 
                                   os.path.join(output_dir, 'visualizations'))
        save_model(lr_model, os.path.join(output_dir, 'models'), 
                  'logistic_regression', {'type': 'baseline'})
    
    # Random Forest
    if optimize:
        rf_params = optimize_hyperparameters(X_train_opt, y_train_opt, X_val, y_val, 
                                           'random_forest', n_trials=50)
    else:
        rf_params = None
    
    rf_model = train_random_forest(X_train, y_train, rf_params)
    models['random_forest'] = rf_model
    results['random_forest'] = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    if output_dir:
        visualize_model_performance(results['random_forest'], 
                                   os.path.join(output_dir, 'visualizations'))
        save_model(rf_model, os.path.join(output_dir, 'models'), 
                  'random_forest', {'params': rf_params})
    
    # XGBoost
    if optimize:
        xgb_params = optimize_hyperparameters(X_train_opt, y_train_opt, X_val, y_val, 
                                            'xgboost', n_trials=50)
    else:
        xgb_params = None
    
    xgb_model = train_xgboost(X_train, y_train, xgb_params)
    models['xgboost'] = xgb_model
    results['xgboost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    if output_dir:
        visualize_model_performance(results['xgboost'], 
                                   os.path.join(output_dir, 'visualizations'))
        save_model(xgb_model, os.path.join(output_dir, 'models'), 
                  'xgboost', {'params': xgb_params})
    
    # LightGBM
    if optimize:
        lgb_params = optimize_hyperparameters(X_train_opt, y_train_opt, X_val, y_val, 
                                            'lightgbm', n_trials=50)
    else:
        lgb_params = None
    
    lgb_model = train_lightgbm(X_train, y_train, lgb_params)
    models['lightgbm'] = lgb_model
    results['lightgbm'] = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    
    if output_dir:
        visualize_model_performance(results['lightgbm'], 
                                   os.path.join(output_dir, 'visualizations'))
        save_model(lgb_model, os.path.join(output_dir, 'models'), 
                  'lightgbm', {'params': lgb_params})
    
    # Create ensemble
    ensemble_models = [models['random_forest'], models['xgboost'], models['lightgbm']]
    
    # Determine weights based on AUC-ROC scores
    weights = [
        results['random_forest']['auc_roc'],
        results['xgboost']['auc_roc'],
        results['lightgbm']['auc_roc']
    ]
    
    ensemble = create_ensemble(ensemble_models, X_train, y_train, weights)
    models['ensemble'] = ensemble
    
    # Evaluate ensemble
    y_pred, y_pred_proba = predict_ensemble(ensemble, X_test)
    
    # Calculate metrics for ensemble
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print ensemble results
    print("Ensemble Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Store ensemble results
    results['ensemble'] = {
        'model_name': 'Ensemble',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'weights': weights
    }
    
    if output_dir:
        visualize_model_performance(results['ensemble'], 
                                   os.path.join(output_dir, 'visualizations'))
        save_model(ensemble, os.path.join(output_dir, 'models'), 
                  'ensemble', {'weights': weights})
    
    # Compare all models
    if output_dir:
        compare_models(results, os.path.join(output_dir, 'visualizations'))
    
    return models, results

def compare_models(results, output_dir=None):
    """
    Compare performance of multiple models.
    
    Args:
        results: Dictionary of evaluation results for each model
        output_dir: Directory to save the comparison visualizations
    """
    print("Comparing model performance...")
    
    # Extract model names and metrics
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_rocs = []
    
    for model_name, result in results.items():
        model_names.append(model_name)
        accuracies.append(result['accuracy'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        f1_scores.append(result['f1_score'])
        auc_rocs.append(result['auc_roc'])
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'AUC-ROC': auc_rocs
    })
    
    print("Model Comparison:")
    print(comparison_df)
    
    # Create bar plots for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, data=comparison_df)
        plt.title(f'Comparison of {metric}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    
    plt.close()
    
    # Create a radar chart for overall comparison
    plt.figure(figsize=(10, 8))
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw the chart for each model
    for i, model in enumerate(model_names):
        values = comparison_df.iloc[i, 1:].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison - All Metrics')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'model_comparison_radar.png'))
    
    plt.close()

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    import numpy as np
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features.resampling import apply_smote_tomek
    
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.join(base_dir, "models")
    
    print(f"Loading data from {processed_dir}...")
    
    # Load preprocessed data
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # Load feature names
    with open(os.path.join(processed_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().splitlines()
    
    print(f"Data loaded successfully.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # For large datasets, we can use a subset for training to speed up the process
    # while still maintaining the class distribution
    print("Taking a balanced subset of the training data for faster processing...")
    fraud_indices = np.where(y_train == 1)[0]
    non_fraud_indices = np.where(y_train == 0)[0]
    
    # Use all fraud samples and a random subset of non-fraud samples
    n_non_fraud = min(len(non_fraud_indices), 100000)  # Limit to 100k non-fraud samples
    selected_non_fraud = np.random.choice(non_fraud_indices, n_non_fraud, replace=False)
    
    # Combine indices and extract the subset
    selected_indices = np.concatenate([fraud_indices, selected_non_fraud])
    X_train_subset = X_train[selected_indices]
    y_train_subset = y_train[selected_indices]
    
    print(f"Subset training set shape: {X_train_subset.shape}")
    print(f"Subset class distribution: {np.bincount(y_train_subset)}")
    
    # Apply SMOTE-Tomek to handle imbalance
    print("Applying SMOTE-Tomek to handle class imbalance...")
    X_train_resampled, y_train_resampled = apply_smote_tomek(X_train_subset, y_train_subset, sampling_strategy=0.1)
    
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
    
    # Train and evaluate models with optimized threshold
    print(f"Training models and saving to {output_dir}...")
    models, results = train_and_evaluate_all_models(
        X_train_resampled, y_train_resampled, X_test, y_test, 
        optimize=True, output_dir=output_dir  # Set optimize to True to find optimal thresholds
    )
    
    print("Model training and evaluation complete.")
