"""
Visualization module for financial fraud detection.
This module provides functions for visualizing data and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_class_distribution(y, title="Class Distribution", output_dir=None):
    """
    Plot the distribution of classes in the target variable.
    
    Args:
        y: Target variable
        title: Plot title
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Count classes
    class_counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    
    # Create bar plot
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add percentage labels
    for i, count in enumerate(class_counts.values):
        percentage = 100 * count / total
        ax.text(i, count + (total * 0.01), f"{percentage:.2f}%", ha='center')
    
    # Add labels and title
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks([0, 1], ["Normal (0)", "Fraud (1)"])
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "class_distribution.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_feature_distributions(X, feature_names, y=None, n_features=10, output_dir=None):
    """
    Plot the distributions of the most important features.
    
    Args:
        X: Feature matrix
        feature_names: Names of the features
        y: Target variable for coloring
        n_features: Number of features to plot
        output_dir: Directory to save the plots
    """
    # Convert to DataFrame for easier plotting
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
    
    # Add target variable if provided
    if y is not None:
        X_df['target'] = y
    
    # Select top features
    if len(feature_names) > n_features:
        selected_features = feature_names[:n_features]
    else:
        selected_features = feature_names
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot each feature
    for feature in selected_features:
        plt.figure(figsize=(12, 6))
        
        if y is not None:
            # Plot distribution by class
            sns.histplot(data=X_df, x=feature, hue='target', kde=True, element="step", common_norm=False)
            plt.title(f"Distribution of {feature} by Class")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend(title="Class", labels=["Normal", "Fraud"])
        else:
            # Plot overall distribution
            sns.histplot(data=X_df, x=feature, kde=True)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
        
        # Save plot if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"feature_dist_{feature}.png"))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

def plot_correlation_matrix(X, feature_names, output_dir=None):
    """
    Plot the correlation matrix of features.
    
    Args:
        X: Feature matrix
        feature_names: Names of the features
        output_dir: Directory to save the plot
    """
    # Convert to DataFrame for easier plotting
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
    
    # Calculate correlation matrix
    corr_matrix = X_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    
    plt.title("Feature Correlation Matrix")
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(y_true, y_pred, output_dir=None):
    """
    Plot the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0.5, 1.5], ["Normal", "Fraud"])
    plt.yticks([0.5, 1.5], ["Normal", "Fraud"])
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_roc_curve(y_true, y_pred_proba, output_dir=None):
    """
    Plot the ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save the plot
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, output_dir=None):
    """
    Plot the Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save the plot
    """
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    return pr_auc

def plot_feature_importance(model, feature_names, n_features=20, output_dir=None):
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of the features
        n_features: Number of top features to plot
        output_dir: Directory to save the plot
    """
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not have feature_importances_ or coef_ attribute.")
        return
    
    # Create DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Select top features
    if len(feature_names) > n_features:
        feature_importance_df = feature_importance_df.head(n_features)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    
    plt.title(f"Top {n_features} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_shap_summary(shap_values, X, feature_names=None, output_dir=None):
    """
    Plot SHAP summary.
    
    Args:
        shap_values: SHAP values
        X: Feature matrix
        feature_names: Names of the features
        output_dir: Directory to save the plot
    """
    # Plot SHAP summary
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    plt.title("SHAP Feature Importance")
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "shap_summary.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_shap_dependence(shap_values, X, feature_idx, feature_names=None, interaction_idx=None, output_dir=None):
    """
    Plot SHAP dependence plot.
    
    Args:
        shap_values: SHAP values
        X: Feature matrix
        feature_idx: Index of the feature to plot
        feature_names: Names of the features
        interaction_idx: Index of the interaction feature
        output_dir: Directory to save the plot
    """
    # Get feature name
    if feature_names is not None:
        feature_name = feature_names[feature_idx]
    else:
        feature_name = f"Feature {feature_idx}"
    
    # Get interaction feature name
    if interaction_idx is not None:
        if feature_names is not None:
            interaction_name = feature_names[interaction_idx]
        else:
            interaction_name = f"Feature {interaction_idx}"
    else:
        interaction_name = None
    
    # Plot SHAP dependence plot
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(feature_idx, shap_values, X, interaction_index=interaction_idx, 
                         feature_names=feature_names, show=False)
    
    if interaction_name:
        plt.title(f"SHAP Dependence Plot: {feature_name} (interaction with {interaction_name})")
    else:
        plt.title(f"SHAP Dependence Plot: {feature_name}")
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if interaction_name:
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature_name}_{interaction_name}.png"), 
                       bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature_name}.png"), 
                       bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_tsne(X, y, perplexity=30, n_iter=1000, output_dir=None):
    """
    Plot t-SNE visualization of the data.
    
    Args:
        X: Feature matrix
        y: Target variable
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        output_dir: Directory to save the plot
    """
    # Apply t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame for plotting
    tsne_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'class': y
    })
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x', y='y', hue='class', data=tsne_df, palette=['blue', 'red'], alpha=0.7)
    
    plt.title(f"t-SNE Visualization (perplexity={perplexity}, iterations={n_iter})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Class", labels=["Normal", "Fraud"])
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "tsne_visualization.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_pca(X, y, n_components=2, output_dir=None):
    """
    Plot PCA visualization of the data.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_components: Number of PCA components
        output_dir: Directory to save the plot
    """
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1] if n_components > 1 else np.zeros(X_pca.shape[0]),
        'class': y
    })
    
    # Plot PCA
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df, palette=['blue', 'red'], alpha=0.7)
    
    plt.title(f"PCA Visualization (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    if n_components > 1:
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(title="Class", labels=["Normal", "Fraud"])
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "pca_visualization.png"))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_interactive_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Create an interactive ROC curve using Plotly.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.4f})',
        line=dict(color='darkorange', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=800,
        height=600
    )
    
    # Update axes
    fig.update_xaxes(range=[0, 1], constrain='domain')
    fig.update_yaxes(range=[0, 1], scaleanchor='x', scaleratio=1)
    
    return fig

def create_interactive_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Create an interactive Precision-Recall curve using Plotly.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig = go.Figure()
    
    # Add Precision-Recall curve
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'{model_name} (AUC = {pr_auc:.4f})',
        line=dict(color='darkorange', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Precision-Recall Curve - {model_name}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=800,
        height=600
    )
    
    # Update axes
    fig.update_xaxes(range=[0, 1], constrain='domain')
    fig.update_yaxes(range=[0, 1], scaleanchor='x', scaleratio=1)
    
    return fig

def create_interactive_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Create an interactive confusion matrix using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm / cm.sum() * 100
    
    # Create annotations
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(dict(
                x=j, y=i,
                text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
            ))
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Normal", "Fraud"],
        y=["Normal", "Fraud"],
        colorscale="Blues",
        showscale=False
    ))
    
    # Add annotations
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        annotations=annotations,
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual"),
        width=600,
        height=600
    )
    
    return fig

def create_interactive_feature_importance(feature_names, importances, model_name="Model", n_features=20):
    """
    Create an interactive feature importance plot using Plotly.
    
    Args:
        feature_names: Names of the features
        importances: Feature importance values
        model_name: Name of the model
        n_features: Number of top features to plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Select top features
    if len(feature_names) > n_features:
        feature_importance_df = feature_importance_df.head(n_features)
    
    # Create figure
    fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {n_features} Feature Importance - {model_name}'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        width=800,
        height=600
    )
    
    return fig

def create_interactive_shap_summary(shap_values, X, feature_names=None, model_name="Model"):
    """
    Create an interactive SHAP summary plot using Plotly.
    
    Args:
        shap_values: SHAP values
        X: Feature matrix
        feature_names: Names of the features
        model_name: Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(0)
    
    # Create DataFrame with feature names and mean absolute SHAP values
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': mean_abs_shap
    })
    
    # Sort by SHAP value
    shap_df = shap_df.sort_values('SHAP Value', ascending=False)
    
    # Create figure
    fig = px.bar(
        shap_df,
        x='SHAP Value',
        y='Feature',
        orientation='h',
        title=f'SHAP Feature Importance - {model_name}'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        width=800,
        height=600
    )
    
    return fig

def create_dashboard_components(model, X, y, y_pred, y_pred_proba, feature_names, model_name="Model"):
    """
    Create components for a dashboard.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        feature_names: Names of the features
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of dashboard components
    """
    # Create interactive plots
    roc_curve_fig = create_interactive_roc_curve(y, y_pred_proba, model_name)
    pr_curve_fig = create_interactive_precision_recall_curve(y, y_pred_proba, model_name)
    confusion_matrix_fig = create_interactive_confusion_matrix(y, y_pred, model_name)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = None
    
    if importances is not None:
        feature_importance_fig = create_interactive_feature_importance(
            feature_names, importances, model_name
        )
    else:
        feature_importance_fig = None
    
    # Return dashboard components
    return {
        'roc_curve': roc_curve_fig,
        'pr_curve': pr_curve_fig,
        'confusion_matrix': confusion_matrix_fig,
        'feature_importance': feature_importance_fig
    }

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.preprocess import preprocess_data
    from models.train_model import load_model
    
    # Load preprocessed data
    processed_dir = "../../data/processed"
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # Load feature names
    with open(os.path.join(processed_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().splitlines()
    
    # Load trained model
    model_dir = "../../models"
    model_dict = load_model(os.path.join(model_dir, 'models'), 'xgboost')
    model = model_dict['model']
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Create visualizations
    output_dir = os.path.join(model_dir, 'visualizations')
    
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_roc_curve(y_test, y_pred_proba, output_dir)
    plot_precision_recall_curve(y_test, y_pred_proba, output_dir)
    plot_feature_importance(model, feature_names, output_dir=output_dir)
    
    print("Visualizations created successfully.")
