"""
Resampling module for handling imbalanced data in fraud detection.
This module implements various resampling techniques to address class imbalance.
"""

import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def apply_smote(X, y, random_state=42, sampling_strategy=0.1):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        sampling_strategy: Ratio of minority to majority class after resampling
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Applying SMOTE resampling...")
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def apply_smote_tomek(X, y, random_state=42, sampling_strategy=0.1):
    """
    Apply SMOTE-Tomek to oversample the minority class and clean the resulting data.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        sampling_strategy: Ratio of minority to majority class after resampling
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Applying SMOTE-Tomek resampling...")
    smote_tomek = SMOTETomek(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def apply_smote_enn(X, y, random_state=42, sampling_strategy=0.1):
    """
    Apply SMOTE-ENN to oversample the minority class and clean the resulting data.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        sampling_strategy: Ratio of minority to majority class after resampling
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Applying SMOTE-ENN resampling...")
    smote_enn = SMOTEENN(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def apply_adasyn(X, y, random_state=42, sampling_strategy=0.1):
    """
    Apply ADASYN to adaptively oversample the minority class.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        sampling_strategy: Ratio of minority to majority class after resampling
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Applying ADASYN resampling...")
    adasyn = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def apply_hybrid_resampling(X, y, random_state=42, sampling_strategy=0.1):
    """
    Apply a hybrid resampling approach combining oversampling and undersampling.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        sampling_strategy: Ratio of minority to majority class after resampling
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Applying hybrid resampling...")
    
    # First apply SMOTE to oversample the minority class
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_oversampled, y_oversampled = smote.fit_resample(X, y)
    
    # Then apply random undersampling to reduce the majority class
    undersampler = RandomUnderSampler(random_state=random_state, sampling_strategy=0.5)
    X_resampled, y_resampled = undersampler.fit_resample(X_oversampled, y_oversampled)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def visualize_resampling_results(X, y, X_resampled, y_resampled, feature_names=None, output_dir=None):
    """
    Visualize the distribution of features before and after resampling.
    
    Args:
        X: Original feature matrix
        y: Original target vector
        X_resampled: Resampled feature matrix
        y_resampled: Resampled target vector
        feature_names: Names of the features
        output_dir: Directory to save the visualization
    """
    print("Visualizing resampling results...")
    
    # Convert to DataFrame for easier plotting
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Select a subset of features if there are too many
    if len(feature_names) > 10:
        selected_features = feature_names[:10]
        X_subset = X[:, :10]
        X_resampled_subset = X_resampled[:, :10]
    else:
        selected_features = feature_names
        X_subset = X
        X_resampled_subset = X_resampled
    
    # Create DataFrames
    df_original = pd.DataFrame(X_subset, columns=selected_features)
    df_original['Class'] = y
    df_original['Dataset'] = 'Original'
    
    df_resampled = pd.DataFrame(X_resampled_subset, columns=selected_features)
    df_resampled['Class'] = y_resampled
    df_resampled['Dataset'] = 'Resampled'
    
    # Combine the DataFrames
    df_combined = pd.concat([df_original, df_resampled])
    
    # Create plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        if i < len(axes):
            sns.boxplot(x='Class', y=feature, hue='Dataset', data=df_combined, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('Class (0=Normal, 1=Fraud)')
            axes[i].set_ylabel(feature)
    
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'resampling_visualization.png'))
        print(f"Visualization saved to {os.path.join(output_dir, 'resampling_visualization.png')}")
    
    plt.close()

def compare_resampling_techniques(X, y, feature_names=None, random_state=42, output_dir=None):
    """
    Compare different resampling techniques and visualize their results.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Names of the features
        random_state: Random seed for reproducibility
        output_dir: Directory to save the results
        
    Returns:
        dict: Dictionary containing resampled datasets
    """
    print("Comparing resampling techniques...")
    
    # Apply different resampling techniques
    X_smote, y_smote = apply_smote(X, y, random_state=random_state)
    X_smote_tomek, y_smote_tomek = apply_smote_tomek(X, y, random_state=random_state)
    X_smote_enn, y_smote_enn = apply_smote_enn(X, y, random_state=random_state)
    X_adasyn, y_adasyn = apply_adasyn(X, y, random_state=random_state)
    X_hybrid, y_hybrid = apply_hybrid_resampling(X, y, random_state=random_state)
    
    # Visualize each resampling technique
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        visualize_resampling_results(X, y, X_smote, y_smote, feature_names, 
                                     os.path.join(output_dir, 'smote'))
        
        visualize_resampling_results(X, y, X_smote_tomek, y_smote_tomek, feature_names, 
                                     os.path.join(output_dir, 'smote_tomek'))
        
        visualize_resampling_results(X, y, X_smote_enn, y_smote_enn, feature_names, 
                                     os.path.join(output_dir, 'smote_enn'))
        
        visualize_resampling_results(X, y, X_adasyn, y_adasyn, feature_names, 
                                     os.path.join(output_dir, 'adasyn'))
        
        visualize_resampling_results(X, y, X_hybrid, y_hybrid, feature_names, 
                                     os.path.join(output_dir, 'hybrid'))
    
    # Return all resampled datasets
    return {
        'original': (X, y),
        'smote': (X_smote, y_smote),
        'smote_tomek': (X_smote_tomek, y_smote_tomek),
        'smote_enn': (X_smote_enn, y_smote_enn),
        'adasyn': (X_adasyn, y_adasyn),
        'hybrid': (X_hybrid, y_hybrid)
    }

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create a synthetic imbalanced dataset
    X, y = make_classification(
        n_samples=10000, n_features=10, n_informative=5, n_redundant=3,
        n_classes=2, weights=[0.99, 0.01], random_state=42
    )
    
    # Compare resampling techniques
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    resampled_datasets = compare_resampling_techniques(
        X, y, feature_names=feature_names, output_dir="../data/resampling_results"
    )
    
    print("Resampling comparison complete.")
