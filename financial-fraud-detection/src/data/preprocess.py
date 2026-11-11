"""
Data preprocessing module for financial fraud detection.
This module handles loading, cleaning, and preprocessing the financial dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import category_encoders as ce
import os
import joblib

def load_data(file_path):
    """
    Load the financial dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def analyze_data(df):
    """
    Analyze the dataset and print basic statistics.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing data analysis results
    """
    print("Analyzing data...")
    
    # Basic information
    print(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values[missing_values > 0]}")
    
    # Class distribution
    if 'Class' in df.columns:
        fraud_count = df['Class'].sum()
        total_count = len(df)
        fraud_ratio = fraud_count / total_count
        print(f"Fraud transactions: {fraud_count} ({fraud_ratio:.4%})")
        print(f"Normal transactions: {total_count - fraud_count} ({1-fraud_ratio:.4%})")
    
    # Data types
    print("Data types:")
    print(df.dtypes)
    
    # Return analysis results
    return {
        'shape': df.shape,
        'missing_values': missing_values.to_dict(),
        'fraud_ratio': fraud_ratio if 'is_fraud' in df.columns else None,
        'dtypes': df.dtypes.to_dict()
    }

def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Cleaning data...")
    
    # Create a copy to avoid modifying the original dataframe
    df_cleaned = df.copy()
    
    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype == 'object':
            # For categorical columns, fill with the most frequent value
            df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
        else:
            # For numerical columns, fill with the median
            df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
    
    # Convert timestamp columns if they exist
    if 'trans_date_trans_time' in df_cleaned.columns:
        df_cleaned['trans_date_trans_time'] = pd.to_datetime(df_cleaned['trans_date_trans_time'])
    
    return df_cleaned

def engineer_features(df):
    """
    Create new features to improve model performance.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    print("Engineering features...")
    
    # Create a copy to avoid modifying the original dataframe
    df_featured = df.copy()
    
    # Time-based features if timestamp column exists
    if 'trans_date_trans_time' in df_featured.columns:
        df_featured['hour'] = df_featured['trans_date_trans_time'].dt.hour
        df_featured['day'] = df_featured['trans_date_trans_time'].dt.day
        df_featured['day_of_week'] = df_featured['trans_date_trans_time'].dt.dayofweek
        df_featured['month'] = df_featured['trans_date_trans_time'].dt.month
        
        # Group transactions by customer and sort by time
        if 'cc_num' in df_featured.columns:
            # Create a copy of the dataframe sorted by customer and time
            df_sorted = df_featured.sort_values(['cc_num', 'trans_date_trans_time']).copy()
            
            # Calculate time difference between consecutive transactions for each customer
            df_sorted['prev_time'] = df_sorted.groupby('cc_num')['trans_date_trans_time'].shift(1)
            df_sorted['time_diff'] = (df_sorted['trans_date_trans_time'] - df_sorted['prev_time']).dt.total_seconds() / 3600  # in hours
            
            # Merge the time difference back to the original dataframe
            df_featured = df_featured.merge(df_sorted[['cc_num', 'trans_date_trans_time', 'time_diff']], 
                                           on=['cc_num', 'trans_date_trans_time'], 
                                           how='left')
    
    # Amount-based features
    if 'amt' in df_featured.columns:
        # Calculate transaction amount statistics for each customer
        customer_stats = df_featured.groupby('cc_num')['amt'].agg(['mean', 'std', 'max']).reset_index()
        customer_stats.columns = ['cc_num', 'customer_avg_amt', 'customer_std_amt', 'customer_max_amt']
        
        # Merge statistics back to the original dataframe
        df_featured = df_featured.merge(customer_stats, on='cc_num', how='left')
        
        # Calculate amount ratio compared to customer average
        df_featured['amt_to_avg_ratio'] = df_featured['amt'] / df_featured['customer_avg_amt']
        
        # Flag high-value transactions (above 2 standard deviations)
        df_featured['is_high_value'] = ((df_featured['amt'] > df_featured['customer_avg_amt'] + 2 * df_featured['customer_std_amt']) 
                                       & (df_featured['customer_std_amt'] > 0)).astype(int)
    
    # Merchant-based features
    if 'merchant' in df_featured.columns:
        # Calculate fraud rate by merchant
        merchant_fraud = df_featured.groupby('merchant')['Class'].agg(['mean', 'count']).reset_index()
        merchant_fraud.columns = ['merchant', 'merchant_fraud_rate', 'merchant_trans_count']
        
        # Merge merchant statistics back to the original dataframe
        df_featured = df_featured.merge(merchant_fraud, on='merchant', how='left')
    
    # Category-based features
    if 'category' in df_featured.columns:
        # Calculate fraud rate by category
        category_fraud = df_featured.groupby('category')['Class'].agg(['mean', 'count']).reset_index()
        category_fraud.columns = ['category', 'category_fraud_rate', 'category_trans_count']
        
        # Merge category statistics back to the original dataframe
        df_featured = df_featured.merge(category_fraud, on='category', how='left')
    
    return df_featured

def prepare_features_targets(df, target_col='Class'):
    """
    Separate features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of the target column
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    print("Preparing features and target...")
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != target_col]
    
    # Remove non-feature columns
    exclude_cols = [target_col, 'trans_date_trans_time', 'cc_num', 'prev_time']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate features and target
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, categorical_cols, numerical_cols

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Create a preprocessing pipeline for categorical and numerical features.
    
    Args:
        categorical_cols (list): List of categorical column names
        numerical_cols (list): List of numerical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print("Creating preprocessing pipeline...")
    
    # Categorical features preprocessing
    categorical_transformer = Pipeline(steps=[
        ('target_encoder', ce.TargetEncoder())  # Target encoding works well for fraud detection
    ])
    
    # Numerical features preprocessing
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def preprocess_data(file_path, test_size=0.2, random_state=42, output_dir=None):
    """
    Main function to preprocess the data.
    
    Args:
        file_path (str): Path to the input CSV file
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        output_dir (str): Directory to save the processed data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Load data
    df = load_data(file_path)
    
    # Analyze data
    analyze_data(df)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Engineer features
    df_featured = engineer_features(df_cleaned)
    
    # Prepare features and target
    X, y, categorical_cols, numerical_cols = prepare_features_targets(df_featured)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Save processed data if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
        
        # Save processed data
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train_processed)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test_processed)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)
        
        # Save column names for reference
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(X.columns))
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, X_train.columns

if __name__ == "__main__":
    # Example usage
    import os
    
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(base_dir, "Synthetic_Financial_datasets_log.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    
    print(f"Looking for dataset at: {data_file}")
    print(f"Output directory: {output_dir}")
    
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        data_file, output_dir=output_dir
    )
    
    print(f"Preprocessing complete. Processed data saved to {output_dir}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
