"""Encode categorical vars, split train/test, and balance classes if requested.

    # ! Important Note:
    # ! Balancing the dataset wasn't mentioned in the paper.

    They likely didn't apply balancing because the goal was to simulate real-world fraud detection, where fraud cases are extremely rare.

    Oversampling (like SMOTE) can introduce bias and make the task unrealistically easy, which is not representative of actual deployment conditions.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """Load data from CSV file."""
    print(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def encode_categoricals(df):
    """
    Encode categorical variables using LabelEncoder.
    According to the paper, only three columns were treated as categorical:
    'type', 'nameOrig', and 'nameDest'
    """
    categorical_cols = ['type', 'nameOrig', 'nameDest']
    df_encoded = df.copy()
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_encoded.columns:
            print(f"Encoding categorical column: {col}")
            df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

def split_data(df, target_column='isFraud'):
    """Split data into training and testing sets with stratification."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud percentage: {y.mean() * 100:.2f}%")
    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train, sampling_strategy=1):
    """
    Apply SMOTE to balance the training dataset.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - sampling_strategy: Target ratio of minority to majority class (default: 1)
    
    Returns:
    - X_resampled: Balanced training features
    - y_resampled: Balanced training labels
    """
    print("Before balancing:")
    print(y_train.value_counts())
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("After balancing:")
    print(pd.Series(y_resampled).value_counts())
    
    return X_resampled, y_resampled

def save_data(X_train, X_test, y_train, y_test, balanced=False, save_dir='data/processed'):
    """Save preprocessed data as csv files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create prefix for balanced data
    prefix = "balanced_" if balanced else ""
    
    # Save individual data components
    X_train.to_csv(os.path.join(save_dir, f"{prefix}X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, f"{prefix}X_test.csv"), index=False)
    
    # Convert y to DataFrame if Series
    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(y_train, columns=['isFraud'])
    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame(y_test, columns=['isFraud'])
        
    y_train.to_csv(os.path.join(save_dir, f"{prefix}y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, f"{prefix}y_test.csv"), index=False)
    
    # Save merged data (X and y together)
    train_merged = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_merged = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    train_merged.to_csv(os.path.join(save_dir, f"{prefix}train_merged.csv"), index=False)
    test_merged.to_csv(os.path.join(save_dir, f"{prefix}test_merged.csv"), index=False)
    
    print(f"Saved preprocessed data to {save_dir}")

def main():
    """Main preprocessing pipeline following the paper's methodology."""
    # 1. Load data
    df = load_data('data/raw/ps_raw.csv')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # 2. Check for class imbalance
    target_column = 'isFraud'  # The PaySim dataset uses 'isFraud' as the target column
    class_counts = df[target_column].value_counts()
    print(f"Class distribution:\n{class_counts}")
    
    fraud_percentage = df[target_column].mean() * 100
    print(f"Fraud percentage: {fraud_percentage:.2f}%")
    
    # 3. Encode categorical variables (only type, nameOrig, nameDest as per the paper)
    df_encoded = encode_categoricals(df)
    
    # 4. Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df_encoded, target_column)
    
    # 5. Save original split data
    save_data(X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------------------------------
    # ! For the next two steps, we will not apply them as per the paper's methodology.
    # ! However, we will include them as optional steps for demonstration purposes.

    # ! Note: The paper didn't mention balancing, but we include it as an optional step
    # ! to demonstrate its effects on model performance. In production fraud detection,
    # ! maintaining natural class distribution is often preferred to reflect real-world conditions.

    # 6. Balance training data with SMOTE (if needed)
    # print("\nApplying SMOTE to handle class imbalance...")
    # X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    
    # 7. Save balanced data
    # save_data(
    #     pd.DataFrame(X_train_balanced, columns=X_train.columns),
    #     X_test,
    #     pd.Series(y_train_balanced, name=target_column),
    #     y_test,
    #     balanced=True
    # )
    
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
