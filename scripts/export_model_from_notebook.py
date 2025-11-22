"""
Script to train and export the XGBoost model from the Colab notebook.

This script replicates the training logic from loan_xgboost.ipynb and saves
the model to models/loan_xgb.pkl.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create models directory
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

def train_and_export_model(csv_path: Optional[str] = None):
    """
    Train the XGBoost model and export it.
    
    Args:
        csv_path: Path to the loan data CSV. If None, looks for data/Loan_default.csv
    """
    # Find CSV file
    if csv_path is None:
        csv_path = project_root / "data" / "Loan_default.csv"
        if not csv_path.exists():
            # Try alternative locations
            csv_path = project_root / "Loan_default.csv"
    
    if not csv_path or not Path(csv_path).exists():
        print(f"ERROR: Loan data CSV not found. Please provide the path to Loan_default.csv")
        print(f"Expected locations:")
        print(f"  - data/Loan_default.csv")
        print(f"  - Loan_default.csv (project root)")
        print(f"\nYou can download the dataset or provide the path via:")
        print(f"  python scripts/export_model_from_notebook.py --csv-path /path/to/Loan_default.csv")
        return False
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Preprocessing data...")
    # Drop rows with missing target
    df = df.dropna(subset=['HasCoSigner', 'Default'])
    # Drop LoanID if present
    if 'LoanID' in df.columns:
        df = df.drop(columns=['LoanID'])
    
    # Separate features and target
    X = df.drop("Default", axis=1)
    y = df["Default"]
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns
    
    print(f"Categorical columns: {list(categorical_cols)}")
    print(f"Numeric columns: {list(numeric_cols)}")
    
    # Preprocess
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )
    
    # Model with balancing
    xgb = XGBClassifier(
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),
        n_estimators=500,
        learning_rate=0.05,
        eval_metric="logloss"
    )
    
    # Build pipeline
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("classifier", xgb)
    ])
    
    # Train/test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training model (this may take a few minutes)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = models_dir / "loan_xgb.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and export XGBoost loan prediction model")
    parser.add_argument("--csv-path", type=str, help="Path to Loan_default.csv file")
    
    args = parser.parse_args()
    
    success = train_and_export_model(args.csv_path)
    sys.exit(0 if success else 1)

