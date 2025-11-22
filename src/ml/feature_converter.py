"""
Feature Converter Module - FIXED VERSION

This module converts raw user inputs into the exact feature format expected by the ML model.
The ML model expects ONLY the 16 structured loan features, nothing more.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# Define the EXACT features the ML model expects (in order)
ML_MODEL_FEATURES = [
    "LoanID",
    "Age", 
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]


def convert_to_ml_features(raw_inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw user inputs to ML model feature DataFrame.
    
    CRITICAL: This function ONLY extracts the 16 structured features the model was trained on.
    It completely ignores any unconventional data (mobile, psychometric, etc.)
    
    Args:
        raw_inputs: Dictionary with user inputs (may contain extra fields)
        
    Returns:
        DataFrame with EXACTLY the features the ML model expects
    """
    # Extract only the ML model features, using safe defaults for missing values
    ml_data = {}
    
    for feature in ML_MODEL_FEATURES:
        if feature in raw_inputs:
            ml_data[feature] = raw_inputs[feature]
        else:
            # Provide safe defaults based on feature type
            ml_data[feature] = _get_default_value(feature)
    
    # Create DataFrame with exact column order
    df = pd.DataFrame([ml_data])
    
    # Ensure columns are in the exact order expected by the model
    df = df[ML_MODEL_FEATURES]
    
    logger.info(f"Created ML feature DataFrame with {len(df.columns)} columns: {list(df.columns)}")
    
    return df


def extract_ml_features_only(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only ML-relevant structured features from a full profile.
    
    This is used when you have a complete profile with unconventional data,
    but need to extract just the ML features.
    
    Args:
        profile: Full user profile with demographics, mobile_metadata, etc.
        
    Returns:
        Dictionary with only the ML model features
    """
    ml_features = {}
    
    # Extract from demographics
    demo = profile.get("demographics", {})
    ml_features["Age"] = demo.get("age", 30)
    ml_features["Income"] = demo.get("monthly_income", 0.0)
    ml_features["Education"] = demo.get("education_level", "High School")
    ml_features["EmploymentType"] = demo.get("employment_type", "Full-time")
    ml_features["MaritalStatus"] = demo.get("marital_status", "Single")
    
    # Extract loan-specific features (should be in profile root or loan_history)
    loan = profile.get("loan_history", {})
    ml_features["LoanID"] = profile.get("user_id", "UNKNOWN")
    ml_features["LoanAmount"] = loan.get("current_loan_amount", 0.0)
    ml_features["CreditScore"] = profile.get("credit_score", 650)
    ml_features["MonthsEmployed"] = demo.get("months_employed", 24)
    ml_features["NumCreditLines"] = loan.get("num_credit_lines", 2)
    ml_features["InterestRate"] = loan.get("interest_rate", 10.0)
    ml_features["LoanTerm"] = loan.get("loan_term", 36)
    ml_features["DTIRatio"] = profile.get("dti_ratio", 0.3)
    ml_features["HasMortgage"] = profile.get("has_mortgage", "No")
    ml_features["HasDependents"] = profile.get("has_dependents", "No")
    ml_features["LoanPurpose"] = loan.get("loan_purpose", "Other")
    ml_features["HasCoSigner"] = profile.get("has_cosigner", "No")
    
    return ml_features


def _get_default_value(feature_name: str) -> Any:
    """
    Provide safe default values for missing features based on feature name and type.
    """
    # String/categorical features
    categorical_defaults = {
        "LoanID": "UNKNOWN",
        "Education": "High School",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "No",
        "LoanPurpose": "Other",
        "HasCoSigner": "No"
    }
    
    if feature_name in categorical_defaults:
        return categorical_defaults[feature_name]
    
    # Numeric features - return appropriate numeric defaults
    numeric_defaults = {
        "Age": 30,
        "Income": 50000.0,
        "LoanAmount": 10000.0,
        "CreditScore": 650,
        "MonthsEmployed": 24,
        "NumCreditLines": 2,
        "InterestRate": 10.0,
        "LoanTerm": 36,
        "DTIRatio": 0.3
    }
    
    return numeric_defaults.get(feature_name, 0)


def validate_ml_features(df: pd.DataFrame) -> tuple[bool, List[str]]:
    """
    Validate that the DataFrame has exactly the features the ML model expects.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check column count
    if len(df.columns) != len(ML_MODEL_FEATURES):
        errors.append(f"Expected {len(ML_MODEL_FEATURES)} columns, got {len(df.columns)}")
    
    # Check column names and order
    for i, expected_col in enumerate(ML_MODEL_FEATURES):
        if i >= len(df.columns):
            errors.append(f"Missing column: {expected_col}")
        elif df.columns[i] != expected_col:
            errors.append(f"Column mismatch at position {i}: expected '{expected_col}', got '{df.columns[i]}'")
    
    # Check for extra columns
    extra_cols = set(df.columns) - set(ML_MODEL_FEATURES)
    if extra_cols:
        errors.append(f"Unexpected extra columns: {extra_cols}")
    
    return len(errors) == 0, errors


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value == "NA" or value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    if value == "NA" or value is None or value == "":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default