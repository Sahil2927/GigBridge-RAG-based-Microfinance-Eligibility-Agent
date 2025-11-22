"""
Feature Converter Module

This module converts raw user inputs and profile data into the feature vector
expected by the ML model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def convert_profile_to_model_features(profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a user profile to model feature DataFrame.
    
    This function maps the profile structure to the features expected by the model.
    The model was trained on a CSV with columns that we need to infer or map.
    
    Args:
        profile: User profile dictionary with demographics, mobile_metadata, etc.
        
    Returns:
        DataFrame with features matching model input format
    """
    features = {}
    
    # Demographics
    demo = profile.get("demographics", {})
    features["Age"] = demo.get("age", 0)
    features["Gender"] = demo.get("gender", "unknown")
    features["MaritalStatus"] = demo.get("marital_status", "unknown")
    features["EducationLevel"] = demo.get("education_level", "unknown")
    features["Occupation"] = demo.get("occupation", "unknown")
    features["MonthlyIncome"] = demo.get("monthly_income", 0.0)
    features["IncomeSourceStability"] = demo.get("income_source_stability", 0.5)
    
    # Mobile metadata
    mobile = profile.get("mobile_metadata", {})
    features["AvgDailyCalls"] = _safe_float(mobile.get("avg_daily_calls", 0.0))
    features["UniqueContacts30d"] = _safe_int(mobile.get("unique_contacts_30d", 0))
    features["AirtimeTopupFrequency"] = _safe_float(mobile.get("airtime_topup_frequency", 0.0))
    features["AvgTopupAmount"] = _safe_float(mobile.get("avg_topup_amount", 0.0))
    features["DaysInactiveLast30"] = _safe_int(mobile.get("days_inactive_last_30", 0))
    
    # Psychometrics
    psych = profile.get("psychometrics", {})
    features["ConscientiousnessScore"] = _safe_float(psych.get("conscientiousness_score", 3.0))
    
    # Financial behavior
    fin = profile.get("financial_behavior", {})
    features["SavingsFrequency"] = _safe_float(fin.get("savings_frequency", 0.0))
    features["BillPaymentTimeliness"] = _safe_float(fin.get("bill_payment_timeliness", 0.5))
    features["WalletBalanceLows90d"] = _safe_int(fin.get("wallet_balance_lows_last_90d", 0))
    
    # Social network
    social = profile.get("social_network", {})
    shg_membership = social.get("shg_membership", False)
    features["SHGMembership"] = "Yes" if (shg_membership is True or shg_membership == "Yes") else "No"
    features["PeerMonitoringStrength"] = _safe_float(social.get("peer_monitoring_strength", 0.0))
    
    # Loan history
    loan = profile.get("loan_history", {})
    features["PreviousLoans"] = _safe_int(loan.get("previous_loans", 0))
    features["PreviousDefaults"] = _safe_int(loan.get("previous_defaults", 0))
    features["PreviousLatePayments"] = _safe_int(loan.get("previous_late_payments", 0))
    features["AvgRepaymentDelayDays"] = _safe_float(loan.get("avg_repayment_delay_days", 0.0))
    
    # HasCoSigner (default to No if not provided)
    features["HasCoSigner"] = profile.get("has_cosigner", "No")
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    return df


def convert_raw_inputs_to_features(raw_inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw user inputs (from UI) to model feature DataFrame.
    
    This is the main function that frontend should call. It accepts raw values
    like numbers, strings, and converts them to the model format.
    
    Args:
        raw_inputs: Dictionary with raw input values from UI
        
    Returns:
        DataFrame with features matching model input format
    """
    # Build a profile-like structure from raw inputs
    profile = {
        "demographics": {
            "age": raw_inputs.get("age", 30),
            "gender": raw_inputs.get("gender", "unknown"),
            "marital_status": raw_inputs.get("marital_status", "unknown"),
            "education_level": raw_inputs.get("education_level", "unknown"),
            "occupation": raw_inputs.get("occupation", "unknown"),
            "monthly_income": raw_inputs.get("monthly_income", 0.0),
            "income_source_stability": raw_inputs.get("income_source_stability", 0.5)
        },
        "mobile_metadata": {
            "avg_daily_calls": raw_inputs.get("avg_daily_calls", 0.0),
            "unique_contacts_30d": raw_inputs.get("unique_contacts_30d", 0),
            "airtime_topup_frequency": raw_inputs.get("airtime_topup_frequency", 0.0),
            "avg_topup_amount": raw_inputs.get("avg_topup_amount", 0.0),
            "days_inactive_last_30": raw_inputs.get("days_inactive_last_30", 0)
        },
        "psychometrics": {
            "conscientiousness_score": raw_inputs.get("conscientiousness_score", 3.0)
        },
        "financial_behavior": {
            "savings_frequency": raw_inputs.get("savings_frequency", 0.0),
            "bill_payment_timeliness": raw_inputs.get("bill_payment_timeliness", 0.5),
            "wallet_balance_lows_last_90d": raw_inputs.get("wallet_balance_lows_last_90d", 0)
        },
        "social_network": {
            "shg_membership": raw_inputs.get("shg_membership", False),
            "peer_monitoring_strength": raw_inputs.get("peer_monitoring_strength", 0.0)
        },
        "loan_history": {
            "previous_loans": raw_inputs.get("previous_loans", 0),
            "previous_defaults": raw_inputs.get("previous_defaults", 0),
            "previous_late_payments": raw_inputs.get("previous_late_payments", 0),
            "avg_repayment_delay_days": raw_inputs.get("avg_repayment_delay_days", 0.0)
        }
    }
    
    # Add has_cosigner if provided
    if "has_cosigner" in raw_inputs:
        profile["has_cosigner"] = raw_inputs["has_cosigner"]
    
    return convert_profile_to_model_features(profile)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value == "NA" or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    if value == "NA" or value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

