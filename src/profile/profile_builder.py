"""
Profile Builder Module

This module converts raw unconventional input data into the structured
profile schema required by the system.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_mobile_metadata(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate mobile metadata features from raw call logs and airtime data.
    
    Args:
        raw_data: Dictionary containing raw mobile data
        
    Returns:
        Dictionary of mobile metadata features
    """
    metadata = {}
    
    # Average daily calls
    if "call_logs_30d" in raw_data and isinstance(raw_data["call_logs_30d"], list):
        calls = raw_data["call_logs_30d"]
        total_calls = len(calls)
        metadata["avg_daily_calls"] = round(total_calls / 30.0, 2) if total_calls > 0 else 0.0
    else:
        metadata["avg_daily_calls"] = "NA"
    
    # Unique contacts in last 30 days
    if "call_logs_30d" in raw_data and isinstance(raw_data["call_logs_30d"], list):
        calls = raw_data["call_logs_30d"]
        unique_contacts = set()
        for call in calls:
            if isinstance(call, dict) and "contact" in call:
                unique_contacts.add(call["contact"])
        metadata["unique_contacts_30d"] = len(unique_contacts)
    else:
        metadata["unique_contacts_30d"] = "NA"
    
    # Days inactive in last 30 days
    if "call_logs_30d" in raw_data and isinstance(raw_data["call_logs_30d"], list):
        calls = raw_data["call_logs_30d"]
        active_days = set()
        for call in calls:
            if isinstance(call, dict) and "date" in call:
                try:
                    date_obj = datetime.fromisoformat(call["date"].split("T")[0])
                    active_days.add(date_obj.date())
                except:
                    pass
        metadata["days_inactive_last_30"] = max(0, 30 - len(active_days))
    else:
        metadata["days_inactive_last_30"] = "NA"
    
    # Airtime topup frequency and average amount
    if "airtime_topups_30d" in raw_data and isinstance(raw_data["airtime_topups_30d"], list):
        topups = raw_data["airtime_topups_30d"]
        if topups:
            metadata["airtime_topup_frequency"] = round(len(topups) / 30.0, 2)
            amounts = [t.get("amount", 0) for t in topups if isinstance(t, dict)]
            if amounts:
                metadata["avg_topup_amount"] = round(statistics.mean(amounts), 2)
            else:
                metadata["avg_topup_amount"] = "NA"
        else:
            metadata["airtime_topup_frequency"] = 0.0
            metadata["avg_topup_amount"] = "NA"
    else:
        metadata["airtime_topup_frequency"] = "NA"
        metadata["avg_topup_amount"] = "NA"
    
    # Data usage variance (if available)
    if "data_usage_30d" in raw_data and isinstance(raw_data["data_usage_30d"], list):
        usage = [u.get("mb", 0) for u in raw_data["data_usage_30d"] if isinstance(u, dict)]
        if len(usage) > 1:
            metadata["data_usage_variance"] = round(statistics.variance(usage), 2)
        else:
            metadata["data_usage_variance"] = "NA"
    else:
        metadata["data_usage_variance"] = "NA"
    
    return metadata


def calculate_psychometrics(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate psychometric scores from raw responses.
    
    Args:
        raw_data: Dictionary containing psychometric responses
        
    Returns:
        Dictionary of psychometric features
    """
    psych = {}
    
    # If individual question responses provided
    if "psychometric_responses" in raw_data and isinstance(raw_data["psychometric_responses"], dict):
        responses = raw_data["psychometric_responses"]
        
        # Store individual responses
        for key, value in responses.items():
            if key.startswith("C_q") or key.startswith("E_q") or key.startswith("O_q") or \
               key.startswith("A_q") or key.startswith("N_q"):
                psych[key] = int(value) if isinstance(value, (int, float)) else "NA"
        
        # Calculate aggregated scores if we have conscientiousness questions
        c_questions = [v for k, v in responses.items() if k.startswith("C_q") and isinstance(v, (int, float))]
        if c_questions:
            psych["conscientiousness_score"] = round(statistics.mean(c_questions), 2)
        else:
            psych["conscientiousness_score"] = "NA"
    else:
        # If pre-calculated score provided
        if "conscientiousness_score" in raw_data:
            psych["conscientiousness_score"] = float(raw_data["conscientiousness_score"])
        else:
            psych["conscientiousness_score"] = "NA"
    
    return psych


def calculate_financial_behavior(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate financial behavior features from raw transaction data.
    
    Args:
        raw_data: Dictionary containing financial transaction data
        
    Returns:
        Dictionary of financial behavior features
    """
    fin = {}
    
    # Savings frequency and variance
    if "savings_history_90d" in raw_data and isinstance(raw_data["savings_history_90d"], list):
        savings = raw_data["savings_history_90d"]
        if savings:
            fin["savings_frequency"] = round(len(savings) / 90.0, 2)
            amounts = [s.get("amount", 0) for s in savings if isinstance(s, dict)]
            if len(amounts) > 1:
                fin["savings_amount_variance"] = round(statistics.variance(amounts), 2)
            else:
                fin["savings_amount_variance"] = "NA"
        else:
            fin["savings_frequency"] = 0.0
            fin["savings_amount_variance"] = "NA"
    else:
        fin["savings_frequency"] = "NA"
        fin["savings_amount_variance"] = "NA"
    
    # Wallet balance lows
    if "wallet_balance_timeseries_90d" in raw_data and isinstance(raw_data["wallet_balance_timeseries_90d"], list):
        balances = raw_data["wallet_balance_timeseries_90d"]
        if balances:
            # Count days with balance below threshold (e.g., 10% of average)
            balance_values = [b.get("balance", 0) for b in balances if isinstance(b, dict)]
            if balance_values:
                avg_balance = statistics.mean(balance_values)
                threshold = avg_balance * 0.1
                lows = sum(1 for b in balance_values if b < threshold)
                fin["wallet_balance_lows_last_90d"] = lows
            else:
                fin["wallet_balance_lows_last_90d"] = "NA"
        else:
            fin["wallet_balance_lows_last_90d"] = "NA"
    else:
        fin["wallet_balance_lows_last_90d"] = "NA"
    
    # Bill payment timeliness
    if "transaction_history_180d" in raw_data and isinstance(raw_data["transaction_history_180d"], list):
        transactions = raw_data["transaction_history_180d"]
        bill_payments = [t for t in transactions if isinstance(t, dict) and 
                        t.get("type") in ["bill", "utility", "payment"]]
        if bill_payments:
            on_time = sum(1 for t in bill_payments if t.get("on_time", True))
            fin["bill_payment_timeliness"] = round(on_time / len(bill_payments), 2)
        else:
            fin["bill_payment_timeliness"] = "NA"
    else:
        fin["bill_payment_timeliness"] = "NA"
    
    return fin


def calculate_social_network(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate social network features from raw data.
    
    Args:
        raw_data: Dictionary containing social network data
        
    Returns:
        Dictionary of social network features
    """
    social = {}
    
    # SHG membership
    if "shg_membership_info" in raw_data:
        shg_info = raw_data["shg_membership_info"]
        if isinstance(shg_info, dict):
            social["shg_membership"] = shg_info.get("is_member", False)
            social["peer_monitoring_strength"] = shg_info.get("peer_monitoring_score", "NA")
        elif isinstance(shg_info, bool):
            social["shg_membership"] = shg_info
            social["peer_monitoring_strength"] = "NA"
        else:
            social["shg_membership"] = "NA"
            social["peer_monitoring_strength"] = "NA"
    else:
        social["shg_membership"] = "NA"
        social["peer_monitoring_strength"] = "NA"
    
    return social


def calculate_loan_history(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate loan history features from raw data.
    
    Args:
        raw_data: Dictionary containing loan history data
        
    Returns:
        Dictionary of loan history features
    """
    loan = {}
    
    if "loan_history" in raw_data and isinstance(raw_data["loan_history"], list):
        loans = raw_data["loan_history"]
        loan["previous_loans"] = len(loans)
        
        defaults = sum(1 for l in loans if isinstance(l, dict) and l.get("defaulted", False))
        loan["previous_defaults"] = defaults
        
        late_payments = sum(1 for l in loans if isinstance(l, dict) and l.get("late_payment", False))
        loan["previous_late_payments"] = late_payments
        
        # Average repayment delay
        delays = [l.get("delay_days", 0) for l in loans if isinstance(l, dict) and l.get("delay_days") is not None]
        if delays:
            loan["avg_repayment_delay_days"] = round(statistics.mean(delays), 2)
        else:
            loan["avg_repayment_delay_days"] = "NA"
    else:
        loan["previous_loans"] = "NA"
        loan["previous_defaults"] = "NA"
        loan["previous_late_payments"] = "NA"
        loan["avg_repayment_delay_days"] = "NA"
    
    return loan


def build_profile(user_id: str,
                 demographics: Dict[str, Any],
                 raw_unconventional_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete user profile from demographics and raw unconventional data.
    
    Args:
        user_id: Unique user identifier
        demographics: Dictionary with age, gender, occupation, monthly_income
        raw_unconventional_data: Dictionary containing raw mobile, psychometric, financial, etc. data
        
    Returns:
        Complete profile dictionary matching the required schema
    """
    profile = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "demographics": demographics.copy(),
        "mobile_metadata": calculate_mobile_metadata(raw_unconventional_data),
        "psychometrics": calculate_psychometrics(raw_unconventional_data),
        "financial_behavior": calculate_financial_behavior(raw_unconventional_data),
        "social_network": calculate_social_network(raw_unconventional_data),
        "loan_history": calculate_loan_history(raw_unconventional_data),
        "_raw_unconventional": raw_unconventional_data.copy()  # Store raw for audit
    }
    
    logger.info(f"Built profile for user {user_id}")
    return profile


def build_profile_from_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build profile from a JSON structure that may already be partially structured.
    
    Args:
        json_data: Dictionary that may contain demographics and raw data
        
    Returns:
        Complete profile dictionary
    """
    user_id = json_data.get("user_id", f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    demographics = json_data.get("demographics", {})
    raw_data = json_data.get("_raw_unconventional", json_data.get("raw_data", {}))
    
    return build_profile(user_id, demographics, raw_data)


