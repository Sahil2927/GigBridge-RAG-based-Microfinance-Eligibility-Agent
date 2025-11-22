"""
Validation and Schema Utilities

This module provides functions to validate user profiles and RAG outputs
against the required JSON schemas.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Required unconventional data types
REQUIRED_DATA_TYPES = [
    "mobile_call_logs_30d",
    "airtime_topups_30d",
    "psychometric_responses",
    "savings_history_90d",
    "wallet_balance_timeseries_90d",
    "shg_membership_info",
    "transaction_history_180d",
    "sms_patterns_30d"
]


def validate_profile_schema(profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a user profile against the required schema.
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required top-level fields
    required_fields = ["user_id", "timestamp", "demographics"]
    for field in required_fields:
        if field not in profile:
            errors.append(f"Missing required field: {field}")
    
    # Validate demographics
    if "demographics" in profile:
        demo = profile["demographics"]
        required_demo = ["age", "gender", "occupation", "monthly_income"]
        for field in required_demo:
            if field not in demo:
                errors.append(f"Missing demographics field: {field}")
    
    # Validate timestamp format (ISO8601)
    if "timestamp" in profile:
        try:
            datetime.fromisoformat(profile["timestamp"].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            errors.append("Invalid timestamp format (should be ISO8601)")
    
    # Optional fields should exist (even if "NA")
    optional_sections = [
        "mobile_metadata", "psychometrics", "financial_behavior",
        "social_network", "loan_history", "_raw_unconventional"
    ]
    for section in optional_sections:
        if section not in profile:
            profile[section] = {}
    
    return len(errors) == 0, errors


def sanitize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and normalize a user profile.
    Ensures all fields exist with "NA" defaults where appropriate.
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Sanitized profile dictionary
    """
    sanitized = profile.copy()
    
    # Ensure timestamp exists
    if "timestamp" not in sanitized:
        sanitized["timestamp"] = datetime.now().isoformat()
    
    # Ensure user_id exists
    if "user_id" not in sanitized:
        sanitized["user_id"] = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Ensure demographics structure
    if "demographics" not in sanitized:
        sanitized["demographics"] = {}
    
    # Ensure optional sections exist
    optional_sections = {
        "mobile_metadata": {},
        "psychometrics": {},
        "financial_behavior": {},
        "social_network": {},
        "loan_history": {},
        "_raw_unconventional": {}
    }
    
    for section, default in optional_sections.items():
        if section not in sanitized:
            sanitized[section] = default
    
    return sanitized


def validate_rag_output(rag_output: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate RAG agent output against the required schema.
    
    Args:
        rag_output: RAG output dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    required_fields = [
        "eligibility", "risk_score", "verdict_text", "strong_points",
        "weak_points", "required_unconventional_data", "actionable_recommendations",
        "confidence"
    ]
    
    for field in required_fields:
        if field not in rag_output:
            errors.append(f"Missing required RAG output field: {field}")
    
    # Validate eligibility value
    if "eligibility" in rag_output:
        valid_eligibility = ["yes", "no", "maybe"]
        if rag_output["eligibility"].lower() not in valid_eligibility:
            errors.append(f"Invalid eligibility value: {rag_output['eligibility']}")
    
    # Validate risk_score range
    if "risk_score" in rag_output:
        try:
            score = float(rag_output["risk_score"])
            if not (0.0 <= score <= 1.0):
                errors.append(f"risk_score must be between 0.0 and 1.0, got {score}")
        except (ValueError, TypeError):
            errors.append("risk_score must be a number")
    
    # Validate confidence
    if "confidence" in rag_output:
        valid_confidence = ["high", "medium", "low"]
        if rag_output["confidence"].lower() not in valid_confidence:
            errors.append(f"Invalid confidence value: {rag_output['confidence']}")
    
    # Validate required_unconventional_data items
    if "required_unconventional_data" in rag_output:
        required_data = rag_output["required_unconventional_data"]
        if isinstance(required_data, list):
            for item in required_data:
                if item not in REQUIRED_DATA_TYPES:
                    errors.append(f"Unknown required data type: {item}")
    
    return len(errors) == 0, errors


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain extra prose.
    Handles cases where LLM output includes explanations before/after JSON.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dictionary or None if extraction fails
    """
    # Try to find JSON object in text
    # Look for { ... } pattern
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        logger.warning("No JSON object found in text")
        return None
    
    json_str = text[start_idx:end_idx + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        # Try to fix common issues
        # Remove trailing commas
        json_str = json_str.replace(',\n}', '\n}').replace(',\n]', '\n]')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


def sanitize_string(value: Any) -> str:
    """
    Sanitize a value to a string, handling None and special cases.
    
    Args:
        value: Value to sanitize
        
    Returns:
        Sanitized string
    """
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()

