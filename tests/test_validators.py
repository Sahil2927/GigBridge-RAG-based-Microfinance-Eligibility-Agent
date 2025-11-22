"""
Unit tests for validators.
"""

import pytest
from src.utils.validators import (
    validate_profile_schema, validate_rag_output,
    extract_json_from_text, sanitize_profile
)


def test_validate_profile_schema_valid():
    """Test profile schema validation with valid profile."""
    profile = {
        "user_id": "test_001",
        "timestamp": "2024-01-01T00:00:00",
        "demographics": {
            "age": 30,
            "gender": "female",
            "occupation": "gig_worker",
            "monthly_income": 20000
        },
        "mobile_metadata": {},
        "psychometrics": {},
        "financial_behavior": {},
        "social_network": {},
        "loan_history": {}
    }
    
    is_valid, errors = validate_profile_schema(profile)
    assert is_valid
    assert len(errors) == 0


def test_validate_profile_schema_invalid():
    """Test profile schema validation with invalid profile."""
    profile = {
        "user_id": "test_002"
        # Missing required fields
    }
    
    is_valid, errors = validate_profile_schema(profile)
    assert not is_valid
    assert len(errors) > 0


def test_validate_rag_output_valid():
    """Test RAG output validation with valid output."""
    rag_output = {
        "eligibility": "yes",
        "risk_score": 0.3,
        "verdict_text": "Test verdict",
        "strong_points": ["point 1", "point 2", "point 3"],
        "weak_points": ["point 1", "point 2", "point 3"],
        "required_unconventional_data": [],
        "actionable_recommendations": ["rec 1", "rec 2", "rec 3", "rec 4"],
        "confidence": "high"
    }
    
    is_valid, errors = validate_rag_output(rag_output)
    assert is_valid
    assert len(errors) == 0


def test_validate_rag_output_invalid():
    """Test RAG output validation with invalid output."""
    rag_output = {
        "eligibility": "maybe"
        # Missing required fields
    }
    
    is_valid, errors = validate_rag_output(rag_output)
    assert not is_valid
    assert len(errors) > 0


def test_extract_json_from_text():
    """Test JSON extraction from text."""
    text = "Here is some text before. {\"key\": \"value\", \"number\": 42} And some text after."
    result = extract_json_from_text(text)
    
    assert result is not None
    assert result["key"] == "value"
    assert result["number"] == 42


def test_extract_json_from_text_no_json():
    """Test JSON extraction when no JSON present."""
    text = "This is just plain text with no JSON."
    result = extract_json_from_text(text)
    
    assert result is None


def test_sanitize_profile():
    """Test profile sanitization."""
    profile = {
        "user_id": "test_003",
        "demographics": {"age": 30}
    }
    
    sanitized = sanitize_profile(profile)
    
    assert "timestamp" in sanitized
    assert "mobile_metadata" in sanitized
    assert "psychometrics" in sanitized


