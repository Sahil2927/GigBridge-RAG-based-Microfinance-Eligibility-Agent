"""
Unit tests for profile builder.
"""

import pytest
from datetime import datetime
from src.profile.profile_builder import (
    build_profile, calculate_mobile_metadata,
    calculate_psychometrics, calculate_financial_behavior
)


def test_build_profile():
    """Test profile building."""
    demographics = {
        "age": 30,
        "gender": "female",
        "occupation": "gig_worker",
        "monthly_income": 20000
    }
    
    raw_data = {
        "call_logs_30d": [
            {"date": "2024-01-01T10:00:00", "contact": "contact_001", "duration": 120}
        ],
        "psychometric_responses": {
            "C_q1": 4,
            "C_q2": 5
        }
    }
    
    profile = build_profile("test_user_001", demographics, raw_data)
    
    assert profile["user_id"] == "test_user_001"
    assert profile["demographics"] == demographics
    assert "mobile_metadata" in profile
    assert "psychometrics" in profile
    assert "timestamp" in profile
    assert isinstance(profile["timestamp"], str)


def test_calculate_mobile_metadata():
    """Test mobile metadata calculation."""
    raw_data = {
        "call_logs_30d": [
            {"date": "2024-01-01T10:00:00", "contact": "contact_001", "duration": 120},
            {"date": "2024-01-02T10:00:00", "contact": "contact_002", "duration": 90}
        ],
        "airtime_topups_30d": [
            {"date": "2024-01-01", "amount": 100},
            {"date": "2024-01-15", "amount": 150}
        ]
    }
    
    metadata = calculate_mobile_metadata(raw_data)
    
    assert "avg_daily_calls" in metadata
    assert "unique_contacts_30d" in metadata
    assert "airtime_topup_frequency" in metadata
    assert metadata["unique_contacts_30d"] == 2


def test_calculate_psychometrics():
    """Test psychometric calculation."""
    raw_data = {
        "psychometric_responses": {
            "C_q1": 4,
            "C_q2": 5,
            "C_q3": 4,
            "C_q4": 5,
            "C_q5": 4
        }
    }
    
    psych = calculate_psychometrics(raw_data)
    
    assert "conscientiousness_score" in psych
    assert isinstance(psych["conscientiousness_score"], (float, int))
    assert 1.0 <= psych["conscientiousness_score"] <= 5.0


def test_calculate_financial_behavior():
    """Test financial behavior calculation."""
    raw_data = {
        "savings_history_90d": [
            {"date": "2024-01-01", "amount": 1000},
            {"date": "2024-01-15", "amount": 1500}
        ],
        "transaction_history_180d": [
            {"date": "2024-01-01", "type": "bill", "amount": 500, "on_time": True},
            {"date": "2024-01-15", "type": "bill", "amount": 600, "on_time": True}
        ]
    }
    
    fin = calculate_financial_behavior(raw_data)
    
    assert "savings_frequency" in fin
    assert "bill_payment_timeliness" in fin
    assert isinstance(fin["bill_payment_timeliness"], (float, int))


