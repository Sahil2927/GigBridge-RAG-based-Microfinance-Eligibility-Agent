"""
Unit tests for memory and storage.
"""

import pytest
import os
import json
from pathlib import Path
from src.memory.store import (
    save_user_profile, get_user_profile, delete_user_data,
    save_consented_submission, get_bank_submissions
)


def test_save_and_get_profile():
    """Test saving and retrieving profile."""
    test_profile = {
        "user_id": "test_user_memory_001",
        "timestamp": "2024-01-01T00:00:00",
        "demographics": {"age": 30, "gender": "female"}
    }
    
    save_user_profile("test_user_memory_001", test_profile)
    retrieved = get_user_profile("test_user_memory_001")
    
    assert retrieved is not None
    assert retrieved["user_id"] == "test_user_memory_001"
    
    # Cleanup
    delete_user_data("test_user_memory_001")


def test_delete_user_data():
    """Test user data deletion."""
    test_profile = {
        "user_id": "test_user_delete_001",
        "timestamp": "2024-01-01T00:00:00",
        "demographics": {"age": 30}
    }
    
    save_user_profile("test_user_delete_001", test_profile)
    delete_user_data("test_user_delete_001")
    
    retrieved = get_user_profile("test_user_delete_001")
    assert retrieved is None


def test_save_consented_submission():
    """Test saving consented submission."""
    test_profile = {
        "user_id": "test_consent_001",
        "timestamp": "2024-01-01T00:00:00",
        "demographics": {"age": 30}
    }
    
    test_decision = {
        "eligibility": "yes",
        "risk_score": 0.3,
        "verdict_text": "Test verdict"
    }
    
    save_consented_submission("test_consent_001", test_profile, test_decision)
    
    # Check that file was created
    submissions_dir = Path("data/consented_submissions")
    assert submissions_dir.exists()
    
    # Check bank database
    submissions = get_bank_submissions(limit=10)
    assert len(submissions) > 0
    
    # Find our submission
    found = False
    for sub in submissions:
        if sub["user_id"] == "test_consent_001":
            found = True
            assert sub["decision"]["eligibility"] == "yes"
            break
    
    assert found


