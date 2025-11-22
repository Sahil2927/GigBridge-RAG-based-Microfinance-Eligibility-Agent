#!/usr/bin/env python3
"""
Example usage of the RAG-based Microfinance Eligibility Agent

This script demonstrates how to use the system programmatically.
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.profile.profile_builder import build_profile
from src.agent.rag_agent import RAGAgent
from src.memory.store import save_user_profile, save_consented_submission, delete_user_data


def example_assessment():
    """Example: Run a complete assessment."""
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  Please set GROQ_API_KEY environment variable")
        print("   export GROQ_API_KEY='your-key-here'")
        return
    
    # Example demographics
    demographics = {
        "age": 32,
        "gender": "female",
        "occupation": "gig_worker",
        "monthly_income": 25000
    }
    
    # Example raw unconventional data
    raw_data = {
        "call_logs_30d": [
            {"date": "2024-01-01T10:00:00", "contact": "contact_001", "duration": 120},
            {"date": "2024-01-02T14:30:00", "contact": "contact_002", "duration": 90},
            # ... more calls
        ],
        "airtime_topups_30d": [
            {"date": "2024-01-01", "amount": 100},
            {"date": "2024-01-05", "amount": 150},
        ],
        "psychometric_responses": {
            "C_q1": 4,
            "C_q2": 5,
            "C_q3": 4,
            "C_q4": 5,
            "C_q5": 4
        },
        "savings_history_90d": [
            {"date": "2024-01-01", "amount": 2000},
            {"date": "2024-01-08", "amount": 2500},
        ],
        "shg_membership_info": {
            "is_member": True,
            "peer_monitoring_score": 0.85
        },
        "loan_history": [
            {"loan_id": "loan_001", "amount": 10000, "defaulted": False, "late_payment": False, "delay_days": 0}
        ]
    }
    
    print("Building profile...")
    profile = build_profile(
        user_id="example_user_001",
        demographics=demographics,
        raw_unconventional_data=raw_data
    )
    
    print(f"Profile built for user: {profile['user_id']}")
    
    # Save to memory
    save_user_profile(profile["user_id"], profile)
    print("Profile saved to memory")
    
    # Run RAG assessment
    print("\nRunning RAG assessment...")
    agent = RAGAgent(groq_api_key=api_key, model_name="mixtral-8x7b-32768")
    decision = agent.assess_eligibility(profile, k=6)
    
    # Display results
    print("\n" + "="*60)
    print("ASSESSMENT RESULTS")
    print("="*60)
    print(f"Eligibility: {decision['eligibility'].upper()}")
    print(f"Risk Score: {decision['risk_score']:.2f}")
    print(f"Confidence: {decision['confidence'].upper()}")
    print(f"\nVerdict: {decision['verdict_text']}")
    print("\nStrong Points:")
    for i, point in enumerate(decision['strong_points'], 1):
        print(f"  {i}. {point}")
    print("\nWeak Points:")
    for i, point in enumerate(decision['weak_points'], 1):
        print(f"  {i}. {point}")
    print("\nRecommendations:")
    for i, rec in enumerate(decision['actionable_recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Simulate consent
    print("\n" + "="*60)
    user_consents = True  # In real app, this comes from UI
    
    if user_consents:
        save_consented_submission(profile["user_id"], profile, decision)
        print("✅ Consent given. Data saved to bank database.")
    else:
        delete_user_data(profile["user_id"])
        print("❌ Consent denied. All data deleted.")


def example_load_from_file():
    """Example: Load profile from JSON file."""
    
    example_file = Path("example_inputs/likely_eligible_gig_worker.json")
    
    if not example_file.exists():
        print(f"Example file not found: {example_file}")
        return
    
    print(f"Loading example profile from {example_file}...")
    
    with open(example_file, 'r') as f:
        data = json.load(f)
    
    from src.profile.profile_builder import build_profile_from_json
    profile = build_profile_from_json(data)
    
    print(f"Profile loaded: {profile['user_id']}")
    print(f"Demographics: {profile['demographics']}")
    print(f"Mobile metadata: {profile['mobile_metadata']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of RAG Microfinance Agent")
    parser.add_argument(
        "--mode",
        choices=["assessment", "load"],
        default="assessment",
        help="Example mode to run"
    )
    
    args = parser.parse_args()
    
    if args.mode == "assessment":
        example_assessment()
    elif args.mode == "load":
        example_load_from_file()

