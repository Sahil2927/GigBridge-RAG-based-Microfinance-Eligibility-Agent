"""
Streamlit UI for RAG-based Microfinance Eligibility Agent

This app provides:
- User input form for demographics and unconventional data
- RAG-based eligibility assessment
- Consent mechanism
- Admin view of bank database
"""

import streamlit as st
import streamlit.errors as st_errors
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd

logger = logging.getLogger(__name__)

# Fix TensorFlow DLL issue on Windows - disable TensorFlow for sentence-transformers
# We only need PyTorch backend, not TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TRANSFORMERS_NO_TF'] = '1'  # Skip TensorFlow in transformers

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(str(Path.cwd()))

from src.profile.profile_builder import build_profile, build_profile_from_json
from src.agent.rag_agent import RAGAgent
from src.ml.prediction_service import PredictionService
from src.memory.store import (
    save_user_profile, get_user_profile, delete_user_data,
    save_consented_submission, get_bank_submissions, log_action
)
from src.utils.validators import validate_profile_schema, sanitize_profile



# Page config
st.set_page_config(
    page_title="Microfinance Eligibility Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "decision" not in st.session_state:
    st.session_state.decision = None
if "ml_prediction" not in st.session_state:
    st.session_state.ml_prediction = None
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False
if "prediction_service" not in st.session_state:
    st.session_state.prediction_service = None

# ---------- Helper functions for alignment and mapping ----------

def _find_raw_key(raw: Dict[str, Any], col_name: str) -> Optional[str]:
    """Case-insensitive search for a raw_inputs key that matches col_name."""
    if col_name in raw:
        return col_name
    lower = col_name.lower()
    for k in raw.keys():
        if k.lower() == lower:
            return k
    # try common variants (underscores/spaces)
    for k in raw.keys():
        if k.replace("_", " ").lower() == lower.replace("_", " ").lower():
            return k
    return None

def _to_numeric_if_possible(val):
    """Return numeric val if convertible, else original"""
    try:
        if isinstance(val, bool):
            return int(val)
        return float(val)
    except Exception:
        return val

def build_aligned_inputs_from_expected(expected_features: list, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given predictor.expected feature names (possibly one-hot like 'cat__Education_Bachelor's' or 'num__Age'),
    and the user's raw_inputs (with canonical keys like 'Age', 'Income', 'LoanAmount', 'Education', etc.),
    produce a dict of aligned inputs keyed by exact expected_features with appropriate values:
      - for 'cat__<Col>_<Value>' -> 1 if raw_inputs[Col] == Value else 0
      - for 'num__<Col>' -> numeric value from raw_inputs[Col] (or 0 if missing)
      - otherwise if expected is simple column, try to map directly
    """
    aligned = {}
    for feat in expected_features:
        # categorical one-hot prefix
        if feat.startswith("cat__"):
            try:
                rest = feat.split("cat__", 1)[1]
                col, val = rest.split("_", 1)
            except Exception:
                parts = feat.split("__", 1)[-1].split("_")
                col = parts[0]
                val = "_".join(parts[1:]) if len(parts) > 1 else ""
            raw_key = _find_raw_key(raw_inputs, col)
            raw_val = raw_inputs.get(raw_key) if raw_key else None
            match = False
            if raw_val is not None:
                try:
                    if str(raw_val).strip().lower() == str(val).strip().lower():
                        match = True
                except Exception:
                    match = False
            aligned[feat] = 1 if match else 0

        elif feat.startswith("num__"):
            col = feat.split("num__", 1)[1]
            raw_key = _find_raw_key(raw_inputs, col)
            if raw_key:
                v = raw_inputs.get(raw_key)
                num = _to_numeric_if_possible(v)
                aligned[feat] = num if isinstance(num, (int, float)) else 0.0
            else:
                aligned[feat] = 0.0

        else:
            raw_key = _find_raw_key(raw_inputs, feat)
            if raw_key:
                v = raw_inputs.get(raw_key)
                num = _to_numeric_if_possible(v)
                aligned[feat] = num if isinstance(num, (int, float)) else v
            else:
                aligned[feat] = 0.0
    return aligned

def human_readable_expected_list(expected_features: list) -> list:
    """
    Collapse expected one-hot / num__ names to human-friendly canonical column names,
    preserving order and deduplicating.
    """
    human_readable = []
    seen = set()
    for feat in expected_features:
        name = feat
        if feat.startswith("cat__"):
            try:
                rest = feat.split("cat__", 1)[1]
                col = rest.split("_", 1)[0]
                name = col
            except Exception:
                name = feat
        elif feat.startswith("num__"):
            name = feat.split("num__", 1)[1]
        else:
            if "__" in feat:
                name = feat.split("__")[-1]
            elif "_" in feat:
                name = feat.split("_")[0]
            else:
                name = feat

        name = str(name)
        if name not in seen:
            human_readable.append(name)
            seen.add(name)

    canonical_order = [
        "LoanID", "Age", "Income", "LoanAmount", "RequestedLoanAmount",
        "CreditScore", "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm", "DTIRatio",
        "Education", "EmploymentType", "MaritalStatus",
        "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner", "Default"
    ]

    final_expected = []
    for col in canonical_order:
        if col in human_readable and col not in final_expected:
            final_expected.append(col)
    for col in human_readable:
        if col not in final_expected:
            final_expected.append(col)

    return final_expected

# -------------------- Streamlit UI --------------------

st.title("💰 RAG-based Microfinance Eligibility Agent")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    if "page" not in st.session_state:
        st.session_state.page = "Assessment"

    page = st.radio(
        "Select Page",
        ["Assessment", "Admin View", "Wellness Coach"],
        index=0 if st.session_state.page == "Assessment" else (1 if st.session_state.page == "Admin View" else 2),
        label_visibility="collapsed"
    )
    st.session_state.page = page

    st.markdown("---")
    st.header("Privacy & Ethics")
    st.info("""
    **Privacy Statement:**
    - Your data is stored locally until you provide explicit consent
    - If you decline consent, all your data is immediately deleted
    - Only aggregated features are shared with the bank (not raw logs)
    - You can request data deletion at any time
    """)

    st.markdown("---")
    st.header("Setup")
    # if not check_groq_api_key():
    #     st.error("⚠️ GROQ_API_KEY not set!")
    #     st.code("export GROQ_API_KEY='your-key-here'")
    #     st.info("Get your API key from: https://console.groq.com/")
    # else:
    #     st.success("✅ Groq API key configured")


def main():
    if st.session_state.page == "Assessment":
        assessment_page()
    elif st.session_state.page == "Admin View":
        admin_page()
    elif st.session_state.page == "Wellness Coach":
        wellness_coach_page()


# def assessment_page():
#     """Assessment page with user input form."""
#     st.header("Eligibility Assessment")
    
#     # Initialize prediction service
#     if st.session_state.prediction_service is None:
#         try:
#             st.session_state.prediction_service = PredictionService()
#             if not st.session_state.prediction_service.predictor.load_model():
#                 st.warning("⚠️ ML Model not found. Please train the model first.")
#                 with st.expander("How to train the model"):
#                     st.code("""
#                             # Option 1: If you have the CSV file
#                             python scripts/export_model_from_notebook.py --csv-path /path/to/Loan_default.csv

#                             # Option 2: Place Loan_default.csv in data/ folder
#                             python scripts/export_model_from_notebook.py
#                     """)
#         except Exception as e:
#             st.error(f"Error initializing prediction service: {e}")
    
#     # Assessment mode selection
#     assessment_mode = st.radio(
#         "Assessment Mode",
#         ["ML Model (Recommended)", "RAG Agent", "Both"],
#         horizontal=True
#     )
    
#     use_ml = assessment_mode in ["ML Model (Recommended)", "Both"]
#     use_rag = assessment_mode in ["RAG Agent", "Both"]
    
#     # Check RAG requirements if needed
#     if use_rag:
#         # if not check_groq_api_key():
#         #     st.error("Please set GROQ_API_KEY environment variable to use RAG assessment.")
#         #     if not use_ml:
#         #         st.stop()
        
#         index_path = Path("./data/faiss_index.bin")
#         metadata_path = Path("./data/metadata.jsonl")
        
#         if not index_path.exists() or not metadata_path.exists():
#             st.warning("⚠️ Knowledge Base Not Found (RAG will be unavailable)")
#             if not use_ml:
#                 st.stop()
    
#     # User ID input
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         user_id_input = st.text_input(
#             "User ID",
#             value=st.session_state.user_id or "",
#             help="Enter a unique user identifier"
#         )
#     with col2:
#         if st.button("Load Saved Profile"):
#             if user_id_input:
#                 profile = get_user_profile(user_id_input)
#                 if profile:
#                     st.session_state.user_id = user_id_input
#                     st.session_state.profile = profile
#                     st.success(f"Loaded profile for {user_id_input}")
#                 else:
#                     st.warning(f"No saved profile found for {user_id_input}")
    
#     if user_id_input:
#         st.session_state.user_id = user_id_input

#     # ----- MANUAL SINGLE RECORD UI (OUTSIDE the eligibility form) -----
#     st.markdown("---")
#     st.subheader("Optional: Fill a single structured record (quick load)")

#     with st.expander("Manual structured record (fill and click Use this structured record)"):
#         # Use types consistent for each widget
#         loanid_manual = st.text_input("LoanID", value=st.session_state.get("loanid", "I38PQUQS"), key="manual_loanid")
#         age_manual = st.number_input("Age (structured)", min_value=18, max_value=100, value=int(st.session_state.get("age", 35)), key="manual_age")
#         income_manual = st.number_input("Income (structured)", min_value=0.0, value=float(st.session_state.get("income", 50000.0)), key="manual_income")
#         loan_amount_manual = st.number_input("LoanAmount (existing/current outstanding)", min_value=0.0, value=float(st.session_state.get("loan_amount", 20000.0)), key="manual_loan_amount")
#         requested_loan_amount_manual = st.number_input("RequestedLoanAmount (amount user requests)", min_value=0.0, value=float(st.session_state.get("requested_loan_amount", loan_amount_manual)), key="manual_requested_amount")
#         credit_score_manual = st.number_input("CreditScore", min_value=0, max_value=1000, value=int(st.session_state.get("credit_score", 650)), key="manual_credit_score")
#         months_employed_manual = st.number_input("MonthsEmployed", min_value=0, value=int(st.session_state.get("months_employed", 24)), key="manual_months_employed")
#         num_credit_lines_manual = st.number_input("NumCreditLines", min_value=0, max_value=50, value=int(st.session_state.get("num_credit_lines", 3)), key="manual_num_lines")
#         interest_rate_manual = st.number_input("InterestRate", min_value=0.0, value=float(st.session_state.get("interest_rate", 10.0)), format="%.2f", key="manual_interest")
#         loan_term_manual = st.number_input("LoanTerm", min_value=1, value=int(st.session_state.get("loan_term", 36)), key="manual_loan_term")
#         dti_ratio_manual = st.number_input("DTIRatio", min_value=0.0, max_value=10.0, value=float(st.session_state.get("dti_ratio", 0.30)), format="%.2f", key="manual_dti")
#         education_manual = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], index=0 if st.session_state.get("education") is None else ["High School", "Bachelor's", "Master's", "PhD"].index(st.session_state.get("education")), key="manual_edu")
#         employment_type_manual = st.selectbox("EmploymentType", ["Full-time", "Part-time", "Unemployed", "Self-employed"], index=0 if st.session_state.get("employment_type") is None else ["Full-time", "Part-time", "Unemployed", "Self-employed"].index(st.session_state.get("employment_type")), key="manual_emp")
#         marital_status_manual = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"], index=0 if st.session_state.get("marital_status") is None else ["Single", "Married", "Divorced"].index(st.session_state.get("marital_status")), key="manual_marital")
#         has_mortgage_manual = st.selectbox("HasMortgage", ["Yes", "No"], index=0 if st.session_state.get("has_mortgage") is None else ["Yes", "No"].index(st.session_state.get("has_mortgage")), key="manual_mortgage")
#         has_dependents_manual = st.selectbox("HasDependents", ["Yes", "No"], index=0 if st.session_state.get("has_dependents") is None else ["Yes", "No"].index(st.session_state.get("has_dependents")), key="manual_dependents")
#         loan_purpose_manual = st.selectbox("LoanPurpose", ["Auto", "Business", "Education", "Home", "Other"], index=0 if st.session_state.get("loan_purpose") is None else ["Auto", "Business", "Education", "Home", "Other"].index(st.session_state.get("loan_purpose")), key="manual_purpose")
#         has_cosigner_manual = st.selectbox("HasCoSigner", ["Yes", "No"], index=0 if st.session_state.get("has_cosigner") is None else ["Yes", "No"].index(st.session_state.get("has_cosigner")), key="manual_cosigner")

#         if st.button("Use this structured record (load into form)"):
#             # Copy into session_state so form can pick them as defaults
#             st.session_state["loanid"] = loanid_manual
#             st.session_state["age"] = age_manual
#             st.session_state["income"] = income_manual
#             st.session_state["loan_amount"] = loan_amount_manual
#             st.session_state["requested_loan_amount"] = requested_loan_amount_manual
#             st.session_state["credit_score"] = credit_score_manual
#             st.session_state["months_employed"] = months_employed_manual
#             st.session_state["num_credit_lines"] = num_credit_lines_manual
#             st.session_state["interest_rate"] = interest_rate_manual
#             st.session_state["loan_term"] = loan_term_manual
#             st.session_state["dti_ratio"] = dti_ratio_manual
#             st.session_state["education"] = education_manual
#             st.session_state["employment_type"] = employment_type_manual
#             st.session_state["marital_status"] = marital_status_manual
#             st.session_state["has_mortgage"] = has_mortgage_manual
#             st.session_state["has_dependents"] = has_dependents_manual
#             st.session_state["loan_purpose"] = loan_purpose_manual
#             st.session_state["has_cosigner"] = has_cosigner_manual

#             st.success("Structured record loaded. Now open the main form below and click Run Assessment.")

#     # Form sections (single form, no nested forms)
#     with st.form("eligibility_form"):
#         st.subheader("1. Demographics")
#         col1, col2 = st.columns(2)

#         # left column (age, gender) - use session_state defaults if present
#         with col1:
#             age = st.number_input(
#                 "Age",
#                 min_value=18,
#                 max_value=100,
#                 value=int(st.session_state.get("age", 30))
#             )

#             gender_options = ["male", "female", "other", "prefer_not_to_say"]
#             gender_default = st.session_state.get("gender", None)
#             gender_index = 0
#             if gender_default in gender_options:
#                 gender_index = gender_options.index(gender_default)
#             gender = st.selectbox(
#                 "Gender",
#                 gender_options,
#                 index=gender_index
#             )

#         # right column (occupation, monthly income) - also prefill from session_state
#         with col2:
#             occupation_options = ["gig_worker", "small_business_owner", "daily_wage_worker", "farmer", "other"]
#             occ_default = st.session_state.get("occupation", None)
#             occ_index = 0
#             if occ_default in occupation_options:
#                 occ_index = occupation_options.index(occ_default)
#             occupation = st.selectbox(
#                 "Occupation",
#                 occupation_options,
#                 index=occ_index
#             )

#             monthly_income = st.number_input(
#                 "Monthly Income (₹)",
#                 min_value=0.0,
#                 value=float(st.session_state.get("monthly_income", 20000.0))
#             )

#         st.markdown("---")
#         st.subheader("2. Unconventional Data")
#         st.info(
#             """
#             **Data Privacy Notice:** 
#             Raw data (call logs, transaction history, etc.) will be stored temporarily for processing.
#             Only aggregated features will be shared with the bank after your consent.
#             You can decline consent and all data will be deleted.
#             """
#         )

#         # Mobile metadata
#         with st.expander("Mobile Metadata", expanded=False):
#             st.write("Upload JSON file or enter data manually")
            
#             mobile_file = st.file_uploader("Upload mobile data JSON", type=["json"], key="mobile")
#             if mobile_file:
#                 mobile_data = json.load(mobile_file)
#             else:
#                 mobile_data = {}
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     mobile_data["avg_daily_calls"] = st.number_input("Avg Daily Calls", min_value=0.0, value=float(mobile_data.get("avg_daily_calls", 0.0)), key="avg_calls")
#                     mobile_data["unique_contacts_30d"] = st.number_input("Unique Contacts (30d)", min_value=0.0, value=float(mobile_data.get("unique_contacts_30d", 0.0)), key="contacts")
#                 with col2:
#                     mobile_data["airtime_topup_frequency"] = st.number_input("Airtime Topup Frequency", min_value=0.0, value=float(mobile_data.get("airtime_topup_frequency", 0.0)), key="topup_freq")
#                     mobile_data["avg_topup_amount"] = st.number_input("Avg Topup Amount", min_value=0.0, value=float(mobile_data.get("avg_topup_amount", 0.0)), key="topup_amt")
        
#         # Psychometrics
#         with st.expander("Psychometric Data", expanded=False):
#             psych_file = st.file_uploader("Upload psychometric responses JSON", type=["json"], key="psych")
#             if psych_file:
#                 psych_data = json.load(psych_file)
#             else:
#                 psych_data = {}
#                 st.write("Enter psychometric responses (1-5 scale)")
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     psych_data["C_q1"] = st.slider("C_q1", 1, 5, int(psych_data.get("C_q1", 3)), key="c1")
#                     psych_data["C_q2"] = st.slider("C_q2", 1, 5, int(psych_data.get("C_q2", 3)), key="c2")
#                 with col2:
#                     psych_data["C_q3"] = st.slider("C_q3", 1, 5, int(psych_data.get("C_q3", 3)), key="c3")
#                     psych_data["C_q4"] = st.slider("C_q4", 1, 5, int(psych_data.get("C_q4", 3)), key="c4")
#                 with col3:
#                     psych_data["C_q5"] = st.slider("C_q5", 1, 5, int(psych_data.get("C_q5", 3)), key="c5")
        
#         # Financial behavior
#         with st.expander("Financial Behavior", expanded=False):
#             fin_file = st.file_uploader("Upload financial data JSON", type=["json"], key="fin")
#             if fin_file:
#                 fin_data = json.load(fin_file)
#             else:
#                 fin_data = {}
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     fin_data["savings_frequency"] = st.number_input("Savings Frequency", min_value=0.0, value=float(fin_data.get("savings_frequency", 0.0)), key="sav_freq")
#                     fin_data["bill_payment_timeliness"] = st.number_input("Bill Payment Timeliness (0-1)", min_value=0.0, max_value=1.0, value=float(fin_data.get("bill_payment_timeliness", 0.0)), key="bill_time")
#                 with col2:
#                     fin_data["wallet_balance_lows_last_90d"] = st.number_input("Wallet Balance Lows (90d)", min_value=0, value=int(fin_data.get("wallet_balance_lows_last_90d", 0)), key="wallet_lows")
        
#         # Social network
#         with st.expander("Social Network", expanded=False):
#             social_data = {}
#             social_data["shg_membership"] = st.checkbox("SHG Membership", key="shg")
#             if social_data["shg_membership"]:
#                 social_data["peer_monitoring_strength"] = st.number_input("Peer Monitoring Strength (0-1)", min_value=0.0, max_value=1.0, value=float(st.session_state.get("peer_monitoring_strength", 0.5)), key="peer_mon")
#             else:
#                 social_data["peer_monitoring_strength"] = "NA"
        
#         # Loan history
#         with st.expander("Loan History", expanded=False):
#             loan_file = st.file_uploader("Upload loan history JSON", type=["json"], key="loan")
#             if loan_file:
#                 loan_data = json.load(loan_file)
#             else:
#                 loan_data = {}
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     loan_data["previous_loans"] = st.number_input("Previous Loans", min_value=0, value=int(loan_data.get("previous_loans", 0)), key="prev_loans")
#                     loan_data["previous_defaults"] = st.number_input("Previous Defaults", min_value=0, value=int(loan_data.get("previous_defaults", 0)), key="defaults")
#                 with col2:
#                     loan_data["previous_late_payments"] = st.number_input("Previous Late Payments", min_value=0, value=int(loan_data.get("previous_late_payments", 0)), key="late_pay")
#                     loan_data["avg_repayment_delay_days"] = st.number_input("Avg Repayment Delay (days)", min_value=0.0, value=float(loan_data.get("avg_repayment_delay_days", 0.0)), key="delay")
        
#         # Full JSON upload option
#         st.markdown("---")
#         st.subheader("Or: Upload Complete Profile JSON")
#         full_json_file = st.file_uploader("Upload complete profile JSON", type=["json"], key="full_json")

#         submitted = st.form_submit_button("Run Assessment", type="primary", use_container_width=True)
        
#         if submitted:
#             with st.spinner("Processing assessment..."):
#                 try:
#                     # Build profile
#                     if full_json_file:
#                         raw_data = json.load(full_json_file)
#                         profile = build_profile_from_json(raw_data)
#                         if user_id_input:
#                             profile["user_id"] = user_id_input
#                     else:
#                         demographics = {
#                             "age": age,
#                             "gender": gender,
#                             "occupation": occupation,
#                             "monthly_income": monthly_income
#                         }
                        
#                         raw_unconventional = {
#                             "mobile_metadata": mobile_data,
#                             "psychometric_responses": psych_data,
#                             "financial_behavior": fin_data,
#                             "social_network": social_data,
#                             "loan_history": loan_data
#                         }
                        
#                         profile = build_profile(
#                             user_id=user_id_input or f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}",
#                             demographics=demographics,
#                             raw_unconventional_data=raw_unconventional
#                         )
                    
#                     # Validate profile
#                     is_valid, errors = validate_profile_schema(profile)
#                     if not is_valid:
#                         st.error(f"Profile validation errors: {errors}")
#                         st.stop()
                    
#                     # Sanitize profile
#                     profile = sanitize_profile(profile)
                    
#                     # Save to memory
#                     save_user_profile(profile["user_id"], profile)
#                     st.session_state.profile = profile
#                     st.session_state.user_id = profile["user_id"]
                    
#                     # Prepare raw inputs for ML model
#                     # Prefer session_state manual structured values if present
#                     loanid = st.session_state.get("loanid", profile.get("user_id"))
#                     loan_amount = st.session_state.get("loan_amount", 0.0)
#                     requested_loan_amount = st.session_state.get("requested_loan_amount", loan_amount)
#                     credit_score = st.session_state.get("credit_score", int(650))
#                     months_employed = st.session_state.get("months_employed", 0)
#                     num_credit_lines = st.session_state.get("num_credit_lines", 0)
#                     interest_rate = st.session_state.get("interest_rate", 0.0)
#                     loan_term = st.session_state.get("loan_term", 0)
#                     dti_ratio = st.session_state.get("dti_ratio", 0.0)
#                     education = st.session_state.get("education", profile.get("demographics", {}).get("education", "High School"))
#                     employment_type = st.session_state.get("employment_type", profile.get("demographics", {}).get("employment_type", "Full-time"))
#                     marital_status = st.session_state.get("marital_status", profile.get("demographics", {}).get("marital_status", "Single"))
#                     has_mortgage = st.session_state.get("has_mortgage", "No")
#                     has_dependents = st.session_state.get("has_dependents", "No")
#                     loan_purpose = st.session_state.get("loan_purpose", "Other")
#                     has_cosigner = st.session_state.get("has_cosigner", "No")

#                     # Build ml_raw_inputs separately (only structured features)
#                     ml_raw_inputs = {
#                         "LoanID": loanid,
#                         "Age": age,
#                         "Income": monthly_income,
#                         "LoanAmount": loan_amount,
#                         "RequestedLoanAmount": requested_loan_amount,
#                         "CreditScore": credit_score,
#                         "MonthsEmployed": months_employed,
#                         "NumCreditLines": num_credit_lines,
#                         "InterestRate": interest_rate,
#                         "LoanTerm": loan_term,
#                         "DTIRatio": dti_ratio,
#                         "Education": education,
#                         "EmploymentType": employment_type,
#                         "MaritalStatus": marital_status,
#                         "HasMortgage": has_mortgage,
#                         "HasDependents": has_dependents,
#                         "LoanPurpose": loan_purpose,
#                         "HasCoSigner": has_cosigner
#                     }

#                     # Keep unconventional full raw inputs for RAG profile only
#                     full_profile_raw = {
#                         "demographics": {"age": age, "gender": gender, "occupation": occupation, "monthly_income": monthly_income},
#                         "mobile_metadata": mobile_data,
#                         "psychometric_responses": psych_data,
#                         "financial_behavior": fin_data,
#                         "social_network": social_data,
#                         "loan_history": loan_data
#                     }

#                     # Now call only ml_raw_inputs for ML
#                     if use_ml and st.session_state.prediction_service and st.session_state.prediction_service.predictor.is_loaded:
#                         try:
#                             ml_prediction = st.session_state.prediction_service.predict_from_raw_inputs(
#                                 ml_raw_inputs,
#                                 consent=False,
#                                 user_id=profile["user_id"]
#                             )
#                             st.session_state.ml_prediction = ml_prediction
#                         except Exception as e:
#                             st.warning(f"ML prediction failed: {e}")
#                             # show what we passed (temporary)
#                             st.info(f"ML input keys passed: {list(ml_raw_inputs.keys())}")
#                             if not use_rag:
#                                 st.error("Assessment failed. Please check model configuration.")
#                                 st.stop()


#                     # ---- NEW: align to model's expected features to avoid ColumnTransformer errors ----
#                     # ---------- REPLACE the current ML-prediction block with this ----------
#                     # ml_prediction = None
#                     # if use_ml and st.session_state.prediction_service and st.session_state.prediction_service.predictor.is_loaded:
#                     #     try:
#                     #         # Use the predictor's frontend wrapper that knows how to build the correct model input
#                     #         ml_prediction = st.session_state.prediction_service.predict_from_raw_inputs(
#                     #             raw_inputs,
#                     #             consent=False,
#                     #             user_id=profile["user_id"]
#                     #         )
#                     #         st.session_state.ml_prediction = ml_prediction

#                     #     except RuntimeError as e:
#                     #         # Model-level runtime errors are surfaced as RuntimeError; inspect message
#                     #         msg = str(e)
#                     #         if "FeatureMismatchError" in msg or "ColumnTransformer" in msg:
#                     #             st.error("❌ ML prediction failed: input features do not match model's expected structured features.")
#                     #             st.info("Make sure you only pass the ML structured fields (LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner).")
#                     #             # Show expected columns (text-only) if predictor exposes them
#                     #             try:
#                     #                 expected_raw = st.session_state.prediction_service.predictor.expected_raw_columns()
#                     #                 if expected_raw:
#                     #                     with st.expander("Model expects these structured columns (example):"):
#                     #                         for i, c in enumerate(expected_raw, 1):
#                     #                             st.write(f"{i}. {c}")
#                     #             except Exception:
#                     #                 pass
#                     #             if not use_rag:
#                     #                 st.stop()
#                     #         else:
#                     #             # Re-raise fallback to see full exception during debugging
#                     #             raise
#                     #     except Exception as e:
#                     #         st.warning(f"ML prediction failed unexpectedly: {e}")
#                     #         if not use_rag:
#                     #             st.error("Assessment failed. Please check model configuration.")
#                     #             st.stop()
# # ---------- end replacement ----------


#                     # Run RAG assessment if enabled
#                     decision = None
#                     if use_rag:
#                         api_key = os.getenv("GROQ_API_KEY")
#                         if not api_key:
#                             try:
#                                 api_key = st.secrets.get("GROQ_API_KEY")
#                             except (st_errors.StreamlitSecretNotFoundError, FileNotFoundError, AttributeError, Exception):
#                                 api_key = None
                        
#                         if not api_key or api_key == "your-groq-api-key-here":
#                             if not use_ml:
#                                 st.error("⚠️ Invalid API Key Configuration")
#                                 st.warning("""
#                                 Your Groq API key is not configured correctly.
                                
#                                 **To fix:**
#                                 1. Open the `.env` file in your project root
#                                 2. Replace `your-groq-api-key-here` with your actual API key
#                                 3. Get your API key from: https://console.groq.com/
#                                 4. The key should look like: `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
#                                 5. Save the file and restart the app
#                                 """)
#                                 st.stop()
#                         else:
#                             model_name = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")
                            
#                             try:
#                                 agent = RAGAgent(groq_api_key=api_key, model_name=model_name)
#                                 decision = agent.assess_eligibility(profile, k=6)
#                                 st.session_state.decision = decision
#                             except ValueError as e:
#                                 if "Invalid Groq API Key" in str(e):
#                                     if not use_ml:
#                                         st.error("❌ Invalid API Key")
#                                         st.error(str(e))
#                                         st.info("💡 Tip: Run `python check_api_key.py` to verify your configuration")
#                                         st.stop()
#                                 else:
#                                     raise
                    
#                     st.session_state.consent_given = False
                    
#                     if decision:
#                         log_action("assessment_run", profile["user_id"], {"eligibility": decision["eligibility"]})
#                     if ml_prediction:
#                         log_action("ml_prediction_run", profile["user_id"], {"prediction": ml_prediction.get("prediction") if isinstance(ml_prediction, dict) else ml_prediction})
                    
#                     st.success("Assessment complete!")
#                     st.rerun()
                    
#                 except Exception as e:
#                     st.error(f"Error during assessment: {str(e)}")
#                     st.exception(e)
    
    
    
#     # Display results if available
#     if st.session_state.ml_prediction:
#         display_ml_results(st.session_state.ml_prediction)
    
#     if st.session_state.decision:
#         display_results(st.session_state.decision)
    
#     if st.session_state.ml_prediction or st.session_state.decision:
#         consent_section()
def assessment_page():
    """Assessment page with split inputs: Structured ML form and Unconventional (RAG) form."""
    st.header("Eligibility Assessment")

    # init prediction service once
    if st.session_state.prediction_service is None:
        try:
            st.session_state.prediction_service = PredictionService()
            # Try to load model
            if not st.session_state.prediction_service.predictor.load_model():
                st.warning("⚠️ ML Model not found. Please train the model first.")
                with st.expander("How to train the model"):
                    st.code("""
# Option 1: If you have the CSV file
python scripts/export_model_from_notebook.py --csv-path /path/to/Loan_default.csv

# Option 2: Place Loan_default.csv in data/ folder
python scripts/export_model_from_notebook.py
                    """)
        except Exception as e:
            st.error(f"Error initializing prediction service: {e}")

    # Assessment mode selector (keeps original UI feel but we still keep separate forms)
    st.write("Select what you want to run:")
    assessment_mode = st.radio(
        "Assessment Mode",
        ["ML Model (Recommended)", "RAG Agent", "Both"],
        horizontal=True
    )
    use_ml = assessment_mode in ["ML Model (Recommended)", "Both"]
    use_rag = assessment_mode in ["RAG Agent", "Both"]

    # RAG checks (only relevant if user tries to use RAG)
    if use_rag:
        # if not check_groq_api_key():
        #     st.error("Please set GROQ_API_KEY environment variable to use RAG assessment.")
        index_path = Path("./data/faiss_index.bin")
        metadata_path = Path("./data/metadata.jsonl")
        if (not index_path.exists() or not metadata_path.exists()) and use_rag:
            st.warning("⚠️ Knowledge Base Not Found (RAG will be unavailable)")

    # Top-level user id input (shared)
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input("User ID", value=st.session_state.user_id or "")
    with col2:
        if st.button("Load Saved Profile"):
            if user_id_input:
                profile = get_user_profile(user_id_input)
                if profile:
                    st.session_state.user_id = user_id_input
                    st.session_state.profile = profile
                    st.success(f"Loaded profile for {user_id_input}")
                else:
                    st.warning(f"No saved profile found for {user_id_input}")
    if user_id_input:
        st.session_state.user_id = user_id_input

    st.markdown("---")

    # -------------------------
    # STRUCTURED FORM (ML only)
    # -------------------------
    st.subheader("A. Structured data (for ML prediction)")
    st.info("Enter only the structured fields the ML model was trained on. You do NOT need to provide unconventional data here.")
    with st.form("structured_form"):
        col1, col2 = st.columns(2)
        with col1:
            loanid = st.text_input("LoanID", value=st.session_state.get("loanid", f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"))
            age = st.number_input("Age", min_value=18, max_value=100, value=int(st.session_state.get("age", 30)))
            income = st.number_input("Income", min_value=0.0, value=float(st.session_state.get("income", 50000.0)))
            loan_amount = st.number_input("LoanAmount (existing/outstanding)", min_value=0.0, value=float(st.session_state.get("loan_amount", 0.0)))
            requested_loan_amount = st.number_input("RequestedLoanAmount (requested)", min_value=0.0, value=float(st.session_state.get("requested_loan_amount", loan_amount)))
            credit_score = st.number_input("CreditScore", min_value=0, max_value=1000, value=int(st.session_state.get("credit_score", 650)))
        with col2:
            months_employed = st.number_input("MonthsEmployed", min_value=0, value=int(st.session_state.get("months_employed", 24)))
            num_credit_lines = st.number_input("NumCreditLines", min_value=0, max_value=100, value=int(st.session_state.get("num_credit_lines", 3)))
            interest_rate = st.number_input("InterestRate (%)", min_value=0.0, value=float(st.session_state.get("interest_rate", 10.0)), format="%.2f")
            loan_term = st.number_input("LoanTerm (months)", min_value=0, value=int(st.session_state.get("loan_term", 36)))
            dti_ratio = st.number_input("DTIRatio", min_value=0.0, value=float(st.session_state.get("dti_ratio", 0.3)), format="%.2f")
            education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], index=0 if st.session_state.get("education") is None else ["High School", "Bachelor's", "Master's", "PhD"].index(st.session_state.get("education")))
            employment_type = st.selectbox("EmploymentType", ["Full-time", "Part-time", "Unemployed", "Self-employed"], index=0 if st.session_state.get("employment_type") is None else ["Full-time", "Part-time", "Unemployed", "Self-employed"].index(st.session_state.get("employment_type")))
            marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"], index=0 if st.session_state.get("marital_status") is None else ["Single", "Married", "Divorced"].index(st.session_state.get("marital_status")))
            has_mortgage = st.selectbox("HasMortgage", ["Yes", "No"], index=0 if st.session_state.get("has_mortgage") is None else ["Yes", "No"].index(st.session_state.get("has_mortgage")))
            has_dependents = st.selectbox("HasDependents", ["Yes", "No"], index=0 if st.session_state.get("has_dependents") is None else ["Yes", "No"].index(st.session_state.get("has_dependents")))
            loan_purpose = st.selectbox("LoanPurpose", ["Auto", "Business", "Education", "Home", "Other"], index=0 if st.session_state.get("loan_purpose") is None else ["Auto", "Business", "Education", "Home", "Other"].index(st.session_state.get("loan_purpose")))
            has_cosigner = st.selectbox("HasCoSigner", ["Yes", "No"], index=0 if st.session_state.get("has_cosigner") is None else ["Yes", "No"].index(st.session_state.get("has_cosigner")))

        run_ml = st.form_submit_button("Run ML Prediction", type="primary")

        if run_ml:
            # Save structured values into session so user can load them later
            st.session_state["loanid"] = loanid
            st.session_state["age"] = age
            st.session_state["income"] = income
            st.session_state["loan_amount"] = loan_amount
            st.session_state["requested_loan_amount"] = requested_loan_amount
            st.session_state["credit_score"] = credit_score
            st.session_state["months_employed"] = months_employed
            st.session_state["num_credit_lines"] = num_credit_lines
            st.session_state["interest_rate"] = interest_rate
            st.session_state["loan_term"] = loan_term
            st.session_state["dti_ratio"] = dti_ratio
            st.session_state["education"] = education
            st.session_state["employment_type"] = employment_type
            st.session_state["marital_status"] = marital_status
            st.session_state["has_mortgage"] = has_mortgage
            st.session_state["has_dependents"] = has_dependents
            st.session_state["loan_purpose"] = loan_purpose
            st.session_state["has_cosigner"] = has_cosigner

            if not (st.session_state.prediction_service and st.session_state.prediction_service.predictor.is_loaded):
                st.warning("ML model not loaded. Please train or load the model first.")
            else:
                # Build structured-only dict to pass to prediction service (only ML features — NO unconventional)
                ml_structured_inputs = {
                    "LoanID": loanid,
                    "Age": age,
                    "Income": income,
                    "LoanAmount": loan_amount,
                    "RequestedLoanAmount": requested_loan_amount,
                    "CreditScore": credit_score,
                    "MonthsEmployed": months_employed,
                    "NumCreditLines": num_credit_lines,
                    "InterestRate": interest_rate,
                    "LoanTerm": loan_term,
                    "DTIRatio": dti_ratio,
                    "Education": education,
                    "EmploymentType": employment_type,
                    "MaritalStatus": marital_status,
                    "HasMortgage": has_mortgage,
                    "HasDependents": has_dependents,
                    "LoanPurpose": loan_purpose,
                    "HasCoSigner": has_cosigner
                }

                # Debug: show keys we send to model (text only)
                st.info(f"ML input keys passed: {list(ml_structured_inputs.keys())}")

                # Call prediction service - it should accept dict/df; your service will align if needed
                try:
                    ml_prediction = st.session_state.prediction_service.predict_from_raw_inputs(
                        ml_structured_inputs,
                        consent=False,
                        user_id=st.session_state.get("user_id", loanid)
                    )
                    st.session_state.ml_prediction = ml_prediction
                    st.success("ML prediction completed.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"ML prediction failed: {e}")
                    logger.exception(e)


    st.markdown("---")

    # -------------------------
    # UNCONVENTIONAL FORM (RAG only)
    # -------------------------
    st.subheader("B. Unconventional data (for RAG assessment)")
    st.info("Upload mobile/psychometric/behavioral/loan-history JSON or enter data manually. This is *not* required for ML prediction.")
    with st.form("unconventional_form"):
        # Mobile metadata
        with st.expander("Mobile Metadata", expanded=False):
            mobile_file = st.file_uploader("Upload mobile data JSON", type=["json"], key="mobile_unconv")
            if mobile_file:
                mobile_data = json.load(mobile_file)
            else:
                mobile_data = {}
                col1, col2 = st.columns(2)
                with col1:
                    mobile_data["avg_daily_calls"] = st.number_input("Avg Daily Calls", min_value=0.0, value=float(mobile_data.get("avg_daily_calls", 0.0)), key="u_avg_calls")
                    mobile_data["unique_contacts_30d"] = st.number_input("Unique Contacts (30d)", min_value=0, value=int(mobile_data.get("unique_contacts_30d", 0)), key="u_contacts")
                with col2:
                    mobile_data["airtime_topup_frequency"] = st.number_input("Airtime Topup Frequency", min_value=0.0, value=float(mobile_data.get("airtime_topup_frequency", 0.0)), key="u_topup_freq")
                    mobile_data["avg_topup_amount"] = st.number_input("Avg Topup Amount", min_value=0.0, value=float(mobile_data.get("avg_topup_amount", 0.0)), key="u_topup_amt")

        # Psychometrics
        with st.expander("Psychometric Data", expanded=False):
            psych_file = st.file_uploader("Upload psychometric responses JSON", type=["json"], key="psych_unconv")
            if psych_file:
                psych_data = json.load(psych_file)
            else:
                psych_data = {}
                st.write("Enter psychometric responses (1-5 scale)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    psych_data["C_q1"] = int(st.slider("C_q1", 1, 5, int(psych_data.get("C_q1", 3)), key="u_c1"))
                    psych_data["C_q2"] = int(st.slider("C_q2", 1, 5, int(psych_data.get("C_q2", 3)), key="u_c2"))
                with col2:
                    psych_data["C_q3"] = int(st.slider("C_q3", 1, 5, int(psych_data.get("C_q3", 3)), key="u_c3"))
                    psych_data["C_q4"] = int(st.slider("C_q4", 1, 5, int(psych_data.get("C_q4", 3)), key="u_c4"))
                with col3:
                    psych_data["C_q5"] = int(st.slider("C_q5", 1, 5, int(psych_data.get("C_q5", 3)), key="u_c5"))

        # Financial behavior
        with st.expander("Financial Behavior", expanded=False):
            fin_file = st.file_uploader("Upload financial data JSON", type=["json"], key="fin_unconv")
            if fin_file:
                fin_data = json.load(fin_file)
            else:
                fin_data = {}
                col1, col2 = st.columns(2)
                with col1:
                    fin_data["savings_frequency"] = st.number_input("Savings Frequency", min_value=0.0, value=float(fin_data.get("savings_frequency", 0.0)), key="u_sav_freq")
                    fin_data["bill_payment_timeliness"] = st.number_input("Bill Payment Timeliness (0-1)", min_value=0.0, max_value=1.0, value=float(fin_data.get("bill_payment_timeliness", 0.0)), key="u_bill_time")
                with col2:
                    fin_data["wallet_balance_lows_last_90d"] = st.number_input("Wallet Balance Lows (90d)", min_value=0, value=int(fin_data.get("wallet_balance_lows_last_90d", 0)), key="u_wallet_lows")

        # Social network
        with st.expander("Social Network", expanded=False):
            social_data = {}
            social_data["shg_membership"] = st.checkbox("SHG Membership", key="u_shg")
            if social_data["shg_membership"]:
                social_data["peer_monitoring_strength"] = float(st.slider("Peer Monitoring Strength (0-1)", 0.0, 1.0, float(st.session_state.get("peer_monitoring_strength", 0.5)), key="u_peer_mon"))
            else:
                social_data["peer_monitoring_strength"] = "NA"

        # Loan history
        with st.expander("Loan History", expanded=False):
            loan_file = st.file_uploader("Upload loan history JSON", type=["json"], key="u_loan")
            if loan_file:
                loan_data = json.load(loan_file)
            else:
                loan_data = {}
                col1, col2 = st.columns(2)
                with col1:
                    loan_data["previous_loans"] = int(st.number_input("Previous Loans", min_value=0, value=int(loan_data.get("previous_loans", 0)), key="u_prev_loans"))
                    loan_data["previous_defaults"] = int(st.number_input("Previous Defaults", min_value=0, value=int(loan_data.get("previous_defaults", 0)), key="u_defaults"))
                with col2:
                    loan_data["previous_late_payments"] = int(st.number_input("Previous Late Payments", min_value=0, value=int(loan_data.get("previous_late_payments", 0)), key="u_late_pay"))
                    loan_data["avg_repayment_delay_days"] = float(st.number_input("Avg Repayment Delay (days)", min_value=0.0, value=float(loan_data.get("avg_repayment_delay_days", 0.0)), key="u_delay"))

        # Full profile upload option (JSON)
        st.markdown("---")
        st.subheader("Or: Upload Complete Profile JSON to build RAG profile")
        full_json_file = st.file_uploader("Upload complete profile JSON", type=["json"], key="full_profile_json")
        run_rag = st.form_submit_button("Run RAG Assessment", type="secondary")

        if run_rag:
            # Build profile for RAG agent
            try:
                if full_json_file:
                    raw_data = json.load(full_json_file)
                    profile = build_profile_from_json(raw_data)
                    if user_id_input:
                        profile["user_id"] = user_id_input
                else:
                    # use demographics from ML structured form if present in session_state, else defaults
                    demographics = {
                        "age": st.session_state.get("age", age if 'age' in locals() else 30),
                        "gender": st.session_state.get("gender", "prefer_not_to_say"),
                        "occupation": st.session_state.get("occupation", "other"),
                        "monthly_income": st.session_state.get("monthly_income", st.session_state.get("income", income if 'income' in locals() else 0.0))
                    }
                    raw_unconventional = {
                        "mobile_metadata": mobile_data,
                        "psychometric_responses": psych_data,
                        "financial_behavior": fin_data,
                        "social_network": social_data,
                        "loan_history": loan_data
                    }
                    profile = build_profile(
                        user_id=st.session_state.get("user_id", user_id_input or f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                        demographics=demographics,
                        raw_unconventional_data=raw_unconventional
                    )

                # Validate profile
                is_valid, errors = validate_profile_schema(profile)
                if not is_valid:
                    st.error(f"Profile validation errors: {errors}")
                    st.stop()

                profile = sanitize_profile(profile)
                save_user_profile(profile["user_id"], profile)
                st.session_state.profile = profile
                st.session_state.user_id = profile["user_id"]

                # Run RAG if configured
                if not check_groq_api_key():
                    st.error("GROQ_API_KEY not configured - cannot run RAG.")
                else:
                    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
                    if api_key and api_key != "your-groq-api-key-here":
                        model_name = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")
                        agent = RAGAgent(groq_api_key=api_key, model_name=model_name)
                        try:
                            decision = agent.assess_eligibility(profile, k=6)
                            st.session_state.decision = decision
                            log_action("assessment_run", profile["user_id"], {"eligibility": decision.get("eligibility")})
                            st.success("RAG assessment complete.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"RAG agent failed: {e}")
                            logger.exception(e)
                    else:
                        st.error("Invalid GROQ API key configuration.")

            except Exception as e:
                st.error(f"Error building profile / running RAG: {e}")
                logger.exception(e)

    # -------------------------
    # Display results & consent
    # -------------------------
    if st.session_state.ml_prediction:
        display_ml_results(st.session_state.ml_prediction)
    if st.session_state.decision:
        display_results(st.session_state.decision)
    if st.session_state.ml_prediction or st.session_state.decision:
        consent_section()


def display_ml_results(ml_prediction: Dict[str, Any]):
    """Display ML model prediction results."""
    st.markdown("---")
    st.header("ML Model Prediction Results")
    
    prediction = ml_prediction.get("prediction", 0) if isinstance(ml_prediction, dict) else (ml_prediction if isinstance(ml_prediction, int) else 0)
    probability = ml_prediction.get("probability", 0.5) if isinstance(ml_prediction, dict) else 0.5
    explanation = ml_prediction.get("explanation", []) if isinstance(ml_prediction, dict) else []
    next_step = ml_prediction.get("next_step", {}) if isinstance(ml_prediction, dict) else {}
    meta = ml_prediction.get("meta", {}) if isinstance(ml_prediction, dict) else {}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction == 1:
            st.success(f"✅ Prediction: **ELIGIBLE**")
        else:
            st.error(f"❌ Prediction: **NOT ELIGIBLE**")
    
    with col2:
        st.metric("Eligibility Probability", f"{probability:.2%}")
    
    with col3:
        if meta.get("confidence_low", False):
            st.warning("⚠️ Low Confidence")
            if meta.get("confidence_reasons"):
                with st.expander("Confidence Issues"):
                    for reason in meta["confidence_reasons"]:
                        st.write(f"• {reason}")
        else:
            st.success("✅ High Confidence")
    
    if explanation:
        st.markdown("### Key Factors")
        if prediction == 1:
            st.info("**Strong Points (contributing to eligibility):**")
        else:
            st.info("**Weak Points (reducing eligibility):**")
        
        for i, exp in enumerate(explanation[:6], 1):
            shap_val = exp.get("shap_value", 0)
            note = exp.get("note", "")
            feature = exp.get("feature", "Unknown")
            
            if prediction == 1 and shap_val > 0:
                st.success(f"{i}. **{feature}**: {note}")
            elif prediction == 0 and shap_val < 0:
                st.error(f"{i}. **{feature}**: {note}")
            else:
                st.info(f"{i}. **{feature}**: {note}")
    
    if next_step:
        st.markdown("### Next Steps")
        action = next_step.get("action", "")
        message = next_step.get("message", "")
        
        if action == "wellness_coach":
            st.warning(message)
            if st.button("Go to Wellness Coach", type="primary"):
                st.session_state.page = "Wellness Coach"
                st.rerun()
        else:
            st.info(message)
    
    with st.expander("View Raw Prediction Data"):
        st.json(ml_prediction)


def display_results(decision: Dict[str, Any]):
    """Display assessment results."""
    st.markdown("---")
    st.header("Assessment Results")
    
    eligibility = decision.get("eligibility", "maybe").lower()
    risk_score = decision.get("risk_score", 0.5)
    confidence = decision.get("confidence", "low").lower()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if eligibility == "yes":
            st.success(f"✅ Eligibility: **{eligibility.upper()}**")
        elif eligibility == "no":
            st.error(f"❌ Eligibility: **{eligibility.upper()}**")
        else:
            st.warning(f"⚠️ Eligibility: **{eligibility.upper()}**")
    
    with col2:
        if risk_score < 0.3:
            st.metric("Risk Score", f"{risk_score:.2f}", delta="Low Risk", delta_color="inverse")
        elif risk_score < 0.7:
            st.metric("Risk Score", f"{risk_score:.2f}", delta="Medium Risk")
        else:
            st.metric("Risk Score", f"{risk_score:.2f}", delta="High Risk", delta_color="inverse")
    
    with col3:
        if confidence == "high":
            st.success(f"Confidence: **{confidence.upper()}**")
        elif confidence == "medium":
            st.info(f"Confidence: **{confidence.upper()}**")
        else:
            st.warning(f"Confidence: **{confidence.upper()}**")
    
    st.markdown("### Verdict")
    st.info(decision.get("verdict_text", "No verdict available"))
    
    st.markdown("### Strong Points")
    strong_points = decision.get("strong_points", [])
    for i, point in enumerate(strong_points[:3], 1):
        st.success(f"{i}. {point}")
    
    st.markdown("### Weak Points")
    weak_points = decision.get("weak_points", [])
    for i, point in enumerate(weak_points[:3], 1):
        st.error(f"{i}. {point}")
    
    st.markdown("### Actionable Recommendations")
    recommendations = decision.get("actionable_recommendations", [])
    for i, rec in enumerate(recommendations[:4], 1):
        st.info(f"{i}. {rec}")
    
    required_data = decision.get("required_unconventional_data", [])
    if required_data:
        st.markdown("### Required Additional Data")
        for item in required_data:
            st.warning(f"• {item}")
    
    with st.expander("View Raw JSON"):
        st.json(decision)


def consent_section():
    """Consent mechanism."""
    st.markdown("---")
    st.header("Data Consent")
    
    st.info("""
    **What will be shared with the bank:**
    - Your profile (demographics and aggregated features)
    - Assessment results (eligibility, risk score, recommendations)
    
    **What will NOT be shared:**
    - Raw call logs, transaction details, or personal identifiers
    - Psychometric responses (only aggregated scores)
    
    **Your rights:**
    - You can decline consent and all your data will be deleted
    - You can request data deletion at any time
    """)
    
    consent = st.checkbox("I consent to sharing my data with the bank", key="consent_checkbox")
    
    if not st.session_state.consent_given:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Submit with Consent", type="primary", use_container_width=True, disabled=not consent):
                if st.session_state.profile:
                    user_id = st.session_state.profile["user_id"]
                    
                    if st.session_state.ml_prediction and st.session_state.prediction_service:
                        try:
                            raw_inputs = {
                                "age": st.session_state.profile["demographics"].get("age", 30),
                                "gender": st.session_state.profile["demographics"].get("gender", "unknown"),
                                "occupation": st.session_state.profile["demographics"].get("occupation", "unknown"),
                                "monthly_income": st.session_state.profile["demographics"].get("monthly_income", 0.0),
                            }
                            st.session_state.prediction_service.predict_from_raw_inputs(
                                raw_inputs,
                                consent=True,
                                user_id=user_id
                            )
                        except Exception as e:
                            logger.error(f"Error saving ML prediction with consent: {e}")
                    
                    if st.session_state.decision:
                        save_consented_submission(
                            user_id,
                            st.session_state.profile,
                            st.session_state.decision
                        )
                    
                    st.session_state.consent_given = True
                    log_action("consent_given", user_id)
                    st.success("✅ Consent given! Data transferred to bank (sandbox).")
                    st.rerun()
        
        with col2:
            if st.button("❌ Decline Consent", use_container_width=True):
                if st.session_state.profile:
                    user_id = st.session_state.profile["user_id"]
                    delete_user_data(user_id)
                    st.session_state.profile = None
                    st.session_state.decision = None
                    st.session_state.ml_prediction = None
                    st.session_state.user_id = None
                    st.session_state.consent_given = False
                    log_action("consent_denied", user_id)
                    st.warning("❌ Consent declined. All data deleted. You can re-run the assessment when ready.")
                    st.rerun()
    else:
        st.success("✅ You have already given consent. Data is stored in the bank database.")

def admin_page():
    """Admin view of bank database."""
    st.header("Admin View - Bank Database")
    
    admin_password = st.text_input("Admin Password", type="password", key="admin_pwd")
    
    if admin_password == "admin123":
        st.success("✅ Authenticated")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Retrain Model", help="Trigger model retraining"):
                st.info("Model retraining triggered. This would run the training script in production.")
                log_action("admin_retrain_triggered", None)
        
        with col2:
            if st.button("📊 View Telemetry", help="View feature distribution telemetry"):
                st.info("Telemetry view would show feature distributions here.")
        
        with col3:
            if st.button("🔄 Reload Model", help="Reload the ML model"):
                if st.session_state.prediction_service:
                    if st.session_state.prediction_service.predictor.load_model():
                        st.success("Model reloaded successfully")
                    else:
                        st.error("Failed to reload model")
        
        submissions = get_bank_submissions(limit=100)
        
        st.metric("Total Consented Submissions", len(submissions))
        
        if submissions:
            st.markdown("### Recent Submissions")
            
            for i, sub in enumerate(submissions[:20], 1):
                with st.expander(f"Submission {i}: {sub['user_id']} - {sub['timestamp'][:19]}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        decision = sub.get('decision', {})
                        eligibility = decision.get('eligibility', 'unknown')
                        st.write("**Eligibility:**", eligibility.upper())
                        st.write("**Risk Score:**", decision.get('risk_score', 'N/A'))
                        if 'confidence' in decision:
                            st.write("**Confidence:**", decision['confidence'].upper())
                    with col2:
                        st.write("**User ID:**", sub['user_id'])
                        st.write("**Timestamp:**", sub['timestamp'])
                    with col3:
                        if st.button(f"Delete", key=f"delete_{i}"):
                            st.warning("Deletion would be implemented here")
                            log_action("admin_delete_submission", sub['user_id'])
                    
                    st.json(sub)
        else:
            st.info("No submissions yet.")
    else:
        if admin_password:
            st.error("❌ Incorrect password")
        else:
            st.info("Enter admin password to view bank database")

def wellness_coach_page():
    """Wellness coach page for users not eligible for loans."""
    st.header("💚 Microfinance Wellness Coach")
    
    st.info("""
    Welcome to the Microfinance Wellness Coach! 
    
    If you were not eligible for a loan, we're here to help you improve your financial profile
    and work towards eligibility in the future.
    """)
    
    st.markdown("### Areas to Improve")
    
    suggestions = [
        "**Build Credit History**: Start with smaller loans and repay on time",
        "**Increase Savings Frequency**: Regular savings demonstrate financial discipline",
        "**Improve Bill Payment Timeliness**: Pay bills on time to build trust",
        "**Join SHG (Self-Help Group)**: Group membership shows social capital",
        "**Increase Income Stability**: Show consistent income over time",
        "**Reduce Wallet Balance Lows**: Maintain minimum balance to show financial stability"
    ]
    
    for suggestion in suggestions:
        st.info(suggestion)
    
    st.markdown("### Resources")
    st.write("""
    - Financial literacy workshops
    - Savings group formation assistance
    - Credit counseling services
    - Income generation training
    """)
    
    if st.button("Return to Assessment"):
        st.session_state.page = "Assessment"
        st.rerun()


if __name__ == "__main__":
    main()
