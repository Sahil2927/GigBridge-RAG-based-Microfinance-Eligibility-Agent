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


def check_groq_api_key() -> bool:
    """Check if Groq API key is set."""
    # First check environment variable
    api_key = os.getenv("GROQ_API_KEY")
    
    # Then check Streamlit secrets (if available)
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", None)
        except (st_errors.StreamlitSecretNotFoundError, FileNotFoundError, AttributeError, Exception):
            # Secrets file doesn't exist, that's okay - we'll use .env file
            api_key = None
    
    return api_key is not None and api_key != "your-groq-api-key-here"


def main():
    """Main application."""
    st.title("💰 RAG-based Microfinance Eligibility Agent")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        # Handle page navigation
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
        if not check_groq_api_key():
            st.error("⚠️ GROQ_API_KEY not set!")
            st.code("export GROQ_API_KEY='your-key-here'")
            st.info("Get your API key from: https://console.groq.com/")
        else:
            st.success("✅ Groq API key configured")
    
    if st.session_state.page == "Assessment":
        assessment_page()
    elif st.session_state.page == "Admin View":
        admin_page()
    elif st.session_state.page == "Wellness Coach":
        wellness_coach_page()


def assessment_page():
    """Assessment page with user input form."""
    st.header("Eligibility Assessment")
    
    # Initialize prediction service
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
    
    # Assessment mode selection
    assessment_mode = st.radio(
        "Assessment Mode",
        ["ML Model (Recommended)", "RAG Agent", "Both"],
        horizontal=True
    )
    
    use_ml = assessment_mode in ["ML Model (Recommended)", "Both"]
    use_rag = assessment_mode in ["RAG Agent", "Both"]
    
    # Check RAG requirements if needed
    if use_rag:
        if not check_groq_api_key():
            st.error("Please set GROQ_API_KEY environment variable to use RAG assessment.")
            if not use_ml:
                st.stop()
        
        # Check if FAISS index exists
        index_path = Path("./data/faiss_index.bin")
        metadata_path = Path("./data/metadata.jsonl")
        
        if not index_path.exists() or not metadata_path.exists():
            st.warning("⚠️ Knowledge Base Not Found (RAG will be unavailable)")
            if not use_ml:
                st.stop()
    
    # User ID input
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input(
            "User ID",
            value=st.session_state.user_id or "",
            help="Enter a unique user identifier"
        )
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
    
    # Form sections
    with st.form("eligibility_form"):
        st.subheader("1. Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["male", "female", "other", "prefer_not_to_say"])
        with col2:
            occupation = st.selectbox(
                "Occupation",
                ["gig_worker", "small_business_owner", "daily_wage_worker", "farmer", "other"]
            )
            monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=20000)
        
        st.markdown("---")
        st.subheader("2. Unconventional Data")
        st.info("""
        **Data Privacy Notice:** 
        Raw data (call logs, transaction history, etc.) will be stored temporarily for processing.
        Only aggregated features will be shared with the bank after your consent.
        You can decline consent and all data will be deleted.
        """)
        
        # Mobile metadata
        with st.expander("Mobile Metadata", expanded=False):
            st.write("Upload JSON file or enter data manually")
            
            mobile_file = st.file_uploader("Upload mobile data JSON", type=["json"], key="mobile")
            if mobile_file:
                mobile_data = json.load(mobile_file)
            else:
                mobile_data = {}
                col1, col2 = st.columns(2)
                with col1:
                    mobile_data["avg_daily_calls"] = st.number_input("Avg Daily Calls", value=0.0, key="avg_calls")
                    mobile_data["unique_contacts_30d"] = st.number_input("Unique Contacts (30d)", value=0, key="contacts")
                with col2:
                    mobile_data["airtime_topup_frequency"] = st.number_input("Airtime Topup Frequency", value=0.0, key="topup_freq")
                    mobile_data["avg_topup_amount"] = st.number_input("Avg Topup Amount", value=0.0, key="topup_amt")
        
        # Psychometrics
        with st.expander("Psychometric Data", expanded=False):
            psych_file = st.file_uploader("Upload psychometric responses JSON", type=["json"], key="psych")
            if psych_file:
                psych_data = json.load(psych_file)
            else:
                psych_data = {}
                st.write("Enter psychometric responses (1-5 scale)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    psych_data["C_q1"] = st.slider("C_q1", 1, 5, 3, key="c1")
                    psych_data["C_q2"] = st.slider("C_q2", 1, 5, 3, key="c2")
                with col2:
                    psych_data["C_q3"] = st.slider("C_q3", 1, 5, 3, key="c3")
                    psych_data["C_q4"] = st.slider("C_q4", 1, 5, 3, key="c4")
                with col3:
                    psych_data["C_q5"] = st.slider("C_q5", 1, 5, 3, key="c5")
        
        # Financial behavior
        with st.expander("Financial Behavior", expanded=False):
            fin_file = st.file_uploader("Upload financial data JSON", type=["json"], key="fin")
            if fin_file:
                fin_data = json.load(fin_file)
            else:
                fin_data = {}
                col1, col2 = st.columns(2)
                with col1:
                    fin_data["savings_frequency"] = st.number_input("Savings Frequency", value=0.0, key="sav_freq")
                    fin_data["bill_payment_timeliness"] = st.number_input("Bill Payment Timeliness (0-1)", value=0.0, min_value=0.0, max_value=1.0, key="bill_time")
                with col2:
                    fin_data["wallet_balance_lows_last_90d"] = st.number_input("Wallet Balance Lows (90d)", value=0, key="wallet_lows")
        
        # Social network
        with st.expander("Social Network", expanded=False):
            social_data = {}
            social_data["shg_membership"] = st.checkbox("SHG Membership", key="shg")
            if social_data["shg_membership"]:
                social_data["peer_monitoring_strength"] = st.slider("Peer Monitoring Strength (0-1)", 0.0, 1.0, 0.5, key="peer_mon")
            else:
                social_data["peer_monitoring_strength"] = "NA"
        
        # Loan history
        with st.expander("Loan History", expanded=False):
            loan_file = st.file_uploader("Upload loan history JSON", type=["json"], key="loan")
            if loan_file:
                loan_data = json.load(loan_file)
            else:
                loan_data = {}
                col1, col2 = st.columns(2)
                with col1:
                    loan_data["previous_loans"] = st.number_input("Previous Loans", value=0, key="prev_loans")
                    loan_data["previous_defaults"] = st.number_input("Previous Defaults", value=0, key="defaults")
                with col2:
                    loan_data["previous_late_payments"] = st.number_input("Previous Late Payments", value=0, key="late_pay")
                    loan_data["avg_repayment_delay_days"] = st.number_input("Avg Repayment Delay (days)", value=0.0, key="delay")
        
        # Full JSON upload option
        st.markdown("---")
        st.subheader("Or: Upload Complete Profile JSON")
        full_json_file = st.file_uploader("Upload complete profile JSON", type=["json"], key="full_json")
        
        submitted = st.form_submit_button("Run Assessment", type="primary", use_container_width=True)
        
        if submitted:
            with st.spinner("Processing assessment..."):
                try:
                    # Build profile
                    if full_json_file:
                        # Load from full JSON
                        raw_data = json.load(full_json_file)
                        profile = build_profile_from_json(raw_data)
                        if user_id_input:
                            profile["user_id"] = user_id_input
                    else:
                        # Build from form inputs
                        demographics = {
                            "age": age,
                            "gender": gender,
                            "occupation": occupation,
                            "monthly_income": monthly_income
                        }
                        
                        raw_unconventional = {
                            "mobile_metadata": mobile_data,
                            "psychometric_responses": psych_data,
                            "financial_behavior": fin_data,
                            "social_network": social_data,
                            "loan_history": loan_data
                        }
                        
                        profile = build_profile(
                            user_id=user_id_input or f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            demographics=demographics,
                            raw_unconventional_data=raw_unconventional
                        )
                    
                    # Validate profile
                    is_valid, errors = validate_profile_schema(profile)
                    if not is_valid:
                        st.error(f"Profile validation errors: {errors}")
                        st.stop()
                    
                    # Sanitize profile
                    profile = sanitize_profile(profile)
                    
                    # Save to memory
                    save_user_profile(profile["user_id"], profile)
                    st.session_state.profile = profile
                    st.session_state.user_id = profile["user_id"]
                    
                    # Prepare raw inputs for ML model
                    raw_inputs = {
                        "age": age,
                        "gender": gender,
                        "occupation": occupation,
                        "monthly_income": monthly_income,
                        "avg_daily_calls": mobile_data.get("avg_daily_calls", 0.0),
                        "unique_contacts_30d": mobile_data.get("unique_contacts_30d", 0),
                        "airtime_topup_frequency": mobile_data.get("airtime_topup_frequency", 0.0),
                        "avg_topup_amount": mobile_data.get("avg_topup_amount", 0.0),
                        "days_inactive_last_30": mobile_data.get("days_inactive_last_30", 0),
                        "conscientiousness_score": psych_data.get("conscientiousness_score", 
                            sum([psych_data.get(f"C_q{i}", 3) for i in range(1, 6)]) / 5.0 if any(f"C_q{i}" in psych_data for i in range(1, 6)) else 3.0),
                        "savings_frequency": fin_data.get("savings_frequency", 0.0),
                        "bill_payment_timeliness": fin_data.get("bill_payment_timeliness", 0.5),
                        "wallet_balance_lows_last_90d": fin_data.get("wallet_balance_lows_last_90d", 0),
                        "shg_membership": social_data.get("shg_membership", False),
                        "peer_monitoring_strength": social_data.get("peer_monitoring_strength", 0.0) if social_data.get("shg_membership") else 0.0,
                        "previous_loans": loan_data.get("previous_loans", 0),
                        "previous_defaults": loan_data.get("previous_defaults", 0),
                        "previous_late_payments": loan_data.get("previous_late_payments", 0),
                        "avg_repayment_delay_days": loan_data.get("avg_repayment_delay_days", 0.0)
                    }
                    
                    # Run ML prediction if enabled
                    ml_prediction = None
                    if use_ml and st.session_state.prediction_service and st.session_state.prediction_service.predictor.is_loaded:
                        try:
                            ml_prediction = st.session_state.prediction_service.predict_from_raw_inputs(
                                raw_inputs,
                                consent=False,  # Consent handled separately
                                user_id=profile["user_id"]
                            )
                            st.session_state.ml_prediction = ml_prediction
                        except Exception as e:
                            st.warning(f"ML prediction failed: {e}")
                            if not use_rag:
                                st.error("Assessment failed. Please check model configuration.")
                                st.stop()
                    
                    # Run RAG assessment if enabled
                    decision = None
                    if use_rag:
                        api_key = os.getenv("GROQ_API_KEY")
                        if not api_key:
                            try:
                                api_key = st.secrets.get("GROQ_API_KEY")
                            except (st_errors.StreamlitSecretNotFoundError, FileNotFoundError, AttributeError, Exception):
                                api_key = None
                        
                        # Validate API key
                        if not api_key or api_key == "your-groq-api-key-here":
                            if not use_ml:
                                st.error("⚠️ Invalid API Key Configuration")
                                st.warning("""
                                Your Groq API key is not configured correctly.
                                
                                **To fix:**
                                1. Open the `.env` file in your project root
                                2. Replace `your-groq-api-key-here` with your actual API key
                                3. Get your API key from: https://console.groq.com/
                                4. The key should look like: `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
                                5. Save the file and restart the app
                                """)
                                st.stop()
                        else:
                            # Configure LLM model here: "mixtral-8x7b-32768", "llama3-70b-8192", "gemma-7b-it"
                            model_name = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")
                            
                            try:
                                agent = RAGAgent(groq_api_key=api_key, model_name=model_name)
                                decision = agent.assess_eligibility(profile, k=6)
                                st.session_state.decision = decision
                            except ValueError as e:
                                # Handle API key validation errors
                                if "Invalid Groq API Key" in str(e):
                                    if not use_ml:
                                        st.error("❌ Invalid API Key")
                                        st.error(str(e))
                                        st.info("💡 Tip: Run `python check_api_key.py` to verify your configuration")
                                        st.stop()
                                else:
                                    raise
                    
                    st.session_state.consent_given = False
                    
                    if decision:
                        log_action("assessment_run", profile["user_id"], {"eligibility": decision["eligibility"]})
                    if ml_prediction:
                        log_action("ml_prediction_run", profile["user_id"], {"prediction": ml_prediction["prediction"]})
                    
                    st.success("Assessment complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during assessment: {str(e)}")
                    st.exception(e)
    
    # Display results if available
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
    
    prediction = ml_prediction.get("prediction", 0)
    probability = ml_prediction.get("probability", 0.5)
    explanation = ml_prediction.get("explanation", [])
    next_step = ml_prediction.get("next_step", {})
    meta = ml_prediction.get("meta", {})
    
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
    
    # Explanation
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
    
    # Next step
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
    
    # Raw JSON (expandable)
    with st.expander("View Raw Prediction Data"):
        st.json(ml_prediction)


def display_results(decision: Dict[str, Any]):
    """Display assessment results."""
    st.markdown("---")
    st.header("Assessment Results")
    
    # Eligibility badge
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
        # Color code risk score
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
    
    # Verdict
    st.markdown("### Verdict")
    st.info(decision.get("verdict_text", "No verdict available"))
    
    # Strong points
    st.markdown("### Strong Points")
    strong_points = decision.get("strong_points", [])
    for i, point in enumerate(strong_points[:3], 1):
        st.success(f"{i}. {point}")
    
    # Weak points
    st.markdown("### Weak Points")
    weak_points = decision.get("weak_points", [])
    for i, point in enumerate(weak_points[:3], 1):
        st.error(f"{i}. {point}")
    
    # Recommendations
    st.markdown("### Actionable Recommendations")
    recommendations = decision.get("actionable_recommendations", [])
    for i, rec in enumerate(recommendations[:4], 1):
        st.info(f"{i}. {rec}")
    
    # Required data
    required_data = decision.get("required_unconventional_data", [])
    if required_data:
        st.markdown("### Required Additional Data")
        for item in required_data:
            st.warning(f"• {item}")
    
    # Raw JSON (expandable)
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
                    
                    # Save ML prediction with consent if available
                    if st.session_state.ml_prediction and st.session_state.prediction_service:
                        try:
                            # Re-run prediction with consent
                            raw_inputs = {
                                "age": st.session_state.profile["demographics"].get("age", 30),
                                "gender": st.session_state.profile["demographics"].get("gender", "unknown"),
                                "occupation": st.session_state.profile["demographics"].get("occupation", "unknown"),
                                "monthly_income": st.session_state.profile["demographics"].get("monthly_income", 0.0),
                            }
                            # Add other fields from profile...
                            st.session_state.prediction_service.predict_from_raw_inputs(
                                raw_inputs,
                                consent=True,
                                user_id=user_id
                            )
                        except Exception as e:
                            logger.error(f"Error saving ML prediction with consent: {e}")
                    
                    # Save RAG decision with consent if available
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
    
    # Simple password protection
    admin_password = st.text_input("Admin Password", type="password", key="admin_pwd")
    
    if admin_password == "admin123":  # In production, use proper authentication
        st.success("✅ Authenticated")
        
        # Admin actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Retrain Model", help="Trigger model retraining"):
                st.info("Model retraining triggered. This would run the training script in production.")
                # In production, this would trigger: python scripts/export_model_from_notebook.py
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
        
        # Get submissions
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
                            # Delete submission (would implement proper deletion)
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
    
    # Show improvement suggestions
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

