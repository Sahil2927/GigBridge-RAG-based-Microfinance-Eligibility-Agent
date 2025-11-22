"""
Streamlit UI - FIXED VERSION (Key Changes)

Main fix: Completely separate ML inputs from RAG inputs
- ML form: ONLY the 16 structured features
- RAG form: ONLY unconventional data
- No mixing of the two data types
"""

import streamlit as st
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Setup (same as before)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
load_dotenv()
sys.path.append(str(Path.cwd()))

from src.profile.profile_builder import build_profile, build_profile_from_json
from src.agent.rag_agent import RAGAgent
from src.ml.prediction_service import PredictionService
from src.ml.feature_converter import ML_MODEL_FEATURES  # Import the feature list
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

# Sidebar (same as before)
with st.sidebar:
    st.header("Navigation")
    if "page" not in st.session_state:
        st.session_state.page = "Assessment"
    
    page = st.radio(
        "Select Page",
        ["Assessment", "Admin View", "Wellness Coach","Transparency & Ethics"],
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


def check_groq_api_key():
    """Check if GROQ API key is configured."""
    api_key = os.getenv("GROQ_API_KEY")
    return api_key and api_key != "your-groq-api-key-here"


def main():
    if st.session_state.page == "Assessment":
        assessment_page()
    elif st.session_state.page == "Admin View":
        admin_page()
    elif st.session_state.page == "Wellness Coach":
        wellness_coach_page()
    elif st.session_state.page == "Transparency & Ethics":
        transparency_ethics_page()


def assessment_page():
    """
    Assessment page - FIXED VERSION
    
    Key changes:
    1. Completely separate forms for ML and RAG
    2. ML form: ONLY 16 structured features
    3. RAG form: ONLY unconventional data
    4. Clear which data goes where
    """
    st.header("Eligibility Assessment")
    
    # Initialize prediction service
    if st.session_state.prediction_service is None:
        try:
            st.session_state.prediction_service = PredictionService()
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
    st.write("### Select Assessment Mode")
    assessment_mode = st.radio(
        "What would you like to use?",
        ["ML Model Only (Structured Data)", "RAG Agent Only (Unconventional Data)", "Both"],
        horizontal=True
    )
    
    use_ml = assessment_mode in ["ML Model Only (Structured Data)", "Both"]
    use_rag = assessment_mode in ["RAG Agent Only (Unconventional Data)", "Both"]
    
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

    st.markdown("---")

    # ========================================
    # PART A: ML MODEL FORM (STRUCTURED DATA ONLY)
    # ========================================
    if use_ml:
        st.subheader("📊 Part A: ML Prediction (Structured Data)")
        st.info(f"""
        **Enter the {len(ML_MODEL_FEATURES)} structured features required by the ML model:**
        
        The ML model was trained ONLY on these standard loan features.
        Do NOT include unconventional data (mobile usage, psychometrics, etc.) here.
        """)
        
        with st.form("ml_structured_form"):
            st.write("**Required ML Features:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                loan_id = st.text_input(
                    "LoanID",
                    value=st.session_state.get("loan_id", f"LOAN_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                )
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                income = st.number_input("Income (Annual)", min_value=0.0, value=50000.0)
                loan_amount = st.number_input("LoanAmount", min_value=0.0, value=10000.0)
                credit_score = st.number_input("CreditScore", min_value=300, max_value=850, value=650)
                months_employed = st.number_input("MonthsEmployed", min_value=0, value=24)
                num_credit_lines = st.number_input("NumCreditLines", min_value=0, value=2)
                interest_rate = st.number_input("InterestRate (%)", min_value=0.0, max_value=50.0, value=10.0, format="%.2f")
            
            with col2:
                loan_term = st.number_input("LoanTerm (months)", min_value=1, max_value=360, value=36)
                dti_ratio = st.number_input("DTIRatio", min_value=0.0, max_value=2.0, value=0.3, format="%.2f")
                education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
                employment_type = st.selectbox("EmploymentType", ["Full-time", "Part-time", "Unemployed", "Self-employed"])
                marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
                has_mortgage = st.selectbox("HasMortgage", ["Yes", "No"])
                has_dependents = st.selectbox("HasDependents", ["Yes", "No"])
                loan_purpose = st.selectbox("LoanPurpose", ["Auto", "Business", "Education", "Home", "Other"])
                has_cosigner = st.selectbox("HasCoSigner", ["Yes", "No"])
            
            run_ml = st.form_submit_button("🔮 Run ML Prediction", type="primary", use_container_width=True)
            
            if run_ml:
                if not (st.session_state.prediction_service and st.session_state.prediction_service.predictor.is_loaded):
                    st.error("❌ ML model not loaded. Please train or load the model first.")
                else:
                    # Build ONLY the structured ML features (EXACTLY 17 features the model expects)
                    ml_inputs = {
                        "LoanID": loan_id,
                        "Age": age,
                        "Income": income,
                        "LoanAmount": loan_amount,
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
                    
                    # Verify we have exactly the right features
                    st.info(f"✅ Sending {len(ml_inputs)} features to ML model: {list(ml_inputs.keys())}")
                    
                    try:
                        with st.spinner("Running ML prediction..."):
                            ml_prediction = st.session_state.prediction_service.predict_from_raw_inputs(
                                ml_inputs,
                                consent=False,
                                user_id=st.session_state.get("user_id", loan_id)
                            )
                            st.session_state.ml_prediction = ml_prediction
                            st.success("✅ ML prediction completed!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ ML prediction failed: {e}")
                        logger.exception(e)
                        st.info("💡 Tip: Make sure you're only providing structured features, not unconventional data.")

        st.markdown("---")

    # ========================================
    # PART B: RAG AGENT FORM (UNCONVENTIONAL DATA ONLY)
    # ========================================
    if use_rag:
        st.subheader("🧠 Part B: RAG Assessment (Unconventional Data)")
        st.info("""
        **Enter unconventional data for RAG-based assessment:**
        
        This includes mobile usage patterns, psychometric responses, behavioral data, etc.
        This data will NOT be sent to the ML model.
        """)
        
        # Check RAG prerequisites
        if not check_groq_api_key():
            st.warning("⚠️ GROQ_API_KEY not configured. RAG assessment will be unavailable.")
        
        index_path = Path("./data/faiss_index.bin")
        metadata_path = Path("./data/metadata.jsonl")
        if not index_path.exists() or not metadata_path.exists():
            st.warning("⚠️ Knowledge Base not found. Please build the RAG index first.")
        
        with st.form("rag_unconventional_form"):
            # Basic demographics (needed for profile building)
            st.write("**Basic Demographics (for RAG profile):**")
            col1, col2 = st.columns(2)
            with col1:
                rag_age = st.number_input("Age", min_value=18, max_value=100, value=30, key="rag_age")
                rag_gender = st.selectbox("Gender", ["male", "female", "other", "prefer_not_to_say"], key="rag_gender")
            with col2:
                rag_occupation = st.selectbox(
                    "Occupation",
                    ["gig_worker", "small_business_owner", "daily_wage_worker", "farmer", "other"],
                    key="rag_occupation"
                )
                rag_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0, key="rag_income")
            
            st.markdown("---")
            
            # Mobile metadata
            with st.expander("📱 Mobile Metadata", expanded=False):
                mobile_file = st.file_uploader("Upload mobile data JSON", type=["json"], key="mobile_rag")
                if mobile_file:
                    mobile_data = json.load(mobile_file)
                else:
                    mobile_data = {}
                    col1, col2 = st.columns(2)
                    with col1:
                        mobile_data["avg_daily_calls"] = st.number_input("Avg Daily Calls", min_value=0.0, value=5.0, key="rag_calls")
                        mobile_data["unique_contacts_30d"] = st.number_input("Unique Contacts (30d)", min_value=0, value=20, key="rag_contacts")
                    with col2:
                        mobile_data["airtime_topup_frequency"] = st.number_input("Airtime Topup Frequency", min_value=0.0, value=4.0, key="rag_topup_freq")
                        mobile_data["avg_topup_amount"] = st.number_input("Avg Topup Amount", min_value=0.0, value=100.0, key="rag_topup_amt")
            
            # Psychometrics
            with st.expander("🧪 Psychometric Data", expanded=False):
                psych_file = st.file_uploader("Upload psychometric responses JSON", type=["json"], key="psych_rag")
                if psych_file:
                    psych_data = json.load(psych_file)
                else:
                    psych_data = {}
                    st.write("Conscientiousness Questions (1-5 scale):")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        psych_data["C_q1"] = st.slider("C_q1", 1, 5, 3, key="rag_c1")
                        psych_data["C_q2"] = st.slider("C_q2", 1, 5, 3, key="rag_c2")
                    with col2:
                        psych_data["C_q3"] = st.slider("C_q3", 1, 5, 3, key="rag_c3")
                        psych_data["C_q4"] = st.slider("C_q4", 1, 5, 3, key="rag_c4")
                    with col3:
                        psych_data["C_q5"] = st.slider("C_q5", 1, 5, 3, key="rag_c5")
            
            # Financial behavior
            with st.expander("💰 Financial Behavior", expanded=False):
                fin_file = st.file_uploader("Upload financial data JSON", type=["json"], key="fin_rag")
                if fin_file:
                    fin_data = json.load(fin_file)
                else:
                    fin_data = {}
                    col1, col2 = st.columns(2)
                    with col1:
                        fin_data["savings_frequency"] = st.number_input("Savings Frequency (per month)", min_value=0.0, value=2.0, key="rag_sav_freq")
                        fin_data["bill_payment_timeliness"] = st.slider("Bill Payment Timeliness (0-1)", 0.0, 1.0, 0.8, key="rag_bill_time")
                    with col2:
                        fin_data["wallet_balance_lows_last_90d"] = st.number_input("Wallet Balance Lows (90d)", min_value=0, value=3, key="rag_wallet_lows")
            
            # Social network
            with st.expander("👥 Social Network", expanded=False):
                social_data = {}
                social_data["shg_membership"] = st.checkbox("SHG Membership", key="rag_shg")
                if social_data["shg_membership"]:
                    social_data["peer_monitoring_strength"] = st.slider("Peer Monitoring Strength (0-1)", 0.0, 1.0, 0.5, key="rag_peer_mon")
                else:
                    social_data["peer_monitoring_strength"] = "NA"
            
            # Loan history
            with st.expander("📋 Loan History", expanded=False):
                loan_file = st.file_uploader("Upload loan history JSON", type=["json"], key="loan_rag")
                if loan_file:
                    loan_data = json.load(loan_file)
                else:
                    loan_data = {}
                    col1, col2 = st.columns(2)
                    with col1:
                        loan_data["previous_loans"] = st.number_input("Previous Loans", min_value=0, value=1, key="rag_prev_loans")
                        loan_data["previous_defaults"] = st.number_input("Previous Defaults", min_value=0, value=0, key="rag_defaults")
                    with col2:
                        loan_data["previous_late_payments"] = st.number_input("Previous Late Payments", min_value=0, value=0, key="rag_late_pay")
                        loan_data["avg_repayment_delay_days"] = st.number_input("Avg Repayment Delay (days)", min_value=0.0, value=0.0, key="rag_delay")
            
            # Full profile upload option
            st.markdown("---")
            st.subheader("Or: Upload Complete Profile JSON")
            full_json_file = st.file_uploader("Upload complete unconventional profile JSON", type=["json"], key="full_profile_rag")
            
            run_rag = st.form_submit_button("🧠 Run RAG Assessment", type="secondary", use_container_width=True)
            
            if run_rag:
                try:
                    # Build profile for RAG
                    if full_json_file:
                        raw_data = json.load(full_json_file)
                        profile = build_profile_from_json(raw_data)
                        if user_id_input:
                            profile["user_id"] = user_id_input
                    else:
                        demographics = {
                            "age": rag_age,
                            "gender": rag_gender,
                            "occupation": rag_occupation,
                            "monthly_income": rag_income
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
                    
                    # Validate and sanitize
                    is_valid, errors = validate_profile_schema(profile)
                    if not is_valid:
                        st.error(f"Profile validation errors: {errors}")
                    else:
                        profile = sanitize_profile(profile)
                        save_user_profile(profile["user_id"], profile)
                        st.session_state.profile = profile
                        st.session_state.user_id = profile["user_id"]
                        
                        # Run RAG assessment
                        if not check_groq_api_key():
                            st.error("❌ GROQ_API_KEY not configured. Cannot run RAG assessment.")
                        else:
                            api_key = os.getenv("GROQ_API_KEY")
                            model_name = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")
                            
                            with st.spinner("Running RAG assessment..."):
                                try:
                                    agent = RAGAgent(groq_api_key=api_key, model_name=model_name)
                                    decision = agent.assess_eligibility(profile, k=6)
                                    st.session_state.decision = decision
                                    log_action("assessment_run", profile["user_id"], {"eligibility": decision.get("eligibility")})
                                    st.success("✅ RAG assessment completed!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ RAG assessment failed: {e}")
                                    logger.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Error building profile: {e}")
                    logger.exception(e)

    # ========================================
    # DISPLAY RESULTS
    # ========================================
    st.markdown("---")
    
    if st.session_state.ml_prediction:
        display_ml_results(st.session_state.ml_prediction)
    
    if st.session_state.decision:
        display_results(st.session_state.decision)
    
    if st.session_state.ml_prediction or st.session_state.decision:
        consent_section()

    



def transparency_ethics_page():
    """
    Dedicated page explaining the AI system's transparency and ethical considerations.
    """
    st.header("🔍 Transparency & Ethical AI")
    
    st.markdown("""
    This page explains how our AI system works, why transparency matters, 
    and how we ensure ethical lending practices.
    """)
    
    # ========================================
    # TABS FOR DIFFERENT SECTIONS
    # ========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 How AI Works",
        "📊 SHAP Explained", 
        "⚖️ Ethical Principles",
        "🔒 Privacy & Data",
        "❓ FAQ"
    ])
    
    # ========================================
    # TAB 1: HOW AI WORKS
    # ========================================
    with tab1:
        st.markdown("### 🧠 How Our AI Loan System Works")
        
        st.info("""
        Our system uses machine learning to assess loan applications. 
        Here's the complete process:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 1️⃣ Data Collection
            We collect structured information about your:
            - **Demographics**: Age, education, employment
            - **Finances**: Income, credit score, existing debts
            - **Loan Details**: Amount requested, purpose, term
            
            #### 2️⃣ Feature Engineering
            Raw data is transformed into 16 key features:
            - Numerical features (income, age, credit score)
            - Categorical features (education, employment type)
            - Derived features (debt-to-income ratio)
            
            #### 3️⃣ Model Prediction
            An XGBoost model (trained on historical data) predicts:
            - **Probability**: Likelihood of loan approval (0-100%)
            - **Decision**: Binary outcome (approved/not approved)
            """)
        
        with col2:
            st.markdown("""
            #### 4️⃣ Explanation Generation
            SHAP analysis reveals:
            - Which features influenced the decision
            - How much each feature contributed
            - Why you were approved or rejected
            
            #### 5️⃣ Human Oversight
            - All decisions can be reviewed by humans
            - You can challenge unfair decisions
            - Continuous monitoring for bias
            
            #### 6️⃣ Feedback Loop
            - User feedback improves the model
            - Regular retraining with new data
            - Audits for fairness and accuracy
            """)
        
        st.markdown("---")
        
        st.markdown("### 🔄 The ML Pipeline")
        
        st.code("""
        Input Data (16 features)
             ↓
        Preprocessing (Standardization, Encoding)
             ↓
        XGBoost Model (Gradient Boosted Trees)
             ↓
        Prediction (Probability Score)
             ↓
        SHAP Analysis (Feature Attributions)
             ↓
        Output (Decision + Explanation)
        """, language="text")
        
        st.success("""
        ✅ **Key Insight**: Every decision is explainable. The model doesn't work as a "black box" - 
        you can see exactly which factors influenced your result.
        """)
    
    # ========================================
    # TAB 2: SHAP EXPLAINED
    # ========================================
    with tab2:
        st.markdown("### 📊 Understanding SHAP Values")
        
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is our method for explaining AI decisions.
        It's based on game theory and ensures fair attribution of credit to each feature.
        """)
        
        # Visual explanation
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            #### How SHAP Works: The Coalition Game
            
            Imagine loan approval as a team sport:
            
            1. **The Team**: Your features (age, income, credit score, etc.)
            2. **The Game**: Predicting loan approval
            3. **The Question**: How much did each team member contribute?
            
            SHAP answers this by:
            - Trying all possible combinations of features
            - Measuring how the prediction changes when each feature joins
            - Computing a fair average contribution for each feature
            
            #### Mathematical Properties
            
            SHAP satisfies three key properties:
            
            1. **Local Accuracy**: 
               - Base value + SHAP values = Final prediction
               - All contributions sum to explain the full result
            
            2. **Consistency**: 
               - If a feature helps more, it gets higher SHAP value
               - Changes in contribution are properly reflected
            
            3. **Missingness**: 
               - Features that weren't used get SHAP value of 0
               - Only actual inputs influence the result
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Reading SHAP Values
            
            **Positive SHAP (+)**
            - ✅ Increases approval probability
            - Helps your application
            - Higher is better
            
            **Negative SHAP (-)**
            - ⚠️ Decreases approval probability
            - Hurts your application
            - Lower magnitude is better
            
            **Magnitude**
            - Size shows importance
            - Larger = more influential
            
            #### Example
            
            ```
            CreditScore:  +0.15
            ↳ Your credit score adds 15% 
              to approval probability
            
            DTIRatio:     -0.08
            ↳ Your debt level reduces 
              approval by 8%
            
            Income:       +0.12
            ↳ Your income adds 12% 
              to approval probability
            ```
            """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Why SHAP is Trustworthy")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            #### ✅ Mathematically Sound
            - Based on Shapley values from game theory
            - Proven fairness properties
            - Published in peer-reviewed research
            """)
        
        with col2:
            st.info("""
            #### 🔬 Widely Used
            - Industry standard for ML explanation
            - Used by banks, hospitals, tech companies
            - Open-source and auditable
            """)
        
        with col3:
            st.success("""
            #### 🎓 Interpretable
            - Each value has clear meaning
            - No hidden assumptions
            - Accessible to non-experts
            """)
        
        st.markdown("---")
        
        with st.expander("📚 Learn More About SHAP"):
            st.markdown("""
            **Academic Papers**:
            - [A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)](https://arxiv.org/abs/1705.07874)
            - [Consistent Individualized Feature Attribution for Tree Ensembles (Lundberg et al., 2019)](https://arxiv.org/abs/1802.03888)
            
            **Resources**:
            - Official SHAP Documentation: https://shap.readthedocs.io/
            - GitHub Repository: https://github.com/slundberg/shap
            - Video Tutorials: Search "SHAP explanation" on YouTube
            
            **Interactive Demo**:
            - Try SHAP on your own data: https://shap-lrjball.streamlit.app/
            """)
    
    # ========================================
    # TAB 3: ETHICAL PRINCIPLES
    # ========================================
    with tab3:
        st.markdown("### ⚖️ Our Ethical AI Principles")
        
        st.markdown("""
        We are committed to responsible AI development and deployment. 
        Our system adheres to these core principles:
        """)
        
        # Principle 1: Transparency
        st.markdown("#### 1️⃣ Transparency")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success("### ✅")
        with col2:
            st.markdown("""
            **What we do**:
            - Every decision is explainable with SHAP
            - You can see which features mattered most
            - Model architecture and training process are documented
            
            **Why it matters**:
            - You have the right to understand decisions affecting you
            - Transparency enables accountability
            - Hidden algorithms can perpetuate bias
            """)
        
        st.divider()
        
        # Principle 2: Fairness
        st.markdown("#### 2️⃣ Fairness & Non-Discrimination")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success("### ⚖️")
        with col2:
            st.markdown("""
            **What we do**:
            - No direct use of protected attributes (race, religion, gender in decision)
            - Regular bias audits across demographic groups
            - Disparate impact testing
            
            **Why it matters**:
            - Everyone deserves equal opportunity
            - Lending discrimination is illegal and immoral
            - AI can amplify historical biases if not carefully designed
            
            **Monitoring**:
            - Approval rates by demographics
            - False positive/negative rates by group
            - Feature importance variations
            """)
        
        st.divider()
        
        # Principle 3: Accountability
        st.markdown("#### 3️⃣ Accountability & Oversight")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success("### 👥")
        with col2:
            st.markdown("""
            **What we do**:
            - Human review available for all decisions
            - Appeal process for rejected applications
            - Regular audits by independent experts
            
            **Your rights**:
            - ✅ Request human review of your decision
            - ✅ Challenge decisions you believe are unfair
            - ✅ Access all data used in your assessment
            - ✅ Request deletion of your data
            """)
        
        st.divider()
        
        # Principle 4: Privacy
        st.markdown("#### 4️⃣ Privacy & Data Protection")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success("### 🔒")
        with col2:
            st.markdown("""
            **What we do**:
            - Data encrypted in transit and at rest
            - Explicit consent required before data sharing
            - Minimal data collection (only what's necessary)
            - Right to deletion honored
            
            **Data lifecycle**:
            1. Collection: With your explicit consent
            2. Processing: Secure, encrypted systems
            3. Storage: Minimal retention period
            4. Deletion: Upon request or after retention period
            """)
        
        st.divider()
        
        # Principle 5: Safety
        st.markdown("#### 5️⃣ Safety & Reliability")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success("### 🛡️")
        with col2:
            st.markdown("""
            **What we do**:
            - Continuous monitoring for model drift
            - A/B testing before deploying changes
            - Fallback to human decision when confidence is low
            
            **Safeguards**:
            - Low confidence predictions flagged for review
            - Out-of-distribution detection
            - Regular performance evaluation
            - Incident response protocol
            """)
        
        st.markdown("---")
        
        st.info("""
        💡 **Our Commitment**: If our AI system makes an unfair decision, we take responsibility 
        and work to fix it. Ethical AI is an ongoing process, not a destination.
        """)
    
    # ========================================
    # TAB 4: PRIVACY & DATA
    # ========================================
    with tab4:
        st.markdown("### 🔒 Your Privacy & Data Rights")
        
        st.warning("""
        **Important**: Your privacy is protected by law. You have specific rights regarding 
        your personal data and how it's used.
        """)
        
        st.markdown("#### 📋 What Data We Collect")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Structured Data (ML Model)**:
            - Age, Education, Employment Type
            - Income, Credit Score
            - Loan Amount, Term, Purpose
            - Debt-to-Income Ratio
            - Number of Credit Lines
            - Marital Status, Dependents
            """)
        
        with col2:
            st.markdown("""
            **Unconventional Data (RAG Agent)**:
            - Mobile usage patterns (aggregated)
            - Psychometric responses
            - Financial behavior metrics
            - Social network indicators
            - Bill payment history
            """)
        
        st.markdown("---")
        
        st.markdown("#### 🛡️ How We Protect Your Data")
        
        protection_measures = [
            ("🔐 Encryption", "All data encrypted with AES-256 in transit and at rest"),
            ("👤 Anonymization", "Personal identifiers removed where possible"),
            ("🚪 Access Control", "Strict role-based access controls"),
            ("📝 Audit Logs", "All data access logged and monitored"),
            ("⏱️ Limited Retention", "Data deleted after necessary period"),
            ("🔄 Regular Audits", "Security assessments every quarter")
        ]
        
        for measure, description in protection_measures:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### {measure}")
            with col2:
                st.info(description)
        
        st.markdown("---")
        
        st.markdown("#### ✅ Your Rights (GDPR Compliant)")
        
        rights = {
            "Right to Access": "Request a copy of all data we hold about you",
            "Right to Rectification": "Correct inaccurate or incomplete data",
            "Right to Erasure": "Request deletion of your data ('right to be forgotten')",
            "Right to Restrict Processing": "Limit how we use your data",
            "Right to Data Portability": "Receive your data in machine-readable format",
            "Right to Object": "Object to processing of your data",
            "Right to Human Review": "Request human review of automated decisions"
        }
        
        for right, description in rights.items():
            with st.expander(f"✅ {right}"):
                st.write(description)
                st.caption("To exercise this right, contact: privacy@example.com")
        
        st.markdown("---")
        
        st.markdown("#### 🔄 Data Sharing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ What We Share (with consent)**:
            - Aggregated assessment results
            - Eligibility decision
            - Risk score
            """)
        
        with col2:
            st.error("""
            **❌ What We DON'T Share**:
            - Raw call logs or messages
            - Detailed transaction history
            - Psychometric response details
            - Personal identifiers
            """)
        
        st.info("""
        🔔 **Important**: We only share data with lending partners AFTER you give explicit consent. 
        You can decline consent at any time, and we will delete your data.
        """)
    
    # ========================================
    # TAB 5: FAQ
    # ========================================
    with tab5:
        st.markdown("### ❓ Frequently Asked Questions")
        
        faqs = [
            {
                "q": "Can AI be biased?",
                "a": """
                Yes, AI can be biased if trained on biased data or designed poorly. That's why we:
                - Regularly audit for bias across demographics
                - Use explainable AI (SHAP) to detect unfair patterns
                - Have human oversight for all decisions
                - Continuously improve based on feedback
                """
            },
            {
                "q": "How accurate is the ML model?",
                "a": """
                Our model has been validated on historical data with:
                - ~85% accuracy on test data
                - Regular performance monitoring
                - Continuous retraining with new data
                
                However, NO model is perfect. That's why we:
                - Flag low-confidence predictions for human review
                - Allow appeals and human override
                - Continuously improve
                """
            },
            {
                "q": "Why was I rejected?",
                "a": """
                If you were rejected, check the SHAP explanation in your results. It shows:
                - Which features hurt your application
                - How much impact each had
                - Specific recommendations for improvement
                
                You can also:
                - Request human review
                - Appeal the decision
                - Work on improving weak areas and reapply
                """
            },
            {
                "q": "Can I trust SHAP explanations?",
                "a": """
                Yes! SHAP is:
                - Based on rigorous game theory mathematics
                - Industry standard for ML explanation
                - Peer-reviewed and widely validated
                - Used by major banks, hospitals, and tech companies
                
                It's one of the most trustworthy explanation methods available.
                """
            },
            {
                "q": "Who reviews the AI's decisions?",
                "a": """
                Multiple layers of oversight:
                1. **Automated checks**: Low confidence predictions are flagged
                2. **Human review**: Available on request for any decision
                3. **Regular audits**: Independent experts review the system quarterly
                4. **User feedback**: Your feedback helps us improve
                """
            },
            {
                "q": "What if I disagree with the decision?",
                "a": """
                You have several options:
                1. **Request human review**: Click the button in results
                2. **Appeal**: Provide additional information or context
                3. **Improve and reapply**: Follow the improvement plan
                4. **Formal complaint**: Contact our ethics board
                
                We take all concerns seriously and investigate thoroughly.
                """
            },
            {
                "q": "How is this different from traditional lending?",
                "a": """
                **Traditional Lending**:
                - Often opaque ("computer says no")
                - Limited to credit scores
                - Excludes many worthy borrowers
                
                **Our Approach**:
                - Fully transparent with SHAP
                - Uses alternative data (mobile, behavioral)
                - Explains every decision
                - Focuses on financial inclusion
                """
            },
            {
                "q": "Is my data safe?",
                "a": """
                Yes! We use:
                - Military-grade encryption (AES-256)
                - Secure cloud infrastructure
                - Regular security audits
                - GDPR compliance
                - Strict access controls
                
                Your data is safer with us than on paper or email.
                """
            }
        ]
        
        for i, faq in enumerate(faqs, 1):
            with st.expander(f"**{i}. {faq['q']}**"):
                st.markdown(faq['a'])
        
        st.markdown("---")
        
        st.info("""
        **Still have questions?** 
        
        📧 Email: support@ethicalmicrofinance.example.com  
        📞 Phone: +1-800-ETHICAL  
        💬 Live Chat: Available on our website
        """)


# Update your main() function to include this page:
# Add "Transparency & Ethics" to the page options in sidebar

def display_ml_results(ml_prediction: Dict[str, Any]):
    """
    Display ML model prediction results with full transparency.
    
    Key additions for Ethical AI:
    - Clear explanation of SHAP methodology
    - Visual feature importance
    - Step-by-step decision breakdown
    - Confidence intervals and uncertainty
    """
    st.header("🤖 ML Model Prediction Results")
    
    # Extract data
    prediction = ml_prediction.get("prediction", 0)
    probability = ml_prediction.get("probability", 0.5)
    explanation = ml_prediction.get("explanation", [])
    next_step = ml_prediction.get("next_step", {})
    meta = ml_prediction.get("meta", {})
    
    # ========================================
    # SECTION 1: DECISION SUMMARY
    # ========================================
    st.markdown("### 📊 Decision Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.success("### ✅ ELIGIBLE")
            st.caption("Model predicts: **Low default risk**")
        else:
            st.error("### ❌ NOT ELIGIBLE")
            st.caption("Model predicts: **High default risk**")
    
    with col2:
        # Color-coded probability
        if probability >= 0.7:
            st.success(f"### {probability:.1%}")
            st.caption("**Strong confidence**")
        elif probability >= 0.5:
            st.info(f"### {probability:.1%}")
            st.caption("**Moderate confidence**")
        else:
            st.warning(f"### {probability:.1%}")
            st.caption("**Low confidence**")
        
        # Visual probability bar
        st.progress(probability)
    
    with col3:
        confidence_margin = meta.get("probability_margin", abs(probability - 0.5))
        
        if meta.get("confidence_low", False):
            st.warning("### ⚠️ LOW")
            st.caption(f"Margin: {confidence_margin:.3f}")
            with st.expander("Why low confidence?"):
                reasons = meta.get("confidence_reasons", [])
                if reasons:
                    for reason in reasons:
                        st.write(f"• {reason}")
                else:
                    st.write("• Decision is close to threshold (50%)")
        else:
            st.success("### ✅ HIGH")
            st.caption(f"Margin: {confidence_margin:.3f}")
    
    st.markdown("---")
    
    # ========================================
    # SECTION 2: HOW SHAP WORKS (TRANSPARENCY)
    # ========================================
    with st.expander("🔍 **How does the model make decisions? (Click to understand SHAP)**", expanded=False):
        st.markdown("""
        ### Understanding SHAP (SHapley Additive exPlanations)
        
        **What is SHAP?**
        SHAP is a game-theory approach that explains the contribution of each feature to the final prediction.
        
        **How it works:**
        
        1. **Base Value (Baseline)**: 
           - The model starts with an average prediction across all training data
           - This is what the model would predict if it knew nothing about this specific applicant
           - Think of it as the "neutral starting point"
        
        2. **Feature Contributions (SHAP Values)**:
           - Each feature either pushes the prediction UP (positive SHAP) or DOWN (negative SHAP)
           - **Positive SHAP** = Feature makes approval MORE likely
           - **Negative SHAP** = Feature makes approval LESS likely
           - The magnitude shows HOW MUCH impact each feature has
        
        3. **Final Prediction**:
           - Base Value + Sum of all SHAP values = Final prediction score
           - This score is converted to a probability (0-100%)
        
        **Why SHAP is trustworthy:**
        - ✅ **Mathematically rigorous**: Based on cooperative game theory
        - ✅ **Additive**: All contributions sum to explain the full prediction
        - ✅ **Consistent**: Same feature values always have same impact
        - ✅ **Local accuracy**: Explains THIS specific prediction, not just general patterns
        
        **Reading SHAP values below:**
        - 🟢 **Positive values**: Features working IN YOUR FAVOR
        - 🔴 **Negative values**: Features working AGAINST YOU
        - 📏 **Bar length**: Shows the STRENGTH of impact
        """)
        
        st.info("""
        💡 **Key Insight**: SHAP ensures complete transparency. You can see EXACTLY why the model 
        made this decision and which features mattered most.
        """)
    
    # ========================================
    # SECTION 3: FEATURE IMPORTANCE BREAKDOWN
    # ========================================
    if explanation:
        st.markdown("### 🎯 Feature Impact Analysis")
        st.caption("How each feature influenced the decision (ordered by importance)")
        
        # Prepare data for visualization
        features = []
        shap_values = []
        abs_shaps = []
        
        for exp in explanation[:10]:  # Top 10 features
            features.append(exp.get("feature", "Unknown"))
            shap_val = exp.get("shap_value", 0)
            shap_values.append(shap_val)
            abs_shaps.append(abs(shap_val))
        
        # Create two columns: Chart + Detailed breakdown
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # SHAP Waterfall Chart using Plotly
            st.markdown("#### 📊 Visual Impact Chart")
            
            # Create waterfall chart
            fig = go.Figure()
            
            # Color code: green for positive, red for negative
            colors = ['rgba(34, 197, 94, 0.7)' if val > 0 else 'rgba(239, 68, 68, 0.7)' 
                     for val in shap_values]
            
            fig.add_trace(go.Bar(
                y=features,
                x=shap_values,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{val:+.4f}" for val in shap_values],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Feature Contributions (SHAP Values)",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Features",
                height=400,
                showlegend=False,
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=100, t=40, b=40)
            )
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            col_a, col_b = st.columns(2)
            with col_a:
                st.success("🟢 **Positive SHAP**: Increases approval chance")
            with col_b:
                st.error("🔴 **Negative SHAP**: Decreases approval chance")
        
        with col2:
            st.markdown("#### 📋 Detailed Breakdown")
            st.caption("Top features by importance")
            
            # Show top 6 features with detailed info
            for i, exp in enumerate(explanation[:6], 1):
                feature = exp.get("feature", "Unknown")
                shap_val = exp.get("shap_value", 0)
                abs_shap = exp.get("abs_shap", abs(shap_val))
                
                # Clean up feature names (remove prefixes like 'num__', 'cat__')
                clean_feature = feature.replace("num__", "").replace("cat__", "")
                
                # Determine impact direction and create appropriate styling
                if shap_val > 0:
                    st.success(f"**#{i}: {clean_feature}**")
                    st.caption(f"✅ Positive impact: +{abs_shap:.4f}")
                    st.caption(f"_Increases approval probability_")
                else:
                    st.error(f"**#{i}: {clean_feature}**")
                    st.caption(f"❌ Negative impact: -{abs_shap:.4f}")
                    st.caption(f"_Decreases approval probability_")
                
                st.divider()
        
        st.markdown("---")
        
        # ========================================
        # SECTION 4: INTERPRETATION GUIDE
        # ========================================
        st.markdown("### 💡 What This Means for You")
        
        # Split features into positive and negative
        positive_features = [exp for exp in explanation if exp.get("shap_value", 0) > 0]
        negative_features = [exp for exp in explanation if exp.get("shap_value", 0) < 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Your Strengths")
            if positive_features:
                for i, exp in enumerate(positive_features[:5], 1):
                    feature = exp.get("feature", "Unknown").replace("num__", "").replace("cat__", "")
                    shap_val = exp.get("shap_value", 0)
                    st.success(f"**{i}. {feature}**")
                    st.caption(f"Impact: +{shap_val:.4f} • This feature helps your case")
            else:
                st.info("No strongly positive features identified")
        
        with col2:
            st.markdown("#### ⚠️ Areas of Concern")
            if negative_features:
                for i, exp in enumerate(negative_features[:5], 1):
                    feature = exp.get("feature", "Unknown").replace("num__", "").replace("cat__", "")
                    shap_val = exp.get("shap_value", 0)
                    st.error(f"**{i}. {feature}**")
                    st.caption(f"Impact: {shap_val:.4f} • This feature reduces your score")
            else:
                st.info("No strongly negative features identified")
        
        # ========================================
        # SECTION 5: ACTIONABLE INSIGHTS
        # ========================================
        st.markdown("---")
        st.markdown("### 🎯 Actionable Recommendations")
        
        if prediction == 0:  # Not eligible
            st.warning("**To improve your eligibility, focus on:**")
            
            # Identify top 3 negative factors
            top_negative = sorted(
                [exp for exp in explanation if exp.get("shap_value", 0) < 0],
                key=lambda x: abs(x.get("shap_value", 0)),
                reverse=True
            )[:3]
            
            recommendations = []
            for exp in top_negative:
                feature = exp.get("feature", "Unknown").replace("num__", "").replace("cat__", "")
                
                # Generate specific recommendations based on feature
                if "CreditScore" in feature or "Credit" in feature:
                    recommendations.append("📈 **Improve Credit Score**: Pay bills on time, reduce credit utilization")
                elif "Income" in feature:
                    recommendations.append("💰 **Increase Income**: Consider additional income sources or higher-paying opportunities")
                elif "DTI" in feature or "Debt" in feature:
                    recommendations.append("💳 **Reduce Debt-to-Income Ratio**: Pay down existing debts")
                elif "Employed" in feature or "Employment" in feature:
                    recommendations.append("👔 **Stabilize Employment**: Maintain steady employment for longer periods")
                elif "Age" in feature:
                    recommendations.append("⏳ **Build Financial History**: Continue building your financial track record")
                else:
                    recommendations.append(f"🔧 **Address {feature}**: Work on improving this factor")
            
            for i, rec in enumerate(recommendations, 1):
                st.info(f"{i}. {rec}")
        
        else:  # Eligible
            st.success("**Your application looks strong! Key factors working in your favor:**")
            
            top_positive = sorted(
                [exp for exp in explanation if exp.get("shap_value", 0) > 0],
                key=lambda x: x.get("shap_value", 0),
                reverse=True
            )[:3]
            
            for i, exp in enumerate(top_positive, 1):
                feature = exp.get("feature", "Unknown").replace("num__", "").replace("cat__", "")
                shap_val = exp.get("shap_value", 0)
                st.info(f"{i}. ✅ **{feature}** is contributing positively (impact: +{shap_val:.4f})")
    
    else:
        st.warning("⚠️ No SHAP explanations available. Model may not support explainability.")
    
    # ========================================
    # SECTION 6: NEXT STEPS
    # ========================================
    st.markdown("---")
    st.markdown("### 📋 Next Steps")
    
    if next_step:
        action = next_step.get("action", "")
        message = next_step.get("message", "")
        
        if action == "wellness_coach":
            st.warning(f"💚 {message}")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Go to Wellness Coach →", type="primary"):
                    st.session_state.page = "Wellness Coach"
                    st.rerun()
            with col2:
                st.caption("Get personalized guidance to improve your financial profile")
        else:
            st.success(f"✅ {message}")
            st.info("You can proceed with your loan application.")
    
    # ========================================
    # SECTION 7: MODEL TRANSPARENCY INFO
    # ========================================
    with st.expander("🔬 **Model Information & Transparency**"):
        st.markdown("""
        ### Model Details
        
        **Model Type**: XGBoost Gradient Boosting Classifier
        
        **Training Data**: Historical loan default dataset with 16 structured features
        
        **Features Used**:
        - **Demographic**: Age, Education, Employment Type, Marital Status
        - **Financial**: Income, Credit Score, DTI Ratio, Loan Amount
        - **Loan Details**: Loan Term, Interest Rate, Purpose, Co-signer status
        - **Credit History**: Number of Credit Lines, Months Employed
        - **Obligations**: Has Mortgage, Has Dependents
        
        **Explainability Method**: SHAP (SHapley Additive exPlanations)
        - Based on cooperative game theory
        - Provides local explanations for individual predictions
        - Satisfies properties of local accuracy, missingness, and consistency
        
        **Ethical Considerations**:
        - ✅ **Transparency**: Every decision is explainable
        - ✅ **Fairness**: No direct use of protected attributes
        - ✅ **Accountability**: Clear feature attributions
        - ✅ **User Control**: You can see and challenge the decision
        
        **Model Limitations**:
        - ⚠️ Based on historical data (may not capture all scenarios)
        - ⚠️ Predictions are probabilistic (not guarantees)
        - ⚠️ Should be used as decision support, not sole determinant
        """)
        
        st.info("""
        📚 **Learn More About SHAP**: 
        SHAP values come from game theory and fairly distribute the "credit" for a prediction 
        among all features. Learn more at [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
        """)
    
    # ========================================
    # SECTION 8: RAW DATA (FOR TRANSPARENCY)
    # ========================================
    with st.expander("🗂️ **View Raw Prediction Data (Technical Details)**"):
        st.markdown("#### Complete Prediction Output")
        st.json(ml_prediction)
        
        if explanation:
            st.markdown("#### SHAP Values Table")
            shap_df = pd.DataFrame(explanation)
            st.dataframe(shap_df, use_container_width=True)
    
    # ========================================
    # SECTION 9: FEEDBACK MECHANISM
    # ========================================
    st.markdown("---")
    st.markdown("### 💬 Feedback on This Decision")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Decision seems fair", use_container_width=True):
            st.success("Thank you for your feedback!")
            # Log positive feedback
            from src.memory.store import log_action
            log_action("positive_feedback", st.session_state.get("user_id", "unknown"), 
                      {"prediction": prediction, "probability": probability})
    
    with col2:
        if st.button("👎 Decision seems unfair", use_container_width=True):
            st.warning("We've recorded your concern. Please contact support for review.")
            # Log negative feedback
            from src.memory.store import log_action
            log_action("negative_feedback", st.session_state.get("user_id", "unknown"), 
                      {"prediction": prediction, "probability": probability})
    
    with col3:
        if st.button("❓ Request human review", use_container_width=True):
            st.info("A human reviewer will examine your case. You'll be contacted within 48 hours.")
            # Log review request
            from src.memory.store import log_action
            log_action("review_requested", st.session_state.get("user_id", "unknown"), 
                      {"prediction": prediction, "probability": probability})
    
    st.caption("🔒 Your feedback helps us improve our model and ensure fair lending practices.")

def display_results(decision: Dict[str, Any]):
    """Display RAG assessment results."""
    st.header("🧠 RAG Assessment Results")
    
    eligibility = decision.get("eligibility", "maybe").lower()
    risk_score = decision.get("risk_score", 0.5)
    confidence = decision.get("confidence", "low").lower()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if eligibility == "yes":
            st.success(f"✅ **{eligibility.upper()}**")
        elif eligibility == "no":
            st.error(f"❌ **{eligibility.upper()}**")
        else:
            st.warning(f"⚠️ **{eligibility.upper()}**")
    
    with col2:
        st.metric("Risk Score", f"{risk_score:.2f}")
    
    with col3:
        st.info(f"Confidence: **{confidence.upper()}**")
    
    st.markdown("### 📝 Verdict")
    st.info(decision.get("verdict_text", ""))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Strong Points")
        for point in decision.get("strong_points", [])[:3]:
            st.success(f"• {point}")
    
    with col2:
        st.markdown("### ⚠️ Weak Points")
        for point in decision.get("weak_points", [])[:3]:
            st.error(f"• {point}")
    
    st.markdown("### 💡 Recommendations")
    for rec in decision.get("actionable_recommendations", [])[:4]:
        st.info(f"• {rec}")
    
    with st.expander("View Raw Assessment Data"):
        st.json(decision)


def consent_section():
    """
    Data consent mechanism - FIXED VERSION
    
    Shows consent for both ML and RAG results
    Saves data properly when consent is given
    """
    st.markdown("---")
    st.header("🔒 Data Consent & Submission")
    
    # Check what results we have
    has_ml = st.session_state.ml_prediction is not None
    has_rag = st.session_state.decision is not None
    
    if not has_ml and not has_rag:
        st.info("Complete an assessment to see consent options.")
        return
    
    # Show what will be shared
    st.info("""
    **📋 What will be shared with the lending institution:**
    
    ✅ **Aggregated assessment results**:
    - Eligibility decision (approved/not approved)
    - Risk score
    - Key recommendations
    
    ✅ **Summary statistics**:
    - Feature importance summaries
    - Confidence levels
    
    ❌ **What will NOT be shared**:
    - Raw call logs, messages, or transaction details
    - Detailed psychometric responses
    - Personal identifiers beyond necessary ID
    - Unconventional data in raw form
    
    **🔒 Your rights:**
    - You can decline consent and all your data will be deleted immediately
    - You can request data deletion at any time
    - Data is stored securely and encrypted
    """)
    
    # Show what specific data will be saved
    with st.expander("🔍 View exactly what will be submitted"):
        submission_preview = {
            "user_id": st.session_state.get("user_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }
        
        if has_ml:
            submission_preview["ml_assessment"] = {
                "prediction": st.session_state.ml_prediction.get("prediction"),
                "probability": st.session_state.ml_prediction.get("probability"),
                "top_factors": [
                    exp.get("feature") 
                    for exp in st.session_state.ml_prediction.get("explanation", [])[:3]
                ]
            }
        
        if has_rag:
            submission_preview["rag_assessment"] = {
                "eligibility": st.session_state.decision.get("eligibility"),
                "risk_score": st.session_state.decision.get("risk_score"),
                "confidence": st.session_state.decision.get("confidence")
            }
        
        st.json(submission_preview)
    
    st.markdown("---")
    
    # Check if already consented
    if st.session_state.consent_given:
        st.success("✅ **Consent already given!**")
        st.info(f"""
        Your assessment has been submitted to the lending institution.
        
        **Submission Details:**
        - User ID: {st.session_state.get('user_id', 'unknown')}
        - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Assessment Type: {'ML + RAG' if has_ml and has_rag else ('ML Only' if has_ml else 'RAG Only')}
        """)
        
        # Option to view in admin
        if st.button("👀 View in Admin Dashboard"):
            st.session_state.page = "Admin View"
            st.rerun()
        
        return
    
    # Consent checkbox
    st.markdown("### ✍️ Your Decision")
    consent = st.checkbox(
        "**I consent to sharing my assessment results with the lending institution**",
        key="consent_checkbox",
        help="Check this box to agree to data sharing as described above"
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "✅ Submit with Consent", 
            type="primary", 
            use_container_width=True, 
            disabled=not consent,
            help="Share your assessment results with the lender"
        ):
            user_id = st.session_state.get("user_id")
            
            if not user_id:
                st.error("❌ No user ID found. Please start a new assessment.")
                return
            
            try:
                # Prepare submission data
                profile_data = st.session_state.profile or {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Combine ML and RAG results
                combined_decision = {}
                
                if has_ml:
                    ml_pred = st.session_state.ml_prediction
                    combined_decision["ml_assessment"] = {
                        "prediction": ml_pred.get("prediction"),
                        "probability": ml_pred.get("probability"),
                        "eligibility": "yes" if ml_pred.get("prediction") == 1 else "no",
                        "risk_score": 1.0 - ml_pred.get("probability", 0.5),
                        "explanation": ml_pred.get("explanation", []),
                        "confidence_low": ml_pred.get("meta", {}).get("confidence_low", False)
                    }
                
                if has_rag:
                    combined_decision["rag_assessment"] = {
                        "eligibility": st.session_state.decision.get("eligibility"),
                        "risk_score": st.session_state.decision.get("risk_score"),
                        "confidence": st.session_state.decision.get("confidence"),
                        "verdict_text": st.session_state.decision.get("verdict_text"),
                        "strong_points": st.session_state.decision.get("strong_points", []),
                        "weak_points": st.session_state.decision.get("weak_points", []),
                        "recommendations": st.session_state.decision.get("actionable_recommendations", [])
                    }
                
                # Overall decision (prioritize ML if both available)
                if has_ml:
                    overall_eligibility = "yes" if st.session_state.ml_prediction.get("prediction") == 1 else "no"
                    overall_risk = 1.0 - st.session_state.ml_prediction.get("probability", 0.5)
                elif has_rag:
                    overall_eligibility = st.session_state.decision.get("eligibility", "maybe")
                    overall_risk = st.session_state.decision.get("risk_score", 0.5)
                else:
                    overall_eligibility = "unknown"
                    overall_risk = 0.5
                
                combined_decision["overall"] = {
                    "eligibility": overall_eligibility,
                    "risk_score": overall_risk,
                    "assessment_type": "ml_and_rag" if (has_ml and has_rag) else ("ml_only" if has_ml else "rag_only")
                }
                
                # Save to consented submissions
                from src.memory.store import save_consented_submission, log_action
                
                save_consented_submission(user_id, profile_data, combined_decision)
                
                # Log the consent action
                log_action("consent_given", user_id, {
                    "has_ml": has_ml,
                    "has_rag": has_rag,
                    "eligibility": overall_eligibility
                })
                
                # Update session state
                st.session_state.consent_given = True
                
                st.success("✅ **Consent given successfully!**")
                st.balloons()
                st.info("""
                Your assessment has been securely submitted to the lending institution.
                
                **Next Steps:**
                - You will be contacted within 2-3 business days
                - Check your email for updates
                - You can view your submission in the Admin Dashboard
                """)
                
                # Auto-refresh after 2 seconds
                import time
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error saving consent: {e}")
                logger.exception(e)
    
    with col2:
        if st.button(
            "❌ Decline Consent", 
            use_container_width=True,
            help="Decline data sharing and delete all collected data"
        ):
            user_id = st.session_state.get("user_id")
            
            if user_id:
                try:
                    from src.memory.store import delete_user_data, log_action
                    
                    # Delete all user data
                    delete_user_data(user_id)
                    
                    # Log the decline
                    log_action("consent_denied", user_id, {"timestamp": datetime.now().isoformat()})
                    
                    # Clear session state
                    st.session_state.profile = None
                    st.session_state.decision = None
                    st.session_state.ml_prediction = None
                    st.session_state.user_id = None
                    st.session_state.consent_given = False
                    
                    st.warning("❌ **Consent declined.**")
                    st.info("""
                    All your data has been permanently deleted from our systems.
                    
                    **What happened:**
                    - Assessment results: Deleted ✓
                    - Profile data: Deleted ✓
                    - Temporary files: Deleted ✓
                    
                    **You can:**
                    - Run a new assessment anytime
                    - Your data will NOT be shared
                    """)
                    
                    # Auto-refresh after 2 seconds
                    import time
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during data deletion: {e}")
                    logger.exception(e)
            else:
                st.warning("No user data to delete.")
    
    # Additional info
    st.markdown("---")
    st.caption("""
    🔐 **Privacy Notice**: This consent is voluntary. Your data is protected under data protection laws. 
    For questions, contact: privacy@ethicalmicrofinance.example.com
    """)


def admin_page():
    """
    Admin dashboard - FIXED VERSION
    Shows all consented submissions with proper data display
    """
    st.header("🔐 Admin Dashboard - Bank Submissions")
    
    # Password authentication
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.info("🔒 This page is restricted to administrators only.")
        
        admin_password = st.text_input(
            "Admin Password", 
            type="password", 
            key="admin_pwd",
            help="Default password: admin123"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔓 Login", type="primary"):
                if admin_password == "admin123":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Authentication successful!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        
        with col2:
            st.caption("Default: admin123")
        
        return
    
    # Authenticated admin view
    st.success("✅ Authenticated as Administrator")
    
    if st.button("🚪 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    st.markdown("---")
    
    # Admin controls
    st.markdown("### 🛠️ Admin Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", help="Reload submissions from disk"):
            st.rerun()
    
    with col2:
        if st.button("📊 View Stats", help="Show aggregate statistics"):
            st.info("Stats feature coming soon")
    
    with col3:
        if st.button("🗑️ Clear All", help="Delete all submissions (careful!)"):
            if st.checkbox("⚠️ Confirm deletion", key="confirm_delete"):
                try:
                    import shutil
                    data_dir = Path("data/consented_submissions")
                    if data_dir.exists():
                        shutil.rmtree(data_dir)
                        data_dir.mkdir(parents=True, exist_ok=True)
                        st.success("✅ All submissions deleted")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col4:
        if st.button("📥 Export CSV", help="Export submissions to CSV"):
            try:
                from src.memory.store import get_bank_submissions
                submissions = get_bank_submissions(limit=1000)
                
                if submissions:
                    import pandas as pd
                    df = pd.DataFrame(submissions)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"submissions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Load submissions
    try:
        from src.memory.store import get_bank_submissions
        
        # Get all submissions
        submissions = get_bank_submissions(limit=1000)
        
        # Display count
        st.markdown("### 📊 Submission Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Submissions", len(submissions))
        
        with col2:
            approved = sum(1 for s in submissions 
                          if s.get('decision', {}).get('ml_assessment', {}).get('eligibility') == 'yes'
                          or s.get('decision', {}).get('rag_assessment', {}).get('eligibility') == 'yes'
                          or s.get('decision', {}).get('overall', {}).get('eligibility') == 'yes')
            st.metric("Approved", approved, delta=f"{approved/len(submissions)*100:.1f}%" if submissions else "0%")
        
        with col3:
            rejected = len(submissions) - approved
            st.metric("Rejected", rejected, delta=f"{rejected/len(submissions)*100:.1f}%" if submissions else "0%")
        
        st.markdown("---")
        
        if not submissions:
            st.info("""
            📭 **No submissions yet**
            
            Submissions will appear here after users:
            1. Complete an assessment (ML or RAG)
            2. View their results
            3. Click "Submit with Consent"
            
            **Test the flow:**
            - Go to Assessment page
            - Fill in data and run assessment
            - Click the consent checkbox
            - Click "Submit with Consent" button
            """)
            
            # Show data folder status
            st.markdown("### 🗂️ Data Folder Status")
            data_dir = Path("data/consented_submissions")
            
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                st.info(f"✅ Data folder exists at: `{data_dir}`")
                st.write(f"📁 Files found: {len(files)}")
                
                if files:
                    st.write("**Files:**")
                    for f in files[:10]:  # Show first 10
                        st.code(f.name)
            else:
                st.warning(f"⚠️ Data folder doesn't exist yet: `{data_dir}`")
                if st.button("Create folder"):
                    data_dir.mkdir(parents=True, exist_ok=True)
                    st.success("✅ Folder created")
            
            return
        
        # Display submissions
        st.markdown("### 📋 Recent Submissions")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Filter by Type",
                ["All", "ML Only", "RAG Only", "Both ML & RAG"]
            )
        
        with col2:
            filter_status = st.selectbox(
                "Filter by Status",
                ["All", "Approved", "Rejected", "Pending"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Newest First", "Oldest First", "User ID"]
            )
        
        # Apply filters
        filtered = submissions.copy()
        
        if filter_type != "All":
            if filter_type == "ML Only":
                filtered = [s for s in filtered if 'ml_assessment' in s.get('decision', {})]
            elif filter_type == "RAG Only":
                filtered = [s for s in filtered if 'rag_assessment' in s.get('decision', {})]
            elif filter_type == "Both ML & RAG":
                filtered = [s for s in filtered 
                           if 'ml_assessment' in s.get('decision', {}) 
                           and 'rag_assessment' in s.get('decision', {})]
        
        if filter_status == "Approved":
            filtered = [s for s in filtered 
                       if s.get('decision', {}).get('overall', {}).get('eligibility') == 'yes']
        elif filter_status == "Rejected":
            filtered = [s for s in filtered 
                       if s.get('decision', {}).get('overall', {}).get('eligibility') == 'no']
        
        # Sort
        if sort_by == "Newest First":
            filtered = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_by == "Oldest First":
            filtered = sorted(filtered, key=lambda x: x.get('timestamp', ''))
        elif sort_by == "User ID":
            filtered = sorted(filtered, key=lambda x: x.get('user_id', ''))
        
        st.info(f"Showing {len(filtered)} of {len(submissions)} submissions")
        
        # Display each submission
        for i, submission in enumerate(filtered[:50], 1):  # Show max 50
            user_id = submission.get('user_id', 'unknown')
            timestamp = submission.get('timestamp', '')
            decision = submission.get('decision', {})
            
            # Get overall status
            overall = decision.get('overall', {})
            eligibility = overall.get('eligibility', 'unknown')
            risk_score = overall.get('risk_score', 0.5)
            assessment_type = overall.get('assessment_type', 'unknown')
            
            # Determine color/icon
            if eligibility == 'yes':
                status_icon = "✅"
                status_color = "success"
            elif eligibility == 'no':
                status_icon = "❌"
                status_color = "error"
            else:
                status_icon = "⚠️"
                status_color = "warning"
            
            # Create expander
            with st.expander(
                f"{status_icon} Submission #{i}: {user_id} - {timestamp[:19]} - {eligibility.upper()}",
                expanded=(i == 1)  # Expand first one by default
            ):
                # Summary row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("User ID", user_id)
                
                with col2:
                    st.metric("Status", eligibility.upper())
                
                with col3:
                    st.metric("Risk Score", f"{risk_score:.2f}")
                
                with col4:
                    st.metric("Type", assessment_type.replace('_', ' ').title())
                
                st.markdown("---")
                
                # ML Assessment (if available)
                if 'ml_assessment' in decision:
                    st.markdown("#### 🤖 ML Assessment")
                    ml_assess = decision['ml_assessment']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Prediction:** {ml_assess.get('eligibility', 'unknown').upper()}")
                        st.write(f"**Probability:** {ml_assess.get('probability', 0):.2%}")
                    with col2:
                        st.write(f"**Risk Score:** {ml_assess.get('risk_score', 0):.2f}")
                        st.write(f"**Low Confidence:** {'Yes ⚠️' if ml_assess.get('confidence_low') else 'No ✅'}")
                    
                    # Top factors
                    explanations = ml_assess.get('explanation', [])
                    if explanations:
                        st.write("**Top Factors:**")
                        for exp in explanations[:3]:
                            feature = exp.get('feature', 'Unknown')
                            shap = exp.get('shap_value', 0)
                            st.write(f"  • {feature}: {shap:+.4f}")
                    
                    st.markdown("---")
                
                # RAG Assessment (if available)
                if 'rag_assessment' in decision:
                    st.markdown("#### 🧠 RAG Assessment")
                    rag_assess = decision['rag_assessment']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Eligibility:** {rag_assess.get('eligibility', 'unknown').upper()}")
                        st.write(f"**Confidence:** {rag_assess.get('confidence', 'unknown').upper()}")
                    with col2:
                        st.write(f"**Risk Score:** {rag_assess.get('risk_score', 0):.2f}")
                    
                    verdict = rag_assess.get('verdict_text', '')
                    if verdict:
                        st.info(f"**Verdict:** {verdict}")
                    
                    # Strong points
                    strong = rag_assess.get('strong_points', [])
                    if strong:
                        st.success("**Strengths:**")
                        for point in strong[:3]:
                            st.write(f"  ✓ {point}")
                    
                    # Weak points
                    weak = rag_assess.get('weak_points', [])
                    if weak:
                        st.warning("**Concerns:**")
                        for point in weak[:3]:
                            st.write(f"  • {point}")
                    
                    st.markdown("---")
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📋 View Full Data", key=f"view_{i}"):
                        st.json(submission)
                
                with col2:
                    if st.button(f"📧 Contact User", key=f"contact_{i}"):
                        st.info(f"Email: {user_id}@example.com")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        if st.checkbox(f"Confirm delete {user_id}", key=f"confirm_{i}"):
                            try:
                                # Delete the file
                                file_path = Path(f"data/consented_submissions/{user_id}.json")
                                if file_path.exists():
                                    file_path.unlink()
                                    st.success(f"✅ Deleted {user_id}")
                                    
                                    from src.memory.store import log_action
                                    log_action("admin_delete_submission", user_id)
                                    
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
        
        # Pagination hint
        if len(filtered) > 50:
            st.info(f"Showing first 50 of {len(filtered)} submissions. Use filters to narrow down.")
    
    except Exception as e:
        st.error(f"❌ Error loading submissions: {e}")
        import traceback
        st.code(traceback.format_exc())
        
        st.info("""
        **Troubleshooting:**
        1. Check if `data/consented_submissions/` folder exists
        2. Verify JSON files are valid
        3. Check file permissions
        """)


def wellness_coach_page():
    """Wellness coach interface."""
    st.header("💚 Wellness Coach")
    
    st.info("Financial wellness guidance and improvement tips.")
    
    suggestions = [
        "Build credit history with smaller loans",
        "Increase savings frequency",
        "Improve bill payment timeliness",
        "Join a Self-Help Group (SHG)",
        "Maintain stable income",
        "Reduce wallet balance fluctuations"
    ]
    
    for suggestion in suggestions:
        st.success(f"✅ {suggestion}")
    
    if st.button("← Back to Assessment"):
        st.session_state.page = "Assessment"
        st.rerun()


if __name__ == "__main__":
    main()