"""
RAG Agent Module

This module implements the RAG-based decision agent that:
- Retrieves relevant research chunks
- Builds a structured prompt for Groq LLM
- Parses JSON output from LLM
- Returns structured eligibility decision
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
from src.kb.retriever import get_retriever
from src.utils.validators import extract_json_from_text, validate_rag_output

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG-based eligibility decision agent using Groq LLM.
    """
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 model_name: str = "mixtral-8x7b-32768",
                 index_path: str = "./data/faiss_index.bin",
                 metadata_path: str = "./data/metadata.jsonl"):
        """
        Initialize the RAG agent.
        
        Args:
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            model_name: Groq model name (mixtral-8x7b-32768, llama3-70b-8192, etc.)
            index_path: Path to FAISS index
            metadata_path: Path to metadata file
        """
        # Get API key from env or parameter
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Set it as environment variable or pass as parameter.")
        
        self.model_name = model_name
        self.client = Groq(api_key=self.groq_api_key)
        
        # Initialize retriever
        self.retriever = get_retriever(index_path, metadata_path)
        
        logger.info(f"RAG Agent initialized with model: {model_name}")
    
    def build_prompt(self, profile: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for the LLM based on user profile and retrieved chunks.
        
        Args:
            profile: User profile dictionary
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted prompt string
        """
        # Format retrieved chunks (exclude internal metadata from user-facing text)
        chunks_text = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunks_text += f"\n[CHUNK {i}]\n{chunk['text']}\n"
        
        # Format profile summary (key information only)
        profile_summary = self._format_profile_summary(profile)
        
        prompt = f"""You are an expert microfinance eligibility assessment agent. Your task is to analyze a user profile against research evidence and provide a structured JSON decision.

RESEARCH EVIDENCE:
{chunks_text}

USER PROFILE:
{profile_summary}

INSTRUCTIONS:
1. Analyze the user profile against the research evidence provided above.
2. Determine eligibility (yes/no/maybe), risk score (0.0-1.0, where 0.0 is lowest risk), and confidence level.
3. Provide exactly 3 strong points and 3 weak points based on the evidence.
4. List specific required_unconventional_data items from this list if data is missing:
   ["mobile_call_logs_30d", "airtime_topups_30d", "psychometric_responses", "savings_history_90d", "wallet_balance_timeseries_90d", "shg_membership_info", "transaction_history_180d", "sms_patterns_30d"]
5. Provide 4 actionable recommendations prioritized by impact.
6. If the research does not contain relevant information, set eligibility="maybe", risk_score=0.5, verdict_text="The research does not contain sufficient information for this assessment.", confidence="low".

OUTPUT FORMAT (return ONLY valid JSON, no additional text):
{{
  "eligibility": "yes" | "no" | "maybe",
  "risk_score": 0.0-1.0,
  "verdict_text": "Brief explanation of the decision (2-3 sentences). Do NOT mention paper titles or chunk IDs.",
  "strong_points": ["point 1", "point 2", "point 3"],
  "weak_points": ["point 1", "point 2", "point 3"],
  "required_unconventional_data": ["item1", "item2"] or [],
  "actionable_recommendations": ["recommendation 1", "recommendation 2", "recommendation 3", "recommendation 4"],
  "confidence": "high" | "medium" | "low",
  "raw_internal_reasoning": "Internal reasoning with CHUNK references (for audit only, not shown to user)"
}}

Return ONLY the JSON object, no markdown, no code blocks, no explanations before or after."""

        return prompt
    
    def _format_profile_summary(self, profile: Dict[str, Any]) -> str:
        """
        Format user profile into a readable summary for the prompt.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            Formatted profile summary string
        """
        lines = []
        
        # Demographics
        if "demographics" in profile:
            demo = profile["demographics"]
            lines.append(f"Demographics: Age {demo.get('age', 'N/A')}, {demo.get('gender', 'N/A')}, {demo.get('occupation', 'N/A')}, Monthly Income: {demo.get('monthly_income', 'N/A')}")
        
        # Mobile metadata
        if "mobile_metadata" in profile:
            mobile = profile["mobile_metadata"]
            lines.append(f"Mobile: Avg daily calls: {mobile.get('avg_daily_calls', 'NA')}, Unique contacts (30d): {mobile.get('unique_contacts_30d', 'NA')}, Airtime topup frequency: {mobile.get('airtime_topup_frequency', 'NA')}")
        
        # Psychometrics
        if "psychometrics" in profile:
            psych = profile["psychometrics"]
            if "conscientiousness_score" in psych:
                lines.append(f"Psychometrics: Conscientiousness score: {psych.get('conscientiousness_score', 'NA')}")
            else:
                lines.append(f"Psychometrics: Individual responses available")
        
        # Financial behavior
        if "financial_behavior" in profile:
            fin = profile["financial_behavior"]
            lines.append(f"Financial: Savings frequency: {fin.get('savings_frequency', 'NA')}, Bill payment timeliness: {fin.get('bill_payment_timeliness', 'NA')}")
        
        # Social network
        if "social_network" in profile:
            social = profile["social_network"]
            lines.append(f"Social: SHG membership: {social.get('shg_membership', 'NA')}, Peer monitoring: {social.get('peer_monitoring_strength', 'NA')}")
        
        # Loan history
        if "loan_history" in profile:
            loan = profile["loan_history"]
            lines.append(f"Loan History: Previous loans: {loan.get('previous_loans', 'NA')}, Defaults: {loan.get('previous_defaults', 'NA')}, Late payments: {loan.get('previous_late_payments', 'NA')}")
        
        return "\n".join(lines)
    
    def assess_eligibility(self, profile: Dict[str, Any], query: Optional[str] = None, k: int = 6) -> Dict[str, Any]:
        """
        Assess user eligibility using RAG.
        
        Args:
            profile: User profile dictionary
            query: Optional custom query (defaults to profile-based query)
            k: Number of chunks to retrieve
            
        Returns:
            Structured decision dictionary
        """
        # Build query from profile if not provided
        if query is None:
            query = self._build_query_from_profile(profile)
        
        logger.info(f"Retrieving chunks for query: {query[:100]}...")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_chunks(query, k=k)
        
        if not retrieved_chunks:
            logger.warning("No chunks retrieved. Returning default response.")
            return {
                "eligibility": "maybe",
                "risk_score": 0.5,
                "verdict_text": "The research does not contain sufficient information for this assessment.",
                "strong_points": [],
                "weak_points": ["Insufficient research evidence available"],
                "required_unconventional_data": [],
                "actionable_recommendations": ["Gather more research data"],
                "confidence": "low",
                "raw_internal_reasoning": "No chunks retrieved from knowledge base"
            }
        
        # Build prompt
        prompt = self.build_prompt(profile, retrieved_chunks)
        
        # Call Groq LLM
        logger.info(f"Calling Groq LLM ({self.model_name})...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise JSON generator. Always return valid JSON only, no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info("Received response from Groq LLM")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error calling Groq API: {e}")
            
            # Provide helpful error messages
            if "Invalid API Key" in error_msg or "invalid_api_key" in error_msg or "401" in error_msg:
                raise ValueError(
                    "Invalid Groq API Key. Please check your .env file:\n"
                    "1. Make sure GROQ_API_KEY is set to your actual API key (not the placeholder)\n"
                    "2. Get your API key from: https://console.groq.com/\n"
                    "3. The key should start with 'gsk_' and be about 50+ characters long"
                ) from e
            raise
        
        # Extract JSON from response
        decision = extract_json_from_text(response_text)
        
        if decision is None:
            logger.warning("Failed to extract JSON from LLM response. Using fallback.")
            decision = {
                "eligibility": "maybe",
                "risk_score": 0.5,
                "verdict_text": "Error parsing LLM response. Please try again.",
                "strong_points": [],
                "weak_points": ["LLM response parsing failed"],
                "required_unconventional_data": [],
                "actionable_recommendations": ["Retry assessment"],
                "confidence": "low",
                "raw_internal_reasoning": f"Failed to parse: {response_text[:200]}"
            }
        
        # Validate output
        is_valid, errors = validate_rag_output(decision)
        if not is_valid:
            logger.warning(f"RAG output validation errors: {errors}")
            # Fix common issues
            if "eligibility" not in decision:
                decision["eligibility"] = "maybe"
            if "risk_score" not in decision:
                decision["risk_score"] = 0.5
            if "confidence" not in decision:
                decision["confidence"] = "low"
        
        return decision
    
    def _build_query_from_profile(self, profile: Dict[str, Any]) -> str:
        """
        Build a search query from user profile.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            Query string
        """
        query_parts = []
        
        # Add relevant aspects
        if "mobile_metadata" in profile and any(
            v != "NA" and v is not None 
            for v in profile["mobile_metadata"].values()
        ):
            query_parts.append("mobile metadata airtime topup loan repayment")
        
        if "psychometrics" in profile and any(
            v != "NA" and v is not None 
            for v in profile["psychometrics"].values()
        ):
            query_parts.append("psychometric predictors loan repayment creditworthiness")
        
        if "financial_behavior" in profile and any(
            v != "NA" and v is not None 
            for v in profile["financial_behavior"].values()
        ):
            query_parts.append("savings behavior financial habits microfinance eligibility")
        
        if "social_network" in profile and profile["social_network"].get("shg_membership") == True:
            query_parts.append("self-help group SHG social capital loan repayment")
        
        if "loan_history" in profile and profile["loan_history"].get("previous_loans", 0) > 0:
            query_parts.append("loan history defaults repayment microfinance")
        
        # Default query if nothing specific
        if not query_parts:
            query_parts.append("microfinance eligibility unconventional data credit assessment")
        
        return " ".join(query_parts)


def assess_eligibility(profile: Dict[str, Any], 
                      groq_api_key: Optional[str] = None,
                      model_name: str = "mixtral-8x7b-32768",
                      query: Optional[str] = None,
                      k: int = 6) -> Dict[str, Any]:
    """
    Convenience function to assess eligibility.
    
    Args:
        profile: User profile dictionary
        groq_api_key: Groq API key (or set GROQ_API_KEY env var)
        model_name: Groq model name
        query: Optional custom query
        k: Number of chunks to retrieve
        
    Returns:
        Structured decision dictionary
    """
    agent = RAGAgent(groq_api_key=groq_api_key, model_name=model_name)
    return agent.assess_eligibility(profile, query=query, k=k)

