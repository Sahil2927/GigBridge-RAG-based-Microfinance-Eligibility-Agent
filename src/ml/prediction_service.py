"""
Prediction Service Module

This module provides a high-level service for making loan predictions with
SHAP explanations, consent handling, and robustness checks.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from src.ml.model_predictor import ModelPredictor
from src.ml.feature_converter import convert_raw_inputs_to_features, convert_profile_to_model_features
from src.ml.telemetry import log_feature_distribution
from src.memory.store import save_consented_submission, log_action

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for handling loan predictions with explanations and consent.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the prediction service.
        
        Args:
            model_path: Path to model file
        """
        self.predictor = ModelPredictor(model_path)
        self.recent_predictions = []  # For robustness checks
        self.max_recent_predictions = 100
    
    def predict_from_raw_inputs(self, raw_inputs: Dict[str, Any], 
                                consent: bool = False,
                                user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a prediction from raw user inputs.
        
        Args:
            raw_inputs: Dictionary with raw input values from UI
            consent: Whether user consented to data storage
            user_id: Optional user identifier
            
        Returns:
            Dictionary with prediction, probability, explanation, and metadata
        """
        try:
            # Convert raw inputs to features
            features_df = convert_raw_inputs_to_features(raw_inputs)
            
            # Make prediction
            prediction, probability, proba_array = self.predictor.predict(features_df)
            
            # Generate SHAP explanation
            explanations = self.predictor.explain(features_df)
            
            # Check robustness
            robustness = self.predictor.check_robustness(
                features_df, prediction, probability, self.recent_predictions
            )
            
            # Store recent prediction for robustness checks
            self.recent_predictions.append({
                "prediction": prediction,
                "probability": probability,
                "timestamp": datetime.now().isoformat()
            })
            if len(self.recent_predictions) > self.max_recent_predictions:
                self.recent_predictions.pop(0)
            
            # Format explanation
            formatted_explanation = self._format_explanation(explanations, prediction)
            
            # Determine next step
            next_step = self._get_next_step(prediction, probability)
            
            # Build response
            response = {
                "prediction": int(prediction),
                "probability": float(probability),
                "explanation": formatted_explanation,
                "next_step": next_step,
                "meta": {
                    "confidence_low": robustness["confidence_low"],
                    "confidence_reasons": robustness["reasons"],
                    "probability_margin": robustness["probability_margin"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Log telemetry (only if consented)
            log_feature_distribution(features_df, user_id, consent)
            
            # Save with consent
            if consent and user_id:
                try:
                    # Build profile for storage
                    profile = {
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "demographics": raw_inputs.get("demographics", {}),
                        "mobile_metadata": {},
                        "psychometrics": {},
                        "financial_behavior": {},
                        "social_network": {},
                        "loan_history": {}
                    }
                    
                    # Save submission
                    save_consented_submission(
                        user_id,
                        profile,
                        {
                            "eligibility": "yes" if prediction == 1 else "no",
                            "risk_score": 1.0 - probability,  # Convert to risk score
                            "prediction": prediction,
                            "probability": probability,
                            "explanation": formatted_explanation
                        }
                    )
                    log_action("prediction_with_consent", user_id, {"prediction": prediction})
                except Exception as e:
                    logger.error(f"Error saving consented submission: {e}")
            else:
                # Log without saving
                log_action("prediction_no_consent", user_id, {"prediction": prediction})
            
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def predict_from_profile(self, profile: Dict[str, Any],
                            consent: bool = False) -> Dict[str, Any]:
        """
        Make a prediction from a user profile.
        
        Args:
            profile: User profile dictionary
            consent: Whether user consented to data storage
            
        Returns:
            Dictionary with prediction, probability, explanation, and metadata
        """
        try:
            # Convert profile to features
            features_df = convert_profile_to_model_features(profile)
            
            # Make prediction
            prediction, probability, proba_array = self.predictor.predict(features_df)
            
            # Generate SHAP explanation
            explanations = self.predictor.explain(features_df)
            
            # Check robustness
            robustness = self.predictor.check_robustness(
                features_df, prediction, probability, self.recent_predictions
            )
            
            # Store recent prediction
            self.recent_predictions.append({
                "prediction": prediction,
                "probability": probability,
                "timestamp": datetime.now().isoformat()
            })
            if len(self.recent_predictions) > self.max_recent_predictions:
                self.recent_predictions.pop(0)
            
            # Format explanation
            formatted_explanation = self._format_explanation(explanations, prediction)
            
            # Determine next step
            next_step = self._get_next_step(prediction, probability)
            
            # Build response
            response = {
                "prediction": int(prediction),
                "probability": float(probability),
                "explanation": formatted_explanation,
                "next_step": next_step,
                "meta": {
                    "confidence_low": robustness["confidence_low"],
                    "confidence_reasons": robustness["reasons"],
                    "probability_margin": robustness["probability_margin"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Log telemetry (only if consented)
            user_id = profile.get("user_id", f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            log_feature_distribution(features_df, user_id, consent)
            
            # Save with consent
            if consent:
                try:
                    save_consented_submission(
                        user_id,
                        profile,
                        {
                            "eligibility": "yes" if prediction == 1 else "no",
                            "risk_score": 1.0 - probability,
                            "prediction": prediction,
                            "probability": probability,
                            "explanation": formatted_explanation
                        }
                    )
                    log_action("prediction_with_consent", user_id, {"prediction": prediction})
                except Exception as e:
                    logger.error(f"Error saving consented submission: {e}")
            else:
                log_action("prediction_no_consent", user_id, {"prediction": prediction})
            
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def _format_explanation(self, explanations: Optional[List[Dict[str, Any]]], 
                           prediction: int) -> List[Dict[str, Any]]:
        """
        Format SHAP explanations into user-friendly format.
        
        Args:
            explanations: Raw SHAP explanations
            prediction: Model prediction (0 or 1)
            
        Returns:
            List of formatted explanations
        """
        if not explanations:
            return []
        
        # Get top 5-6 features
        top_features = explanations[:6]
        
        # Filter by sign based on prediction
        if prediction == 1:
            # For eligible: show positive contributors (strong points)
            filtered = [f for f in top_features if f["shap_value"] > 0]
            if not filtered:
                filtered = top_features[:5]
        else:
            # For not eligible: show negative contributors (weak points)
            filtered = [f for f in top_features if f["shap_value"] < 0]
            if not filtered:
                filtered = top_features[:5]
        
        # Format with user-friendly notes
        formatted = []
        for feat in filtered[:6]:
            feature_name = feat["feature"]
            shap_val = feat["shap_value"]
            
            # Generate user-friendly note
            note = self._generate_feature_note(feature_name, shap_val, prediction)
            
            formatted.append({
                "feature": feature_name,
                "shap_value": round(shap_val, 4),
                "note": note
            })
        
        return formatted
    
    def _generate_feature_note(self, feature_name: str, shap_value: float, 
                               prediction: int) -> str:
        """
        Generate a user-friendly note for a feature.
        
        Args:
            feature_name: Name of the feature
            shap_value: SHAP value
            prediction: Model prediction
            
        Returns:
            User-friendly note string
        """
        # Map feature names to user-friendly descriptions
        feature_map = {
            "Age": "Age",
            "Gender": "Gender",
            "MonthlyIncome": "Monthly Income",
            "PreviousLoans": "Previous Loans",
            "PreviousDefaults": "Previous Defaults",
            "ConscientiousnessScore": "Conscientiousness",
            "SHGMembership": "SHG Membership",
            "SavingsFrequency": "Savings Frequency",
            "BillPaymentTimeliness": "Bill Payment Timeliness"
        }
        
        friendly_name = feature_map.get(feature_name, feature_name)
        
        if prediction == 1:
            if shap_value > 0:
                return f"Strong: {friendly_name} contributes positively to eligibility"
            else:
                return f"Note: {friendly_name} has some negative impact"
        else:
            if shap_value < 0:
                return f"Weak: {friendly_name} reduces eligibility"
            else:
                return f"Note: {friendly_name} has some positive impact"
    
    def _get_next_step(self, prediction: int, probability: float) -> Dict[str, Any]:
        """
        Determine next step for user based on prediction.
        
        Args:
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            
        Returns:
            Dictionary with next step information
        """
        if prediction == 0:
            # Not eligible - route to wellness coach
            return {
                "action": "wellness_coach",
                "message": "Based on your profile, we recommend connecting with our microfinance wellness coach to explore options for improving your eligibility.",
                "redirect_url": "/wellness-coach"
            }
        else:
            # Eligible - proceed with loan application
            return {
                "action": "proceed_application",
                "message": "You appear eligible for a loan. Proceed with the application process.",
                "redirect_url": "/application"
            }

