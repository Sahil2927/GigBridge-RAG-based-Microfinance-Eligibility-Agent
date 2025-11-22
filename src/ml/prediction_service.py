"""
Prediction Service - FIXED VERSION

Key changes:
- Uses the fixed feature_converter to extract ONLY ML features
- Clear separation: ML model gets structured features only
- Telemetry and consent flows preserved
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
import numpy as np

from src.ml.model_predictor import ModelPredictor
from src.ml.feature_converter import convert_to_ml_features, validate_ml_features
from src.ml.telemetry import log_feature_distribution
from src.memory.store import save_consented_submission, log_action

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_path: Optional[str] = None):
        self.predictor = ModelPredictor(model_path)
        self.recent_predictions: List[Dict[str, Any]] = []
        self.max_recent_predictions = 100

    def predict_from_raw_inputs(self, raw_inputs: Dict[str, Any], 
                                consent: bool = False, 
                                user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction from raw user inputs.
        
        IMPORTANT: raw_inputs should contain the structured ML features only:
        - LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, 
        - NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education,
        - EmploymentType, MaritalStatus, HasMortgage, HasDependents, 
        - LoanPurpose, HasCoSigner
        
        Do NOT include unconventional data (mobile_metadata, psychometrics, etc.)
        
        Args:
            raw_inputs: Dictionary with structured ML features only
            consent: Whether user has given consent
            user_id: User identifier
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if raw_inputs is None:
                raise ValueError("raw_inputs is None")

            if not self.predictor.is_loaded:
                if not self.predictor.load_model():
                    raise RuntimeError("Model load failed")

            # Convert raw inputs to ML feature DataFrame
            logger.info(f"Converting raw inputs with {len(raw_inputs)} fields to ML features")
            ml_features_df = convert_to_ml_features(raw_inputs)
            
            # Validate features
            is_valid, errors = validate_ml_features(ml_features_df)
            if not is_valid:
                error_msg = f"ML feature validation failed: {errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"ML features validated successfully: {list(ml_features_df.columns)}")

            # Make prediction
            preds, probs_pos, proba_matrix = self.predictor.predict(ml_features_df)

            # Generate explanation (SHAP)
            explanations = self.predictor.explain(ml_features_df)

            # Robustness checks
            robustness_list = self.predictor.check_robustness(
                ml_features_df, preds, probs_pos, self.recent_predictions
            )
            robustness = robustness_list[0] if isinstance(robustness_list, list) else {
                "confidence_low": False,
                "reasons": [],
                "probability_margin": abs(float(np.atleast_1d(probs_pos)[0]) - 0.5)
            }

            pred_val = int(np.atleast_1d(preds)[0])
            prob_val = float(np.atleast_1d(probs_pos)[0])

            # Update recent predictions
            self.recent_predictions.append({
                "prediction": pred_val,
                "probability": prob_val,
                "timestamp": datetime.now().isoformat()
            })
            if len(self.recent_predictions) > self.max_recent_predictions:
                self.recent_predictions.pop(0)

            # Format explanation into user-friendly list
            formatted_explanation = []
            if explanations:
                first_expl = explanations[0] if isinstance(explanations, list) else explanations
                if isinstance(first_expl, list):
                    for e in first_expl[:6]:
                        formatted_explanation.append({
                            "feature": e.get("feature", "Unknown"),
                            "shap_value": float(e.get("shap_value", 0.0)),
                            "abs_shap": abs(float(e.get("shap_value", 0.0)))
                        })

            # Determine next step
            if pred_val == 1:
                next_step = {
                    "action": "proceed_application",
                    "message": "Proceed with loan application"
                }
            else:
                next_step = {
                    "action": "wellness_coach",
                    "message": "Consider wellness coach to improve eligibility"
                }

            response = {
                "prediction": pred_val,
                "probability": prob_val,
                "explanation": formatted_explanation,
                "next_step": next_step,
                "meta": {
                    "confidence_low": robustness.get("confidence_low", False),
                    "confidence_reasons": robustness.get("reasons", []),
                    "probability_margin": robustness.get("probability_margin", abs(prob_val - 0.5)),
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Telemetry logging
            try:
                log_feature_distribution(ml_features_df, user_id or "unknown_user", consent)
            except Exception as e:
                logger.debug(f"Telemetry logging failed: {e}")

            # Save consented submission
            if consent and user_id:
                try:
                    save_consented_submission(
                        user_id,
                        {
                            "user_id": user_id,
                            "timestamp": datetime.now().isoformat(),
                            "ml_features": raw_inputs  # Store the structured features used
                        },
                        {
                            "eligibility": "yes" if pred_val == 1 else "no",
                            "risk_score": 1.0 - prob_val,
                            "prediction": pred_val,
                            "probability": prob_val,
                            "explanation": formatted_explanation
                        }
                    )
                    log_action("prediction_with_consent", user_id, {"prediction": pred_val})
                except Exception as e:
                    logger.error(f"Error saving consented submission: {e}")
            else:
                log_action("prediction_no_consent", user_id or "unknown", {"prediction": pred_val})

            return response

        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            raise

    def predict_with_detailed_explanation(self, raw_inputs: Dict[str, Any], 
                                      consent: bool = False, 
                                      user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced prediction with detailed SHAP explanations and user-friendly interpretations.
        
        This method extends predict_from_raw_inputs with additional transparency features.
        """
        # Get base prediction
        base_prediction = self.predict_from_raw_inputs(raw_inputs, consent, user_id)
        
        # Import the SHAP utilities
        try:
            from src.ml.shap_explainer import (
                interpret_shap_value, 
                generate_improvement_plan,
                calculate_shap_statistics,
                format_shap_for_user,
                get_shap_educational_content
            )
            
            # Enhance explanations with interpretations
            enhanced_explanations = []
            for exp in base_prediction.get("explanation", []):
                feature = exp.get("feature", "Unknown")
                shap_value = exp.get("shap_value", 0)
                
                # Get detailed interpretation
                interpretation = interpret_shap_value(feature, shap_value)
                
                # Enhance the explanation dictionary
                enhanced_exp = {
                    **exp,
                    "magnitude": interpretation.get("magnitude", "unknown"),
                    "direction": interpretation.get("direction", "unknown"),
                    "emoji": interpretation.get("emoji", "📊"),
                    "summary": interpretation.get("summary", ""),
                    "detail": interpretation.get("detail", ""),
                    "insight": interpretation.get("insight", ""),
                    "action": interpretation.get("action", ""),
                    "user_friendly": format_shap_for_user(feature, shap_value)
                }
                enhanced_explanations.append(enhanced_exp)
            
            # Calculate statistics
            stats = calculate_shap_statistics(base_prediction.get("explanation", []))
            
            # Generate improvement plan
            improvement_plan = generate_improvement_plan(
                base_prediction.get("explanation", []),
                base_prediction.get("prediction", 0)
            )
            
            # Get educational content
            educational_content = get_shap_educational_content()
            
            # Enhance the response
            enhanced_response = {
                **base_prediction,
                "explanation": enhanced_explanations,
                "shap_statistics": stats,
                "improvement_plan": improvement_plan,
                "educational_content": educational_content,
                "transparency": {
                    "explanation_method": "SHAP (SHapley Additive exPlanations)",
                    "explanation_type": "Local (instance-specific)",
                    "model_type": "XGBoost Gradient Boosting Classifier",
                    "features_used": len(enhanced_explanations),
                    "explainability_score": "High",
                    "audit_trail": {
                        "timestamp": base_prediction.get("meta", {}).get("timestamp", ""),
                        "model_version": "v1.0",
                        "explanation_generated": True,
                        "user_consent": consent
                    }
                }
            }
            
            return enhanced_response
            
        except ImportError:
            logger.warning("SHAP explainer utilities not available. Returning base prediction.")
            return base_prediction
        except Exception as e:
            logger.error(f"Error generating enhanced explanations: {e}")
            return base_prediction

    def predict_from_profile(self, profile: Dict[str, Any], 
                            consent: bool = False,
                            user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction from a full user profile (with unconventional data).
        
        This extracts only the ML features from the profile and makes prediction.
        Unconventional data is ignored for ML prediction.
        
        Args:
            profile: Full user profile (may include unconventional data)
            consent: Whether user has given consent
            user_id: User identifier
            
        Returns:
            Dictionary with prediction results
        """
        # Extract ML features from profile
        from src.ml.feature_converter import extract_ml_features_only
        
        ml_features = extract_ml_features_only(profile)
        
        # Make prediction using extracted features
        return self.predict_from_raw_inputs(ml_features, consent, user_id)