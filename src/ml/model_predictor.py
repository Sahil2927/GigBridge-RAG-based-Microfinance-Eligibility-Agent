"""
Model Predictor - FIXED VERSION

Key changes:
- Strictly enforces that input has EXACTLY the features the model was trained on
- No dynamic feature extraction - uses predefined feature list
- Clear error messages when feature mismatch occurs
"""

import logging
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


# The EXACT features the model expects (must match training data)
EXPECTED_ML_FEATURES = [
    "LoanID",
    "Age",
    "Income", 
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]


class ModelPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/loan_xgb.pkl"
        self.model = None
        self.explainer = None
        self.is_loaded = False
        self.expected_features = EXPECTED_ML_FEATURES

    def load_model(self) -> bool:
        """Load the trained model."""
        p = Path(self.model_path)
        if not p.exists():
            logger.warning(f"Model file not found at {self.model_path}")
            return False
        
        try:
            self.model = joblib.load(p)
            logger.info(f"Loaded model from {self.model_path}")
            logger.info(f"Model expects {len(self.expected_features)} features: {self.expected_features}")
            
            # Initialize SHAP explainer if available
            if SHAP_AVAILABLE:
                try:
                    # Get the final estimator from pipeline
                    if hasattr(self.model, "named_steps"):
                        # Find the estimator (usually last step)
                        estimator = None
                        for step_name in reversed(list(self.model.named_steps.keys())):
                            step = self.model.named_steps[step_name]
                            if hasattr(step, 'predict_proba'):
                                estimator = step
                                break
                        if estimator is not None:
                            self.explainer = shap.TreeExplainer(estimator)
                    else:
                        self.explainer = shap.TreeExplainer(self.model)
                    logger.info("SHAP explainer initialized successfully")
                except Exception as e:
                    logger.warning(f"Could not initialize SHAP explainer: {e}")
                    self.explainer = None
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def _to_dataframe(self, features: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features.copy()
        if isinstance(features, dict):
            return pd.DataFrame([features])
        if isinstance(features, list):
            if all(isinstance(x, dict) for x in features):
                return pd.DataFrame(features)
            else:
                return pd.DataFrame([features])
        raise ValueError("Unsupported features type. Use dict, list of dicts, or DataFrame.")

    def _validate_and_prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that DataFrame has exactly the expected features and prepare for model.
        
        Raises:
            ValueError: If features don't match expected features
        """
        # Check if we have the right columns
        missing_cols = set(self.expected_features) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_features)
        
        if missing_cols:
            raise ValueError(
                f"Missing required features: {missing_cols}. "
                f"Expected features: {self.expected_features}"
            )
        
        if extra_cols:
            logger.warning(f"Removing extra columns not expected by model: {extra_cols}")
            # Remove extra columns
            df = df[self.expected_features]
        
        # Ensure column order matches expected order
        df = df[self.expected_features]
        
        logger.info(f"Feature validation passed. Input has {len(df.columns)} columns matching model expectations.")
        
        return df

    def predict(self, features: Union[Dict, List, pd.DataFrame], threshold: float = 0.5):
        """
        Predict loan default probability.
        
        Args:
            features: Input features (must contain exactly the expected ML features)
            threshold: Classification threshold
            
        Returns:
            (predictions, positive_probabilities, full_probability_matrix)
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Model not loaded")

        # Convert to DataFrame
        df = self._to_dataframe(features)
        
        # Validate and prepare features
        try:
            df_prepared = self._validate_and_prepare_features(df)
        except ValueError as e:
            logger.error(f"Feature validation failed: {e}")
            raise

        # Make predictions
        try:
            proba = self.model.predict_proba(df_prepared)
            
            # Handle binary vs multiclass
            if proba.ndim == 1:
                pos = proba
                preds = (pos >= threshold).astype(int)
                raw = np.vstack([1 - pos, pos]).T
                return preds, pos, raw

            pos = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            preds = (pos >= threshold).astype(int)
            return preds, pos, proba
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(f"Input shape: {df_prepared.shape}, columns: {list(df_prepared.columns)}")
            raise

    def explain(self, features: Union[Dict, List, pd.DataFrame], top_k: int = 10):
        """
        Generate SHAP explanations for predictions.
        
        Returns:
            List of explanation dictionaries (one per instance)
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
                
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP not available or explainer not initialized.")
            return None

        df = self._to_dataframe(features)
        
        try:
            df_prepared = self._validate_and_prepare_features(df)
        except ValueError as e:
            logger.error(f"Feature validation failed for explanation: {e}")
            return None

        try:
            # Get transformed features for SHAP (after preprocessing)
            if hasattr(self.model, "named_steps"):
                # Get preprocessor
                preprocess = None
                for candidate in ("preprocess", "preprocessor", "transformer"):
                    preprocess = self.model.named_steps.get(candidate)
                    if preprocess is not None:
                        break
                
                if preprocess is not None:
                    # Transform the data through preprocessing
                    X_transformed = preprocess.transform(df_prepared)
                    
                    # Get feature names after transformation
                    if hasattr(preprocess, 'get_feature_names_out'):
                        try:
                            feature_names = list(preprocess.get_feature_names_out())
                        except:
                            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                    else:
                        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                    
                    # Calculate SHAP values
                    shap_values = self.explainer.shap_values(X_transformed)
                    
                    # Handle different SHAP output formats
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification, take positive class
                    
                    # Format explanations
                    explanations = []
                    for i in range(len(df_prepared)):
                        row_shap = shap_values[i] if shap_values.ndim > 1 else shap_values
                        
                        expl_list = [
                            {
                                "feature": fname,
                                "shap_value": float(sval),
                                "abs_shap": abs(float(sval))
                            }
                            for fname, sval in zip(feature_names, row_shap)
                        ]
                        
                        # Sort by absolute SHAP value
                        expl_list = sorted(expl_list, key=lambda x: x["abs_shap"], reverse=True)[:top_k]
                        explanations.append(expl_list)
                    
                    return explanations
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None

    def check_robustness(self, features: Union[Dict, List, pd.DataFrame], 
                         preds, probs_pos, 
                         recent_predictions: Optional[List[Dict[str, Any]]] = None):
        """
        Perform robustness checks on predictions.
        
        Returns:
            List of robustness check results (one per instance)
        """
        df = self._to_dataframe(features)
        
        try:
            df_prepared = self._validate_and_prepare_features(df)
        except ValueError as e:
            logger.error(f"Feature validation failed for robustness check: {e}")
            return [{"confidence_low": True, "reasons": ["feature_validation_failed"], "probability_margin": 0.0}]

        results = []
        preds_arr = np.atleast_1d(preds)
        probs_arr = np.atleast_1d(probs_pos)
        
        for i in range(len(df_prepared)):
            prob = float(probs_arr[i])
            checks = {
                "confidence_low": False,
                "reasons": [],
                "probability_margin": abs(prob - 0.5)
            }
            
            # Check probability margin
            if abs(prob - 0.5) < 0.1:
                checks["confidence_low"] = True
                checks["reasons"].append("low_probability_margin")
            
            # Check for extreme values
            for col in df_prepared.columns:
                try:
                    val = df_prepared.iloc[i][col]
                    if pd.isna(val):
                        checks["confidence_low"] = True
                        checks["reasons"].append(f"missing_value_{col}")
                    elif isinstance(val, (int, float)) and abs(val) > 1e7:
                        checks["confidence_low"] = True
                        checks["reasons"].append(f"extreme_value_{col}")
                except Exception:
                    continue
            
            results.append(checks)
        
        return results