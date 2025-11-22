"""
Unit tests for ML prediction functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_converter import convert_raw_inputs_to_features, convert_profile_to_model_features
from src.ml.model_predictor import ModelPredictor
from src.ml.prediction_service import PredictionService


class TestFeatureConverter(unittest.TestCase):
    """Test feature conversion from raw inputs to model format."""
    
    def test_convert_raw_inputs_to_features(self):
        """Test conversion of raw inputs to feature DataFrame."""
        raw_inputs = {
            "age": 30,
            "gender": "female",
            "occupation": "gig_worker",
            "monthly_income": 25000,
            "avg_daily_calls": 1.5,
            "unique_contacts_30d": 10,
            "conscientiousness_score": 4.0,
            "savings_frequency": 0.1,
            "bill_payment_timeliness": 0.9,
            "shg_membership": True,
            "previous_loans": 2,
            "previous_defaults": 0
        }
        
        features_df = convert_raw_inputs_to_features(raw_inputs)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 1)
        self.assertEqual(features_df.iloc[0]["Age"], 30)
        self.assertEqual(features_df.iloc[0]["Gender"], "female")
        self.assertEqual(features_df.iloc[0]["MonthlyIncome"], 25000)
    
    def test_convert_profile_to_model_features(self):
        """Test conversion of profile to feature DataFrame."""
        profile = {
            "demographics": {
                "age": 32,
                "gender": "male",
                "occupation": "small_business_owner",
                "monthly_income": 30000
            },
            "mobile_metadata": {
                "avg_daily_calls": 2.0,
                "unique_contacts_30d": 15
            },
            "psychometrics": {
                "conscientiousness_score": 4.5
            },
            "financial_behavior": {
                "savings_frequency": 0.2,
                "bill_payment_timeliness": 0.95
            },
            "social_network": {
                "shg_membership": False
            },
            "loan_history": {
                "previous_loans": 1,
                "previous_defaults": 0
            }
        }
        
        features_df = convert_profile_to_model_features(profile)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 1)
        self.assertEqual(features_df.iloc[0]["Age"], 32)
    
    def test_na_handling(self):
        """Test handling of NA values."""
        raw_inputs = {
            "age": 30,
            "gender": "female",
            "monthly_income": 20000
        }
        
        features_df = convert_raw_inputs_to_features(raw_inputs)
        
        # Should not raise errors and should use defaults
        self.assertIsInstance(features_df, pd.DataFrame)


class TestModelPredictor(unittest.TestCase):
    """Test model predictor functionality."""
    
    def test_model_predictor_initialization(self):
        """Test model predictor initialization."""
        predictor = ModelPredictor()
        self.assertIsNotNone(predictor)
        self.assertFalse(predictor.is_loaded)
    
    def test_model_load_nonexistent(self):
        """Test loading non-existent model."""
        predictor = ModelPredictor(model_path="nonexistent_model.pkl")
        result = predictor.load_model()
        self.assertFalse(result)
        self.assertFalse(predictor.is_loaded)
    
    def test_robustness_check(self):
        """Test robustness checking."""
        predictor = ModelPredictor()
        
        # Create dummy features
        features_df = pd.DataFrame({
            "Age": [30],
            "MonthlyIncome": [25000],
            "PreviousLoans": [2]
        })
        
        robustness = predictor.check_robustness(
            features_df,
            prediction=1,
            probability=0.6,
            recent_predictions=None
        )
        
        self.assertIn("confidence_low", robustness)
        self.assertIn("reasons", robustness)
        self.assertIn("probability_margin", robustness)


class TestPredictionService(unittest.TestCase):
    """Test prediction service functionality."""
    
    def test_prediction_service_initialization(self):
        """Test prediction service initialization."""
        service = PredictionService()
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.predictor)
    
    def test_format_explanation(self):
        """Test explanation formatting."""
        service = PredictionService()
        
        explanations = [
            {"feature": "Age", "shap_value": 0.5, "abs_shap_value": 0.5},
            {"feature": "Income", "shap_value": -0.3, "abs_shap_value": 0.3},
            {"feature": "Loans", "shap_value": 0.2, "abs_shap_value": 0.2}
        ]
        
        formatted = service._format_explanation(explanations, prediction=1)
        
        self.assertIsInstance(formatted, list)
        if formatted:
            self.assertIn("feature", formatted[0])
            self.assertIn("shap_value", formatted[0])
            self.assertIn("note", formatted[0])
    
    def test_get_next_step(self):
        """Test next step determination."""
        service = PredictionService()
        
        # Not eligible
        next_step = service._get_next_step(prediction=0, probability=0.3)
        self.assertEqual(next_step["action"], "wellness_coach")
        
        # Eligible
        next_step = service._get_next_step(prediction=1, probability=0.8)
        self.assertEqual(next_step["action"], "proceed_application")


if __name__ == "__main__":
    unittest.main()

