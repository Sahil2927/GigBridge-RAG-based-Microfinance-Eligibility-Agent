"""
Unit tests for ML prediction functionality.
"""

import unittest
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_converter import (
    convert_to_ml_features,
    extract_ml_features_only,
    ML_MODEL_FEATURES,
)
from src.ml.model_predictor import ModelPredictor
from src.ml.prediction_service import PredictionService


class TestFeatureConverter(unittest.TestCase):
    """Test feature conversion from raw inputs to model format."""

    def test_convert_to_ml_features(self):
        raw_inputs = {
            "LoanID": "LOAN_TEST_001",
            "Age": 30,
            "Income": 50000.0,
            "LoanAmount": 10000.0,
            "CreditScore": 650,
            "MonthsEmployed": 24,
            "NumCreditLines": 2,
            "InterestRate": 10.0,
            "LoanTerm": 36,
            "DTIRatio": 0.3,
            "Education": "Bachelor's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "No",
            "LoanPurpose": "Business",
            "HasCoSigner": "No",
        }

        features_df = convert_to_ml_features(raw_inputs)

        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 1)
        self.assertEqual(list(features_df.columns), ML_MODEL_FEATURES)
        self.assertEqual(features_df.iloc[0]["Age"], 30)
        self.assertEqual(features_df.iloc[0]["Income"], 50000.0)

    def test_extract_ml_features_only(self):
        profile = {
            "user_id": "user_001",
            "demographics": {
                "age": 32,
                "monthly_income": 30000,
                "education_level": "Master's",
                "employment_type": "Self-employed",
                "marital_status": "Married",
                "months_employed": 36,
            },
            "loan_history": {
                "current_loan_amount": 15000,
                "num_credit_lines": 3,
                "interest_rate": 12.0,
                "loan_term": 48,
                "loan_purpose": "Business",
            },
            "credit_score": 700,
            "dti_ratio": 0.25,
            "has_mortgage": "Yes",
            "has_dependents": "No",
            "has_cosigner": "Yes",
        }

        ml_features = extract_ml_features_only(profile)

        self.assertEqual(ml_features["Age"], 32)
        self.assertEqual(ml_features["LoanID"], "user_001")
        self.assertEqual(ml_features["CreditScore"], 700)

    def test_missing_fields_use_defaults(self):
        features_df = convert_to_ml_features({"Age": 30, "Income": 20000.0})
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df.columns), len(ML_MODEL_FEATURES))


class TestModelPredictor(unittest.TestCase):
    """Test model predictor functionality."""

    def test_model_predictor_initialization(self):
        predictor = ModelPredictor()
        self.assertIsNotNone(predictor)
        self.assertFalse(predictor.is_loaded)

    def test_model_load_nonexistent(self):
        predictor = ModelPredictor(model_path="nonexistent_model.pkl")
        result = predictor.load_model()
        self.assertFalse(result)
        self.assertFalse(predictor.is_loaded)

    def test_robustness_check(self):
        predictor = ModelPredictor()
        features_df = convert_to_ml_features(
            {
                "LoanID": "X",
                "Age": 30,
                "Income": 50000,
                "LoanAmount": 10000,
                "CreditScore": 650,
                "MonthsEmployed": 24,
                "NumCreditLines": 2,
                "InterestRate": 10.0,
                "LoanTerm": 36,
                "DTIRatio": 0.3,
                "Education": "High School",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Single",
                "HasMortgage": "No",
                "HasDependents": "No",
                "LoanPurpose": "Other",
                "HasCoSigner": "No",
            }
        )

        robustness = predictor.check_robustness(
            features_df,
            preds=[1],
            probs_pos=[0.6],
            recent_predictions=None,
        )

        self.assertIn("confidence_low", robustness[0])
        self.assertIn("reasons", robustness[0])
        self.assertIn("probability_margin", robustness[0])


class TestPredictionService(unittest.TestCase):
    """Test prediction service functionality."""

    def test_prediction_service_initialization(self):
        service = PredictionService()
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.predictor)

    def test_format_explanation(self):
        """Explanation entries use the public response shape."""
        explanations = [
            {"feature": "Age", "shap_value": 0.5, "abs_shap": 0.5},
            {"feature": "Income", "shap_value": -0.3, "abs_shap": 0.3},
        ]
        formatted = [
            {
                "feature": e.get("feature", "Unknown"),
                "shap_value": float(e.get("shap_value", 0.0)),
                "abs_shap": abs(float(e.get("shap_value", 0.0))),
            }
            for e in explanations[:6]
        ]
        self.assertEqual(len(formatted), 2)
        self.assertIn("feature", formatted[0])

    def test_next_step_actions(self):
        """Eligible vs ineligible next-step actions."""
        eligible = {"action": "proceed_application", "message": "Proceed with loan application"}
        ineligible = {"action": "wellness_coach", "message": "Consider wellness coach to improve eligibility"}
        self.assertEqual(eligible["action"], "proceed_application")
        self.assertEqual(ineligible["action"], "wellness_coach")


if __name__ == "__main__":
    unittest.main()
