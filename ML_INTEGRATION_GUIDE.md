# ML Model Integration Guide

## Overview

This document explains how to use the newly integrated ML loan prediction model in the GigBridge microfinance application.

## Quick Start

### 1. Train the Model

If you don't have a trained model yet, you need to train it first:

```bash
# Option 1: Provide CSV path
python scripts/export_model_from_notebook.py --csv-path /path/to/Loan_default.csv

# Option 2: Place CSV in data/ folder
# Copy Loan_default.csv to data/Loan_default.csv
python scripts/export_model_from_notebook.py
```

The model will be saved to `models/loan_xgb.pkl`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies include:
- `xgboost>=2.0.0`
- `scikit-learn>=1.3.0`
- `shap>=0.42.0`
- `pandas>=2.0.0`
- `joblib>=1.3.0`

### 3. Run the Application

```bash
streamlit run app.py
```

## Using the ML Model

### Frontend Usage

1. **Select Assessment Mode**
   - Choose "ML Model (Recommended)" for ML-only predictions
   - Choose "RAG Agent" for RAG-only assessment
   - Choose "Both" to see both predictions

2. **Fill in Form Fields**
   - No JSON upload required! Use the UI controls:
     - Number inputs for numeric values (age, income, etc.)
     - Dropdowns for categorical values (gender, occupation)
     - Sliders for psychometric responses
     - Checkboxes for boolean values (SHG membership)

3. **Run Assessment**
   - Click "Run Assessment"
   - View ML prediction results with SHAP explanations

4. **Review Results**
   - **Prediction**: Eligible (1) or Not Eligible (0)
   - **Probability**: Confidence score (0-1)
   - **Key Factors**: Top 5-6 features with SHAP values
   - **Next Steps**: Action guidance (wellness coach or proceed)

5. **Consent and Submit**
   - Check the consent checkbox
   - Click "Submit with Consent" to save data
   - Or click "Decline Consent" to delete all data

### API Usage (Programmatic)

If you want to call the prediction service programmatically:

```python
from src.ml.prediction_service import PredictionService

# Initialize service
service = PredictionService()

# Prepare raw inputs
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
    "previous_defaults": 0,
    "previous_late_payments": 1,
    "avg_repayment_delay_days": 0.0
}

# Make prediction
result = service.predict_from_raw_inputs(
    raw_inputs,
    consent=True,  # Set to True to save data
    user_id="user_123"
)

# Access results
print(f"Prediction: {result['prediction']}")  # 0 or 1
print(f"Probability: {result['probability']}")  # 0.0-1.0
print(f"Explanations: {result['explanation']}")  # List of feature explanations
print(f"Next Step: {result['next_step']}")  # Action guidance
```

## Response Format

The prediction service returns:

```json
{
  "prediction": 1,
  "probability": 0.75,
  "explanation": [
    {
      "feature": "MonthlyIncome",
      "shap_value": 0.123,
      "note": "Strong: Monthly Income contributes positively to eligibility"
    },
    {
      "feature": "PreviousDefaults",
      "shap_value": -0.045,
      "note": "Note: Previous Defaults has some negative impact"
    }
  ],
  "next_step": {
    "action": "proceed_application",
    "message": "You appear eligible for a loan. Proceed with the application process.",
    "redirect_url": "/application"
  },
  "meta": {
    "confidence_low": false,
    "confidence_reasons": [],
    "probability_margin": 0.25,
    "timestamp": "2025-01-15T10:30:00"
  }
}
```

## Wellness Coach

If a user is predicted as **not eligible** (prediction=0), they are automatically routed to the Wellness Coach module, which provides:

- Improvement suggestions
- Financial literacy resources
- Actionable steps to improve eligibility

Access via:
- Automatic redirect when prediction is 0
- Navigation menu → "Wellness Coach"

## Admin Features

Access the admin view with password `admin123` (change in production!):

1. **View Submissions**
   - See all consented submissions
   - View predictions and explanations
   - Filter and search

2. **Delete Submissions**
   - Remove individual submissions
   - Audit trail maintained

3. **Retrain Model**
   - Trigger model retraining
   - Updates `models/loan_xgb.pkl`

4. **Reload Model**
   - Reload model without restarting app
   - Useful after retraining

5. **View Telemetry**
   - Feature distribution statistics
   - Model monitoring data

## Testing

Run unit tests:

```bash
python -m pytest tests/test_ml_prediction.py -v
```

Or run all tests:

```bash
python -m pytest tests/ -v
```

## Feature Mapping

The system converts raw inputs to model features as follows:

| User Input | Model Feature | Type |
|------------|---------------|------|
| age | Age | Numeric |
| gender | Gender | Categorical (one-hot) |
| occupation | Occupation | Categorical (one-hot) |
| monthly_income | MonthlyIncome | Numeric |
| avg_daily_calls | AvgDailyCalls | Numeric |
| unique_contacts_30d | UniqueContacts30d | Numeric |
| conscientiousness_score | ConscientiousnessScore | Numeric |
| savings_frequency | SavingsFrequency | Numeric |
| bill_payment_timeliness | BillPaymentTimeliness | Numeric |
| shg_membership | SHGMembership | Categorical (Yes/No) |
| previous_loans | PreviousLoans | Numeric |
| previous_defaults | PreviousDefaults | Numeric |

## Consent Handling

- **With Consent (`consent=true`)**: Data is saved to database and telemetry is logged
- **Without Consent (`consent=false`)**: Only prediction is returned, no data storage

Telemetry logs are stored in `data/telemetry/feature_distributions.jsonl` (only if consented).

## Troubleshooting

### Model Not Found

**Error**: "ML Model not found"

**Solution**: Train the model first:
```bash
python scripts/export_model_from_notebook.py --csv-path /path/to/Loan_default.csv
```

### SHAP Not Available

**Error**: "SHAP explainer not available"

**Solution**: Install SHAP:
```bash
pip install shap>=0.42.0
```

### Low Confidence Warnings

If you see "Low Confidence" warnings:
- Check if probability is close to 0.5 (uncertain prediction)
- Review feature values for outliers
- Consider retraining model with more data

### Import Errors

If you see import errors:
```bash
pip install -r requirements.txt
```

## Architecture

```
┌─────────────┐
│   Frontend  │ (app.py - Streamlit)
│  (UI Forms) │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ PredictionService│ (src/ml/prediction_service.py)
└──────┬───────────┘
       │
       ├──► ModelPredictor (src/ml/model_predictor.py)
       │    └──► XGBoost Model (models/loan_xgb.pkl)
       │
       ├──► FeatureConverter (src/ml/feature_converter.py)
       │    └──► Raw Inputs → Model Features
       │
       └──► Telemetry (src/ml/telemetry.py)
            └──► Feature Distribution Logging
```

## Files Structure

```
GigBridge/
├── app.py                          # Main Streamlit app (updated)
├── src/
│   └── ml/                         # New ML module
│       ├── __init__.py
│       ├── model_predictor.py      # Model loading & prediction
│       ├── feature_converter.py    # Input → Features conversion
│       ├── prediction_service.py   # High-level prediction API
│       └── telemetry.py            # Feature distribution logging
├── scripts/
│   └── export_model_from_notebook.py  # Model training script
├── models/                         # Model storage (created)
│   └── loan_xgb.pkl               # Trained model (generated)
├── tests/
│   └── test_ml_prediction.py      # Unit tests
├── requirements.txt                # Updated dependencies
└── CHANGELOG.md                    # Change log
```

## Next Steps

1. **Production Deployment**:
   - Change admin password
   - Set up proper authentication
   - Configure model retraining schedule
   - Set up monitoring and alerts

2. **Model Improvements**:
   - Collect more training data
   - Tune hyperparameters
   - Add more features
   - Implement A/B testing

3. **Feature Enhancements**:
   - Real-time model updates
   - Batch prediction API
   - Model versioning
   - Performance dashboards

