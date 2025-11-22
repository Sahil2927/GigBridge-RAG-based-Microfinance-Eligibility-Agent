# Changelog

## [Unreleased] - ML Model Integration

### Added
- **ML Model Integration**: Integrated XGBoost loan prediction model from `loan_xgboost.ipynb`
  - Model predictor module (`src/ml/model_predictor.py`) for loading and using the XGBoost model
  - Feature converter (`src/ml/feature_converter.py`) to convert raw user inputs to model feature vectors
  - Prediction service (`src/ml/prediction_service.py`) for high-level prediction API with SHAP explanations

- **SHAP Explanations**: Added SHAP value computation and user-friendly explanations
  - Top 5-6 features shown based on SHAP absolute magnitude
  - Positive contributors shown for eligible predictions (strong points)
  - Negative contributors shown for ineligible predictions (weak points)
  - User-friendly notes for each feature explaining its impact

- **Raw Input Support**: Frontend now accepts raw inputs via UI controls (meters, text fields, dropdowns)
  - Removed requirement for JSON file uploads (still supported as optional)
  - Form inputs directly converted to model feature vectors
  - Better UX with immediate input validation

- **Consent Handling**: Enhanced consent mechanism
  - ML predictions only saved to database if `consent=true`
  - Telemetry logging respects consent (only logs if consented)
  - Clear separation between consented and non-consented data

- **Wellness Coach Module**: Added routing for ineligible users
  - New `/wellness-coach` page with improvement suggestions
  - Automatic routing when prediction is not eligible
  - Actionable recommendations for improving eligibility

- **Admin Endpoints**: Enhanced admin view with ML model management
  - View all consented submissions with ML predictions
  - Delete submissions functionality
  - Retrain model trigger button
  - Reload model button
  - Telemetry viewing capability

- **Model Robustness Checks**: Added confidence and robustness monitoring
  - Low confidence flagging when probability margin is small
  - Detection of suspicious similarity across predictions
  - Out-of-distribution feature detection
  - Confidence reasons provided in response

- **Telemetry/Logging**: Feature distribution logging for debugging
  - Logs feature statistics only if user consented
  - Stores in `data/telemetry/feature_distributions.jsonl`
  - Helps with model monitoring and debugging

- **Model Training Script**: Created `scripts/export_model_from_notebook.py`
  - Replicates training logic from Colab notebook
  - Trains XGBoost model and saves to `models/loan_xgb.pkl`
  - Handles data preprocessing and model evaluation

- **Unit Tests**: Comprehensive test suite for ML functionality
  - Tests for feature conversion (raw inputs → features)
  - Tests for model predictor initialization and robustness checks
  - Tests for prediction service and explanation formatting

### Changed
- **Frontend**: Updated `app.py` to support both ML and RAG assessments
  - Assessment mode selection (ML, RAG, or Both)
  - ML predictions displayed with SHAP explanations
  - Consent checkbox before submission
  - Improved result display with confidence indicators

- **Dependencies**: Updated `requirements.txt` with ML libraries
  - Added `xgboost>=2.0.0`
  - Added `scikit-learn>=1.3.0`
  - Added `shap>=0.42.0`
  - Added `pandas>=2.0.0`
  - Added `joblib>=1.3.0`

### Technical Details

#### Model Integration
- Model expected at `models/loan_xgb.pkl` (created by training script)
- Model is a scikit-learn Pipeline with ColumnTransformer and XGBClassifier
- Handles both categorical (one-hot encoded) and numeric features

#### Feature Mapping
The feature converter maps user profile data to model features:
- Demographics: Age, Gender, Occupation, MonthlyIncome, etc.
- Mobile metadata: AvgDailyCalls, UniqueContacts30d, AirtimeTopupFrequency, etc.
- Psychometrics: ConscientiousnessScore
- Financial behavior: SavingsFrequency, BillPaymentTimeliness, WalletBalanceLows90d
- Social network: SHGMembership, PeerMonitoringStrength
- Loan history: PreviousLoans, PreviousDefaults, PreviousLatePayments, AvgRepaymentDelayDays

#### API Response Format
```json
{
  "prediction": 0 or 1,
  "probability": 0.0-1.0,
  "explanation": [
    {
      "feature": "FeatureName",
      "shap_value": 0.123,
      "note": "User-friendly explanation"
    }
  ],
  "next_step": {
    "action": "wellness_coach" or "proceed_application",
    "message": "Guidance message",
    "redirect_url": "/wellness-coach" or "/application"
  },
  "meta": {
    "confidence_low": true/false,
    "confidence_reasons": ["reason1", "reason2"],
    "probability_margin": 0.1,
    "timestamp": "ISO8601"
  }
}
```

### How to Use

1. **Train the model** (if model doesn't exist):
   ```bash
   python scripts/export_model_from_notebook.py --csv-path /path/to/Loan_default.csv
   ```
   Or place `Loan_default.csv` in `data/` folder and run:
   ```bash
   python scripts/export_model_from_notebook.py
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Frontend Usage**:
   - Select "ML Model (Recommended)" assessment mode
   - Fill in form fields (no JSON upload required)
   - Click "Run Assessment"
   - Review prediction and SHAP explanations
   - Check consent checkbox and click "Submit with Consent"

4. **Admin Access**:
   - Navigate to "Admin View"
   - Password: `admin123` (change in production!)
   - View submissions, delete entries, retrain model

5. **Run Tests**:
   ```bash
   python -m pytest tests/test_ml_prediction.py
   ```

### Files Changed
- `app.py` - Integrated ML model, updated UI, added wellness coach
- `src/ml/` - New ML module with predictor, converter, service, telemetry
- `scripts/export_model_from_notebook.py` - Model training script
- `requirements.txt` - Added ML dependencies
- `tests/test_ml_prediction.py` - Unit tests for ML functionality
- `CHANGELOG.md` - This file

### Notes
- Model file (`models/loan_xgb.pkl`) must be created before using ML predictions
- SHAP explanations require the model to be loaded successfully
- Consent is required for data storage and telemetry logging
- Admin password should be changed in production environments

