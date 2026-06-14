"""
Generate a synthetic Loan_default.csv for demo / cloud bootstrap.

Column names match the training script in export_model_from_notebook.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = {
    "Education": ["High School", "Bachelor's", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Unemployed", "Self-employed"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "HasMortgage": ["Yes", "No"],
    "HasDependents": ["Yes", "No"],
    "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
    "HasCoSigner": ["Yes", "No"],
}


def generate_synthetic_loan_csv(
    output_path: str = "data/Loan_default.csv",
    n_rows: int = 3000,
    random_state: int = 42,
) -> str:
    """Create synthetic loan data and write CSV. Returns output path."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(22, 70, size=n_rows)
    income = rng.normal(55000, 22000, size=n_rows).clip(15000, 250000)
    loan_amount = rng.normal(12000, 8000, size=n_rows).clip(1000, 80000)
    credit_score = rng.integers(450, 820, size=n_rows)
    months_employed = rng.integers(0, 240, size=n_rows)
    num_credit_lines = rng.integers(0, 10, size=n_rows)
    interest_rate = rng.uniform(4.0, 22.0, size=n_rows)
    loan_term = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 360], size=n_rows)
    dti_ratio = rng.uniform(0.05, 0.85, size=n_rows)

    data = {
        "LoanID": [f"LOAN_{i:06d}" for i in range(n_rows)],
        "Age": age,
        "Income": np.round(income, 2),
        "LoanAmount": np.round(loan_amount, 2),
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": np.round(interest_rate, 2),
        "LoanTerm": loan_term,
        "DTIRatio": np.round(dti_ratio, 3),
    }

    for column, choices in CATEGORICAL_COLUMNS.items():
        data[column] = rng.choice(choices, size=n_rows)

    df = pd.DataFrame(data)

    # Simple rule-based default label with noise for learnable signal.
    risk = (
        (850 - df["CreditScore"]) / 400
        + df["DTIRatio"] * 0.8
        + (df["LoanAmount"] / df["Income"].clip(lower=1)) * 0.4
        + (df["EmploymentType"] == "Unemployed").astype(float) * 0.6
        + (df["HasCoSigner"] == "No").astype(float) * 0.15
    )
    default_prob = 1 / (1 + np.exp(-(risk - 1.1)))
    df["Default"] = rng.binomial(1, default_prob.clip(0.02, 0.85)).astype(int)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return str(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Loan_default.csv")
    parser.add_argument("--output", default="data/Loan_default.csv")
    parser.add_argument("--rows", type=int, default=3000)
    args = parser.parse_args()

    path = generate_synthetic_loan_csv(args.output, n_rows=args.rows)
    print(f"Wrote synthetic dataset to {path}")
