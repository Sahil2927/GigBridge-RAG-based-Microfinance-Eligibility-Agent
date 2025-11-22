"""
SHAP Explainer Utilities

Add this as a new file: src/ml/shap_explainer.py

Provides educational content and utilities for explaining SHAP values
to non-technical users in the context of ethical AI.
"""

from typing import Dict, List, Any
import numpy as np


def get_shap_educational_content() -> Dict[str, str]:
    """
    Returns educational content about SHAP for user transparency.
    """
    return {
        "what_is_shap": """
        SHAP (SHapley Additive exPlanations) is a method to explain individual predictions 
        from machine learning models. It's based on game theory and tells you how much 
        each feature contributed to the final decision.
        """,
        
        "how_shap_works": """
        Imagine your loan application is being reviewed by a committee. Each committee member 
        (representing a feature like age, income, credit score) votes on whether to approve 
        your loan. SHAP calculates how much each member's vote influenced the final decision.
        
        The process:
        1. Start with a baseline (average prediction for everyone)
        2. Add each feature one by one
        3. Measure how much the prediction changes
        4. Average across all possible orders of adding features
        5. This gives a fair "credit" to each feature
        """,
        
        "positive_shap": """
        A positive SHAP value means this feature is pushing the prediction toward APPROVAL.
        
        Example: If your CreditScore has a SHAP value of +0.15, it means your credit score 
        is increasing your approval probability by 15 percentage points compared to the average.
        """,
        
        "negative_shap": """
        A negative SHAP value means this feature is pushing the prediction toward REJECTION.
        
        Example: If your DTIRatio (debt-to-income) has a SHAP value of -0.12, it means your 
        debt level is decreasing your approval probability by 12 percentage points.
        """,
        
        "magnitude": """
        The size (absolute value) of the SHAP value tells you HOW IMPORTANT that feature is.
        
        - Small values (< 0.05): Minor influence
        - Medium values (0.05-0.15): Moderate influence  
        - Large values (> 0.15): Major influence
        """,
        
        "fairness": """
        SHAP ensures fairness through mathematical properties:
        
        1. **Additivity**: All SHAP values sum to explain the full prediction
        2. **Consistency**: If a feature helps more, its SHAP value increases
        3. **Missingness**: Features not used get zero SHAP value
        4. **Symmetry**: Features with equal contributions get equal credit
        """,
        
        "limitations": """
        While SHAP is powerful, it has limitations:
        
        - It explains WHAT the model thinks, not WHY the world is that way
        - High computation time for large datasets
        - Assumes feature independence (which may not be true)
        - Local explanation (specific to YOUR application, not general rules)
        """
    }


def interpret_shap_value(feature_name: str, shap_value: float, 
                        feature_value: Any = None) -> Dict[str, str]:
    """
    Provide human-readable interpretation of a SHAP value.
    
    Args:
        feature_name: Name of the feature
        shap_value: SHAP value for this feature
        feature_value: Actual value of the feature (optional)
    
    Returns:
        Dictionary with interpretation details
    """
    # Determine impact magnitude
    abs_shap = abs(shap_value)
    if abs_shap < 0.05:
        magnitude = "minor"
        emoji = "📊"
    elif abs_shap < 0.15:
        magnitude = "moderate"
        emoji = "📈"
    else:
        magnitude = "major"
        emoji = "⚡"
    
    # Determine direction
    direction = "positive" if shap_value > 0 else "negative"
    impact = "increases" if shap_value > 0 else "decreases"
    
    # Create interpretation
    interpretation = {
        "summary": f"{emoji} {magnitude.capitalize()} {direction} impact",
        "detail": f"This feature {impact} your approval probability by {abs_shap:.1%}",
        "magnitude": magnitude,
        "direction": direction,
        "emoji": emoji
    }
    
    # Add feature-specific insights
    feature_lower = feature_name.lower()
    
    if "creditscore" in feature_lower or "credit_score" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Your credit score is working in your favor"
            interpretation["action"] = "Maintain good credit habits to keep this advantage"
        else:
            interpretation["insight"] = "Your credit score is a concern"
            interpretation["action"] = "Focus on improving credit: pay bills on time, reduce credit utilization"
    
    elif "income" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Your income level supports your application"
            interpretation["action"] = "Keep documentation of income sources ready"
        else:
            interpretation["insight"] = "Income level is below typical threshold"
            interpretation["action"] = "Consider showing additional income sources or co-applicant"
    
    elif "dti" in feature_lower or "debt" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Your debt-to-income ratio is healthy"
            interpretation["action"] = "Maintain low debt levels relative to income"
        else:
            interpretation["insight"] = "Debt-to-income ratio is a risk factor"
            interpretation["action"] = "Pay down existing debts to improve this ratio"
    
    elif "age" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Your age profile is favorable"
            interpretation["action"] = "Continue building your credit history"
        else:
            interpretation["insight"] = "Age is working against you (limited credit history)"
            interpretation["action"] = "Build longer credit history through responsible borrowing"
    
    elif "employed" in feature_lower or "employment" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Employment stability is strong"
            interpretation["action"] = "Maintain steady employment"
        else:
            interpretation["insight"] = "Employment history is a concern"
            interpretation["action"] = "Build longer employment track record"
    
    elif "loanterm" in feature_lower or "loan_term" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Loan term is appropriate for your profile"
            interpretation["action"] = "This loan structure works well for you"
        else:
            interpretation["insight"] = "Loan term may be too aggressive"
            interpretation["action"] = "Consider a longer loan term for lower monthly payments"
    
    elif "cosigner" in feature_lower or "co_signer" in feature_lower:
        if shap_value > 0:
            interpretation["insight"] = "Having a co-signer strengthens your application"
            interpretation["action"] = "Ensure co-signer has strong credit profile"
        else:
            interpretation["insight"] = "Lack of co-signer is a weak point"
            interpretation["action"] = "Consider finding a creditworthy co-signer"
    
    else:
        # Generic interpretation
        if shap_value > 0:
            interpretation["insight"] = "This factor supports your application"
            interpretation["action"] = "Maintain this positive aspect"
        else:
            interpretation["insight"] = "This factor weakens your application"
            interpretation["action"] = "Work on improving this area"
    
    return interpretation


def generate_improvement_plan(explanations: List[Dict[str, Any]], 
                              prediction: int) -> Dict[str, Any]:
    """
    Generate a personalized improvement plan based on SHAP explanations.
    
    Args:
        explanations: List of SHAP explanation dictionaries
        prediction: Model prediction (0 or 1)
    
    Returns:
        Dictionary with improvement plan
    """
    # Separate positive and negative features
    negative_features = [
        exp for exp in explanations 
        if exp.get("shap_value", 0) < 0
    ]
    
    positive_features = [
        exp for exp in explanations 
        if exp.get("shap_value", 0) > 0
    ]
    
    # Sort by absolute impact
    negative_features = sorted(
        negative_features, 
        key=lambda x: abs(x.get("shap_value", 0)), 
        reverse=True
    )
    
    positive_features = sorted(
        positive_features, 
        key=lambda x: x.get("shap_value", 0), 
        reverse=True
    )
    
    plan = {
        "overall_status": "APPROVED" if prediction == 1 else "NOT APPROVED",
        "priority_actions": [],
        "maintain_strengths": [],
        "quick_wins": [],
        "long_term_goals": []
    }
    
    if prediction == 0:  # Not approved
        # Priority actions (top 3 negative factors)
        for i, exp in enumerate(negative_features[:3], 1):
            feature = exp.get("feature", "Unknown")
            shap_val = exp.get("shap_value", 0)
            interpretation = interpret_shap_value(feature, shap_val)
            
            plan["priority_actions"].append({
                "rank": i,
                "feature": feature,
                "impact": abs(shap_val),
                "action": interpretation.get("action", "Improve this factor"),
                "insight": interpretation.get("insight", "")
            })
        
        # Quick wins (medium impact items that are easier to fix)
        quick_win_features = ["DTIRatio", "NumCreditLines", "HasCoSigner"]
        for exp in negative_features[3:6]:
            feature = exp.get("feature", "")
            if any(qw in feature for qw in quick_win_features):
                shap_val = exp.get("shap_value", 0)
                interpretation = interpret_shap_value(feature, shap_val)
                
                plan["quick_wins"].append({
                    "feature": feature,
                    "action": interpretation.get("action", "")
                })
        
        # Long-term goals (harder to change items)
        long_term_features = ["CreditScore", "Income", "Age", "MonthsEmployed"]
        for exp in negative_features:
            feature = exp.get("feature", "")
            if any(lt in feature for lt in long_term_features):
                shap_val = exp.get("shap_value", 0)
                interpretation = interpret_shap_value(feature, shap_val)
                
                plan["long_term_goals"].append({
                    "feature": feature,
                    "action": interpretation.get("action", "")
                })
    
    # Strengths to maintain (for both approved and not approved)
    for exp in positive_features[:5]:
        feature = exp.get("feature", "Unknown")
        shap_val = exp.get("shap_value", 0)
        interpretation = interpret_shap_value(feature, shap_val)
        
        plan["maintain_strengths"].append({
            "feature": feature,
            "impact": shap_val,
            "insight": interpretation.get("insight", ""),
            "action": interpretation.get("action", "")
        })
    
    return plan


def calculate_shap_statistics(explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from SHAP explanations.
    
    Args:
        explanations: List of SHAP explanation dictionaries
    
    Returns:
        Dictionary with statistics
    """
    if not explanations:
        return {}
    
    shap_values = [exp.get("shap_value", 0) for exp in explanations]
    abs_shaps = [abs(val) for val in shap_values]
    
    positive_shaps = [val for val in shap_values if val > 0]
    negative_shaps = [val for val in shap_values if val < 0]
    
    stats = {
        "total_features": len(explanations),
        "positive_features": len(positive_shaps),
        "negative_features": len(negative_shaps),
        "total_positive_impact": sum(positive_shaps) if positive_shaps else 0,
        "total_negative_impact": sum(negative_shaps) if negative_shaps else 0,
        "net_impact": sum(shap_values),
        "max_positive": max(positive_shaps) if positive_shaps else 0,
        "max_negative": min(negative_shaps) if negative_shaps else 0,
        "mean_abs_impact": np.mean(abs_shaps) if abs_shaps else 0,
        "dominant_factors": []
    }
    
    # Identify dominant factors (top 20% by absolute impact)
    threshold = np.percentile(abs_shaps, 80) if abs_shaps else 0
    stats["dominant_factors"] = [
        exp.get("feature", "Unknown") 
        for exp in explanations 
        if abs(exp.get("shap_value", 0)) >= threshold
    ]
    
    return stats


def format_shap_for_user(feature_name: str, shap_value: float) -> str:
    """
    Format SHAP value in user-friendly language.
    
    Args:
        feature_name: Name of the feature
        shap_value: SHAP value
    
    Returns:
        User-friendly description
    """
    # Clean feature name
    clean_name = feature_name.replace("num__", "").replace("cat__", "").replace("_", " ").title()
    
    # Determine impact
    abs_val = abs(shap_value)
    percentage = abs_val * 100  # Convert to percentage points
    
    if shap_value > 0:
        direction = "increases"
        emoji = "✅"
    else:
        direction = "decreases"
        emoji = "⚠️"
    
    # Create description
    if abs_val < 0.01:
        description = f"{emoji} {clean_name}: Minimal impact ({direction} by <1%)"
    elif abs_val < 0.05:
        description = f"{emoji} {clean_name}: Small impact ({direction} by ~{percentage:.1f}%)"
    elif abs_val < 0.15:
        description = f"{emoji} {clean_name}: Moderate impact ({direction} by ~{percentage:.1f}%)"
    else:
        description = f"{emoji} {clean_name}: Major impact ({direction} by ~{percentage:.1f}%)"
    
    return description