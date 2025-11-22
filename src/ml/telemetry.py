"""
Telemetry Module

This module handles logging of feature distributions for debugging and model monitoring.
Only logs if user has consented.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def log_feature_distribution(features: pd.DataFrame, user_id: Optional[str] = None, 
                            consented: bool = False):
    """
    Log feature distribution for telemetry (only if consented).
    
    Args:
        features: DataFrame with feature values
        user_id: Optional user identifier
        consented: Whether user consented to data collection
    """
    if not consented:
        # Log only aggregate stats without user ID
        logger.debug("Feature distribution logged (no consent, anonymized)")
        return
    
    try:
        # Create telemetry directory
        telemetry_dir = Path("data/telemetry")
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        # Log feature statistics
        stats = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "feature_stats": {}
        }
        
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                col_data = features[col].dropna()
                if len(col_data) > 0:
                    stats["feature_stats"][col] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()) if len(col_data) > 1 else 0.0,
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "count": int(len(col_data))
                    }
            else:
                # Categorical
                value_counts = features[col].value_counts().to_dict()
                stats["feature_stats"][col] = {
                    "value_counts": {str(k): int(v) for k, v in value_counts.items()}
                }
        
        # Append to telemetry log
        telemetry_file = telemetry_dir / "feature_distributions.jsonl"
        with open(telemetry_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
        
        logger.info(f"Feature distribution logged for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error logging feature distribution: {e}")

