"""
Improved ModelPredictor

Key features:
- Accepts dict/list/DataFrame inputs (better frontend UX)
- Robust feature name extraction (pipeline-aware)
- Optional probability calibration
- Better SHAP initialization (works with sklearn wrappers & xgboost)
- Improved robustness checks (margin, entropy, z-score OOD)
- Batch-safe predict/explain
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from math import log2

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


def entropy_from_probs(probs: np.ndarray) -> float:
    """Calculate Shannon entropy for binary probs (or multi-class)."""
    # probs: 1D array summing to 1
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return -float(np.sum(probs * np.log2(probs)))


class ModelPredictor:
    def __init__(self,
                 model_path: Optional[str] = None,
                 stats_path: Optional[str] = None,
                 calibrator_path: Optional[str] = None,
                 prob_margin_threshold: float = 0.1,
                 ood_zscore_threshold: float = 5.0,
                 similarity_count_threshold: int = 8):
        """
        Args:
            model_path: path to pickled pipeline/model
            stats_path: optional path to training stats (pickle/dict) containing means/stds per numeric feature for OOD checks
            calibrator_path: optional path to a pre-fitted probability calibrator (e.g., sklearn CalibratedClassifierCV) saved with joblib
            prob_margin_threshold: margin from 0.5 below which confidence is flagged (e.g., 0.1 → probabilities in (0.4,0.6) flagged)
            ood_zscore_threshold: z-score magnitude beyond which a numeric feature is considered OOD
            similarity_count_threshold: number of similar recent preds to flag suspicious similarity
        """
        self.model_path = model_path or "models/loan_xgb.pkl"
        self.calibrator_path = calibrator_path
        self.stats_path = stats_path
        self.prob_margin_threshold = prob_margin_threshold
        self.ood_zscore_threshold = ood_zscore_threshold
        self.similarity_count_threshold = similarity_count_threshold

        self.model = None
        self.calibrator = None
        self.explainer = None
        self.feature_names: Optional[List[str]] = None
        self.train_stats: Optional[Dict[str, Dict[str, float]]] = None  # e.g. {"age": {"mean":.., "std":..}, ...}
        self.is_loaded = False

    def _load_pickle(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.warning(f"File not found: {path}")
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def load_model(self) -> bool:
        """Load model, calibrator and stats (if provided). Extract feature names robustly."""
        model_obj = self._load_pickle(self.model_path)
        if model_obj is None:
            return False
        self.model = model_obj
        logger.info(f"Model loaded from {self.model_path}")

        if self.calibrator_path:
            cal = self._load_pickle(self.calibrator_path)
            if cal is not None:
                self.calibrator = cal
                logger.info("Calibrator loaded")

        if self.stats_path:
            stats = self._load_pickle(self.stats_path)
            if isinstance(stats, dict):
                self.train_stats = stats
                logger.info("Train stats loaded")

        # Extract feature names:
        # 1) If pipeline with preprocess and get_feature_names_out, use it.
        # 2) If pipeline has feature_names_in_ or get_booster feature names, try those.
        try:
            # Case: sklearn.pipeline.Pipeline
            if hasattr(self.model, "named_steps"):
                preprocess = None
                # Try common names
                for candidate in ("preprocess", "preprocessor", "transformer", "feature_union"):
                    preprocess = self.model.named_steps.get(candidate)
                    if preprocess is not None:
                        break
                # If we found preprocess and it supports get_feature_names_out:
                if preprocess is not None:
                    if hasattr(preprocess, "get_feature_names_out"):
                        try:
                            self.feature_names = list(preprocess.get_feature_names_out())
                        except Exception:
                            # If it requires input cols, try passing placeholder names if available
                            self.feature_names = None
                    else:
                        # try transformers_ attr to assemble names
                        if hasattr(preprocess, "transformers_"):
                            names = []
                            for name, trans, cols in preprocess.transformers_:
                                # `cols` can be list of column names or slice; handle common cases
                                if isinstance(cols, (list, tuple)):
                                    if hasattr(trans, "get_feature_names_out"):
                                        try:
                                            out = trans.get_feature_names_out(cols)
                                            names.extend(list(out))
                                        except Exception:
                                            names.extend(list(cols))
                                    else:
                                        names.extend(list(cols))
                            if names:
                                self.feature_names = names
                # fallback: model's first step or final estimator's feature_names_in_
            # Case: estimators with feature_names_in_
            if self.feature_names is None and hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
            # For xgboost Booster or sklearn wrapper
            if self.feature_names is None:
                try:
                    # XGBoost sklearn wrapper may expose get_booster().feature_names
                    if hasattr(self.model, "get_booster"):
                        booster = self.model.get_booster()
                        if booster is not None and hasattr(booster, "feature_names"):
                            self.feature_names = list(booster.feature_names)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Feature name extraction error: {e}")

        if self.feature_names:
            logger.info(f"Feature names found: {len(self.feature_names)} features")
        else:
            logger.info("Feature names not determined; using positional indices.")

        # Initialize SHAP explainer (lazy)
        if SHAP_AVAILABLE:
            try:
                # Prefer to create explainer from final estimator if possible
                estimator = None
                if hasattr(self.model, "named_steps"):
                    # Try common names for estimator
                    for candidate in ("classifier", "estimator", "xgb", "model"):
                        candidate_obj = self.model.named_steps.get(candidate)
                        if candidate_obj is not None:
                            estimator = candidate_obj
                            break
                    # last step fallback
                    if estimator is None:
                        last_step_name = list(self.model.named_steps.keys())[-1]
                        estimator = self.model.named_steps.get(last_step_name)
                else:
                    estimator = self.model

                if estimator is not None:
                    # TreeExplainer for tree models often works best
                    try:
                        self.explainer = shap.Explainer(estimator)
                        logger.info("SHAP Explainer initialized (shap.Explainer)")
                    except Exception as e:
                        logger.debug(f"shap.Explainer failed: {e}")
                        try:
                            self.explainer = shap.TreeExplainer(estimator)
                            logger.info("SHAP TreeExplainer initialized")
                        except Exception as e2:
                            logger.warning(f"Could not initialize SHAP explainer: {e2}")
                else:
                    logger.debug("No estimator found for SHAP explainer.")
            except Exception as e:
                logger.warning(f"SHAP initialization error: {e}")

        self.is_loaded = True
        return True

    def _to_dataframe(self, features: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """Normalize input to a pandas DataFrame (single row or multiple)."""
        if isinstance(features, pd.DataFrame):
            df = features.copy()
        elif isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            # list of dicts or list of values
            if all(isinstance(x, dict) for x in features):
                df = pd.DataFrame(features)
            else:
                # treat as single-row list of values → create positional columns
                df = pd.DataFrame([features])
        else:
            raise ValueError("Unsupported features type. Pass dict, list, or pd.DataFrame.")
        return df

    def _align_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns expected by preprocessor exist. Fill missing numeric features with 0/mean (best-effort)."""
        # If we have a preprocessor that accepts raw columns directly, let pipeline handle it.
        # Otherwise attempt to ensure that DataFrame contains columns present in feature_names (if known).
        if self.feature_names:
            # If feature_names correspond to one-hot output, they might include encoded names.
            # We only add missing columns as zeros to preserve column count for transformers that expect exact names.
            missing = [c for c in self.feature_names if c not in df.columns]
            if missing:
                for c in missing:
                    df[c] = 0
            # Reorder to match feature_names
            try:
                df = df[self.feature_names]
            except Exception:
                # If reordering fails, just return df
                pass
        return df

    def predict(self,
                features: Union[Dict, List, pd.DataFrame],
                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict on one or many instances.

        Returns:
            preds: array of 0/1 predictions
            probs_pos: array of probabilities for positive class
            raw_prob_matrix: full predict_proba matrix (n x classes)
        """
        if not self.is_loaded and not self.load_model():
            raise RuntimeError("Model could not be loaded")

        df = self._to_dataframe(features)
        # Align but prefer to let pipeline handle raw column types - only fill when feature_names known
        df_aligned = self._align_and_fill(df)

        try:
            proba = self.model.predict_proba(df_aligned)
        except Exception as e:
            # Try passing df.values if pipeline expects array
            try:
                proba = self.model.predict_proba(df_aligned.values)
            except Exception as e2:
                logger.error(f"predict_proba failed: {e2}")
                raise

        # Optionally apply calibrator for better probabilities
        if self.calibrator is not None:
            try:
                proba = self.calibrator.predict_proba(df_aligned)
            except Exception as e:
                logger.debug(f"Calibrator apply failed: {e}")

        # handle binary or multiclass
        if proba.ndim == 1:
            # some models may return single-column positive prob
            pos = proba
            preds = (pos >= threshold).astype(int)
            raw = np.vstack([1 - pos, pos]).T
            return preds, pos, raw

        # assume last column is positive class for binary
        pos = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        preds = (pos >= threshold).astype(int)
        return preds, pos, proba

    def explain(self, features: Union[Dict, List, pd.DataFrame], top_k: int = 10) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Return SHAP-style explanations per instance (list of explanations). If SHAP unavailable, return None.

        Each explanation is a list of dicts: {"feature": name, "shap_value": float, "abs_shap": float}
        """
        if not self.is_loaded and not self.load_model():
            return None
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP not available or explainer not initialized.")
            return None

        df = self._to_dataframe(features)
        df_aligned = self._align_and_fill(df)

        try:
            # If our explainer expects raw features (preprocessor included), pass df_aligned
            shap_vals = self.explainer(df_aligned)
            # shap.Explanation object may contain values attribute shaped (n_samples, n_features) or multiple outputs
            # We will handle both possibilities
            out = []
            if hasattr(shap_vals, "values"):
                values = np.array(shap_vals.values)
                # if multi-output, take first output dimension
                if values.ndim == 3:
                    values = values[:, 0, :]
                for i in range(values.shape[0]):
                    row_vals = values[i]
                    # Feature names from explainer or self.feature_names
                    names = None
                    try:
                        names = list(shap_vals.feature_names)
                    except Exception:
                        names = self.feature_names or [f"feature_{j}" for j in range(len(row_vals))]
                    expl = []
                    for name, v in zip(names, row_vals):
                        expl.append({"feature": str(name), "shap_value": float(v), "abs_shap": abs(float(v))})
                    expl = sorted(expl, key=lambda x: x["abs_shap"], reverse=True)[:top_k]
                    out.append(expl)
                return out
            else:
                # Fallback if explainer returns arrays directly
                vals = np.array(self.explainer.shap_values(df_aligned))
                if vals.ndim == 2:
                    vals = vals
                elif vals.ndim == 3:
                    vals = vals[:, 0, :]
                out = []
                for i in range(vals.shape[0]):
                    row_vals = vals[i]
                    names = self.feature_names or [f"feature_{j}" for j in range(len(row_vals))]
                    expl = [{"feature": str(n), "shap_value": float(v), "abs_shap": abs(float(v))} for n, v in zip(names, row_vals)]
                    expl = sorted(expl, key=lambda x: x["abs_shap"], reverse=True)[:top_k]
                    out.append(expl)
                return out
        except Exception as e:
            logger.error(f"Error during SHAP explain: {e}")
            return None

    def check_robustness(self,
                         features: Union[Dict, List, pd.DataFrame],
                         preds: np.ndarray,
                         probs_pos: np.ndarray,
                         recent_predictions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Perform robustness checks for each instance and return a list of checks per input row.

        recent_predictions: list of dicts e.g. [{"probability":0.92, "prediction":1}, ...] (most recent last)
        """
        df = self._to_dataframe(features)
        df_aligned = self._align_and_fill(df)
        results = []

        # Convert recent predictions to numpy of probabilities if present
        recent_probs = None
        if recent_predictions:
            recent_probs = np.array([r.get("probability", np.nan) for r in recent_predictions if "probability" in r])

        for i in range(len(df_aligned)):
            row = df_aligned.iloc[[i]]
            prob = float(probs_pos[i])
            pred = int(preds[i])
            checks = {"confidence_low": False, "reasons": [], "probability_margin": abs(prob - 0.5)}

            # margin check
            if abs(prob - 0.5) < self.prob_margin_threshold:
                checks["confidence_low"] = True
                checks["reasons"].append("low_probability_margin")

            # entropy check
            # build 1xC prob vector; if multi-class scores available, entropy is higher for uncertain
            try:
                # if model can give full proba:
                full_proba = None
                if hasattr(self.model, "predict_proba"):
                    full_proba = self.model.predict_proba(row)[0]
                if full_proba is not None:
                    ent = entropy_from_probs(np.array(full_proba))
                    if ent > 0.9:  # tunable threshold: near max entropy
                        checks["confidence_low"] = True
                        checks["reasons"].append("high_entropy")
                else:
                    # fallback binary entropy
                    ent = entropy_from_probs(np.array([1 - prob, prob]))
                    if ent > 0.9:
                        checks["confidence_low"] = True
                        checks["reasons"].append("high_entropy")
            except Exception:
                pass

            # suspicious similarity check: compare to recent_probs last 10
            if recent_probs is not None and recent_probs.size > 0:
                last = recent_probs[-10:]
                similar_count = int(np.sum(np.isfinite(last) & (np.abs(last - prob) < 0.05)))
                if similar_count >= self.similarity_count_threshold:
                    checks["confidence_low"] = True
                    checks["reasons"].append("suspicious_similarity")

            # simple OOD numeric z-score check using train_stats if available
            if self.train_stats:
                for col, stats in self.train_stats.items():
                    if col in row.columns:
                        try:
                            val = float(row[col].values[0])
                            mean = float(stats.get("mean", 0.0))
                            std = float(stats.get("std", 1.0)) or 1.0
                            z = abs((val - mean) / std)
                            if z > self.ood_zscore_threshold:
                                checks["confidence_low"] = True
                                checks["reasons"].append(f"ood_zscore_{col}")
                        except Exception:
                            continue
            else:
                # fallback check: detect extreme magnitudes
                for col in row.columns:
                    try:
                        v = row[col].values[0]
                        if pd.isna(v) or (isinstance(v, (int, float)) and abs(v) > 1e7):
                            checks["confidence_low"] = True
                            checks["reasons"].append(f"out_of_distribution_{col}")
                    except Exception:
                        continue

            results.append(checks)
        return results
