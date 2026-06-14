"""Application configuration from Streamlit secrets and environment variables."""

from __future__ import annotations

import os
from typing import Any, Optional


def _clean_secret_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    return text or None


def get_config(key: str, default: Optional[Any] = None) -> Any:
    """
    Read config from Streamlit secrets (cloud) then environment (local).

    Streamlit secrets are unavailable outside a running Streamlit app; env vars
    are used as fallback for scripts and tests.
    """
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            cleaned = _clean_secret_value(st.secrets[key])
            if cleaned is not None:
                return cleaned
    except Exception:
        pass

    env_val = _clean_secret_value(os.getenv(key))
    if env_val is not None:
        return env_val

    return default


def get_groq_api_key() -> Optional[str]:
    key = get_config("GROQ_API_KEY")
    if key and key != "your-groq-api-key-here":
        return key
    return None


def get_groq_model_name() -> str:
    return get_config("GROQ_MODEL_NAME", "llama-3.1-8b-instant")


def get_admin_password() -> str:
    return get_config("ADMIN_PASSWORD", "admin123")
