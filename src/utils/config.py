"""Application configuration from Streamlit secrets and environment variables."""

from __future__ import annotations

import os
from typing import Any, Optional


def get_config(key: str, default: Optional[Any] = None) -> Any:
    """
    Read config from Streamlit secrets (cloud) then environment (local).

    Streamlit secrets are unavailable outside a running Streamlit app; env vars
    are used as fallback for scripts and tests.
    """
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


def get_groq_api_key() -> Optional[str]:
    key = get_config("GROQ_API_KEY")
    if key and key != "your-groq-api-key-here":
        return key
    return None


def get_groq_model_name() -> str:
    return get_config("GROQ_MODEL_NAME", "mixtral-8x7b-32768")


def get_admin_password() -> str:
    return get_config("ADMIN_PASSWORD", "admin123")
