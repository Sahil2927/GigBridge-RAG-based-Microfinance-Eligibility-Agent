"""Runtime setup helpers for local and cloud deployments."""

from src.setup.artifacts import bootstrap_if_needed, get_artifact_status

__all__ = ["bootstrap_if_needed", "get_artifact_status"]
