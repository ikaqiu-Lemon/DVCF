"""Statistical graph interfaces; optional discovery backends load only on use."""

from .model import SourceSpec, discover_statistical_graphs
from .profile import DataProfile, resolve_profile

__all__ = ["DataProfile", "SourceSpec", "resolve_profile", "discover_statistical_graphs"]
