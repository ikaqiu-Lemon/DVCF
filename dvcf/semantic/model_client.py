"""Explicit, transport-independent model requests without provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable


def require_filled(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "<Please input your" in value:
        raise ValueError(f"{field} must be supplied by the caller.")
    return value


@dataclass(frozen=True)
class ModelParameters:
    model: str
    temperature: float
    max_output_tokens: int

    def __post_init__(self) -> None:
        require_filled(self.model, "model")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative.")
        if not isinstance(self.max_output_tokens, int) or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer.")


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    parameters: ModelParameters


@dataclass(frozen=True)
class ModelClient:
    """The caller supplies the transport and owns authentication outside this package."""

    transport: Callable[[ModelRequest], str]

    def complete(self, prompt: str, *, parameters: ModelParameters) -> str:
        require_filled(prompt, "prompt")
        response = self.transport(ModelRequest(prompt, parameters))
        if not isinstance(response, str) or not response.strip():
            raise ValueError("The model transport returned no textual response.")
        return response
