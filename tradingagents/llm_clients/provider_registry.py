"""Provider registry: single source of truth for provider metadata.

Stub created by Phase 1. Full implementation arrives in Phase 2 (Task 7).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Type


@dataclass(frozen=True)
class ProviderConfig:
    """Metadata for a single provider."""
    name: str
    client_class: type
    base_url: Optional[str]
    api_key_env: Optional[str]


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {}
