"""Search provider abstraction for deep research.

Provides a uniform interface so any configured search backend can be plugged
into the open_deep_research pipeline via the monkey-patch bridge in
deep_research.py.

Extending with a new provider:
    1. Add a class that implements SearchProvider (base.py)
    2. Register it in factory.py under its provider key
    3. Add the key to the SearchProvider literal type
"""

from .base import SearchProvider, SearchResult
from .factory import get_search_provider, PROVIDER_LABELS

__all__ = ["SearchProvider", "SearchResult", "get_search_provider", "PROVIDER_LABELS"]
