"""Research data collectors.

Each collector module exposes a ``collect() -> dict`` function and degrades
gracefully (empty dict / null fields) when its data source is unavailable.
"""

from research.collectors import fed_futures_collector

__all__ = ["fed_futures_collector"]
