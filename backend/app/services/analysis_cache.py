from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from time import monotonic
from typing import Any

from ..config import ANALYSIS_CACHE_MAX_SIZE, ANALYSIS_CACHE_TTL_SECONDS


class AnalysisCacheService:
    def __init__(
        self,
        max_size: int = ANALYSIS_CACHE_MAX_SIZE,
        ttl_seconds: int = ANALYSIS_CACHE_TTL_SECONDS,
    ) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._store: OrderedDict[tuple[str, int, bool, bool], dict[str, Any]] = OrderedDict()

    def _is_expired(self, created_at: float) -> bool:
        return (monotonic() - created_at) > self.ttl_seconds

    def _prune_expired(self) -> None:
        expired_keys = [
            key for key, value in self._store.items() if self._is_expired(float(value["created_at"]))
        ]
        for key in expired_keys:
            self._store.pop(key, None)

    def get(self, key: tuple[str, int, bool, bool]) -> dict[str, Any] | None:
        self._prune_expired()
        item = self._store.get(key)
        if item is None:
            return None
        if self._is_expired(float(item["created_at"])):
            self._store.pop(key, None)
            return None

        self._store.move_to_end(key)
        return {
            "data": deepcopy(item["data"]),
            "warnings": list(item["warnings"]),
        }

    def set(self, key: tuple[str, int, bool, bool], data: dict[str, Any], warnings: list[str]) -> None:
        self._prune_expired()
        if key in self._store:
            self._store.move_to_end(key)

        self._store[key] = {
            "created_at": monotonic(),
            "data": deepcopy(data),
            "warnings": list(warnings),
        }

        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
