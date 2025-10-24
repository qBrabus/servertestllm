from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional


class TokenStore:
    """Simple file-based persistence for the Hugging Face token."""

    def __init__(self, initial_path: Path) -> None:
        self._lock = threading.Lock()
        self._path = initial_path
        self._ensure_directory()

    def configure(self, cache_dir: Path) -> None:
        with self._lock:
            self._path = cache_dir / ".hf_token"
            self._ensure_directory()

    def _ensure_directory(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[str]:
        with self._lock:
            if not self._path.exists():
                return None
            value = self._path.read_text(encoding="utf-8").strip()
            return value or None

    def save(self, token: str) -> None:
        with self._lock:
            self._path.write_text(token.strip(), encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def has_token(self) -> bool:
        return self.load() is not None


# The store is initialised with a temporary default path. It will be
# reconfigured during application startup once the cache directory is known.
token_store = TokenStore(Path("/models/.hf_token"))
