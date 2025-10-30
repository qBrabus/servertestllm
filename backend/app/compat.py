"""Compatibility shims for third-party dependencies.

This module centralises small runtime patches that keep external libraries
working smoothly in the long-lived Docker image. They are imported as early as
possible from :mod:`app.main` so that the patches take effect before any heavy
initialisation occurs.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def _initialise_tqdm_lock() -> None:
    """Ensure ``tqdm`` exposes a shared lock when used from multiple threads.

    Newer releases of ``tqdm`` lazily create their write lock. When the progress
    bar is only accessed through :mod:`tqdm.contrib.concurrent`, the lock may
    never be initialised, leading to ``AttributeError: type object 'tqdm' has no
    attribute '_lock'`` during model downloads. Touching ``tqdm.get_lock()``
    up-front guarantees that the attribute exists before any concurrent
    download starts.
    """

    try:
        from tqdm import tqdm

        if not hasattr(tqdm, "_lock"):
            tqdm.get_lock()
    except Exception:  # pragma: no cover - defensive, tqdm is third-party
        LOGGER.debug("Unable to prime tqdm lock", exc_info=True)


def apply_runtime_fixes() -> None:
    """Apply all available compatibility adjustments."""

    _initialise_tqdm_lock()


__all__ = ["apply_runtime_fixes"]

