from __future__ import annotations

import logging
import random
import time

from typing import Any

from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from requests import exceptions as requests_exceptions
from urllib3 import exceptions as urllib3_exceptions

LOGGER = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    requests_exceptions.ConnectionError,
    requests_exceptions.Timeout,
    requests_exceptions.SSLError,
    requests_exceptions.ChunkedEncodingError,
    urllib3_exceptions.HTTPError,
)


def _should_retry_http_error(error: HfHubHTTPError) -> bool:
    status = getattr(error.response, "status_code", None)
    if status is None:
        return False
    # Retry only on transient 5xx errors
    return 500 <= status < 600


def snapshot_download_with_retry(
    *,
    max_attempts: int = 5,
    base_sleep: float = 1.5,
    jitter: float = 0.5,
    **kwargs: Any,
) -> str:
    """Wrapper around :func:`huggingface_hub.snapshot_download` with retries.

    Some of the larger checkpoints occasionally fail with abrupt TLS errors when
    being streamed from the Hugging Face CDN. This helper retries the download
    with an exponential backoff so that transient network glitches do not abort
    the whole model setup.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    # Always resume partially completed downloads to avoid restarting from 0.
    kwargs.setdefault("resume_download", True)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return hf_snapshot_download(**kwargs)
        except HfHubHTTPError as error:
            last_error = error
            if not _should_retry_http_error(error) or attempt == max_attempts:
                raise
            sleep_for = base_sleep * (2 ** (attempt - 1))
            sleep_for += random.uniform(0, jitter)
            LOGGER.warning(
                "Hugging Face download failed with HTTP %s (attempt %s/%s) for %s: %s",
                getattr(error.response, "status_code", "?"),
                attempt,
                max_attempts,
                kwargs.get("repo_id"),
                error,
            )
            time.sleep(sleep_for)
        except _RETRYABLE_EXCEPTIONS as error:
            last_error = error
            if attempt == max_attempts:
                raise
            sleep_for = base_sleep * (2 ** (attempt - 1))
            sleep_for += random.uniform(0, jitter)
            LOGGER.warning(
                "Transient error while downloading %s (attempt %s/%s): %s",
                kwargs.get("repo_id"),
                attempt,
                max_attempts,
                error,
            )
            time.sleep(sleep_for)
    # Exhausted attempts
    assert last_error is not None  # for mypy
    raise last_error


__all__ = ["snapshot_download_with_retry"]

