from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

if "backend.app.config" not in sys.modules:
    config_stub = types.ModuleType("backend.app.config")
    config_stub.settings = types.SimpleNamespace(
        api_host="127.0.0.1",
        api_port=8000,
        log_level="info",
        log_dir=Path(tempfile.gettempdir()),
        log_file_name="backend.log",
        log_max_bytes=1_048_576,
        log_backup_count=1,
        huggingface_token=None,
        model_cache_dir=Path(tempfile.gettempdir()),
        cors_origins=["*"],
        openai_api_keys=[],
        lazy_load_models=True,
        frontend_dist=Path("."),
    )
    sys.modules["backend.app.config"] = config_stub

from backend.app.models.base import BaseModelWrapper, ModelMetadata
from backend.app.services.model_registry import ModelRegistry


class DummyModel(BaseModelWrapper):
    """Minimal asynchronous model used for registry unit tests."""

    def __init__(self, metadata: ModelMetadata, cache_dir: Path) -> None:
        self._downloaded = False
        super().__init__(metadata, cache_dir, hf_token=None, preferred_device_ids=None)
        self.download_calls = 0
        self.load_calls = 0
        self.unload_calls = 0

    def is_downloaded(self) -> bool:  # type: ignore[override]
        return self._downloaded

    async def load(self) -> None:  # type: ignore[override]
        self.load_calls += 1
        self._downloaded = True
        self.update_runtime(
            state="ready",
            status="Dummy ready",
            progress=100,
            downloaded=True,
        )
        await asyncio.sleep(0)

    async def download(self) -> None:  # type: ignore[override]
        self.download_calls += 1
        self._downloaded = True
        self.update_runtime(status="Dummy cached", progress=80, downloaded=True)
        await asyncio.sleep(0)

    async def _unload(self) -> None:  # type: ignore[override]
        self.unload_calls += 1
        await asyncio.sleep(0)

    async def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {"ok": True}


class ModelRegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="registry-tests-"))
        self._env_backup = {
            key: os.environ.get(key)
            for key in ("HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        }

    async def asyncSetUp(self) -> None:
        self.registry = ModelRegistry()
        self.registry.configure(hf_token=None, cache_dir=self.tmpdir, with_defaults=False)

    async def asyncTearDown(self) -> None:
        await self.registry.shutdown()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _register_dummy(self, key: str = "dummy", task: str = "unit-test") -> ModelMetadata:
        metadata = ModelMetadata(
            identifier=f"local/{key}",
            task=task,
            description="Dummy model used for unit testing",
            format="raw",
        )
        self.registry.register(
            key,
            metadata=metadata,
            factory=lambda: DummyModel(metadata, self.tmpdir),
        )
        return metadata

    async def test_register_and_status_flow(self) -> None:
        metadata = self._register_dummy()
        await self.registry.ensure_downloaded("dummy")
        await self.registry.ensure_loaded("dummy", device_ids=[1])
        status = await self.registry.status()
        self.assertIn("dummy", status)
        entry = status["dummy"]
        self.assertTrue(entry.loaded)
        self.assertEqual(entry.params["device_ids"], [1])
        self.assertEqual(entry.identifier, metadata.identifier)
        self.assertTrue(entry.runtime["downloaded"])

        model = await self.registry.get("dummy")
        self.assertEqual(model.download_calls, 1)
        self.assertEqual(model.load_calls, 1)

        await self.registry.unload("dummy")
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.unload_calls, 1)

    async def test_get_by_task(self) -> None:
        metadata = self._register_dummy(task="text")
        model = await self.registry.get_by_task("text")
        self.assertIsInstance(model, DummyModel)
        self.assertEqual(model.metadata.identifier, metadata.identifier)

    async def test_set_hf_token_updates_environment(self) -> None:
        self._register_dummy()
        await self.registry.set_hf_token("test-token")
        self.assertEqual(os.environ.get("HUGGINGFACE_TOKEN"), "test-token")
        self.assertEqual(os.environ.get("HUGGING_FACE_HUB_TOKEN"), "test-token")

        await self.registry.set_hf_token(None)
        self.assertIsNone(os.environ.get("HUGGINGFACE_TOKEN"))
        self.assertIsNone(os.environ.get("HUGGING_FACE_HUB_TOKEN"))

    async def test_duplicate_registration_raises(self) -> None:
        self._register_dummy()
        metadata = ModelMetadata(
            identifier="local/dummy",
            task="unit-test",
            description="Duplicate dummy",
            format="raw",
        )
        with self.assertRaises(ValueError):
            self.registry.register(
                "dummy",
                metadata=metadata,
                factory=lambda: DummyModel(metadata, self.tmpdir),
            )


if __name__ == "__main__":
    unittest.main()
