from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseModelWrapper, ModelMetadata


class QwenModel(BaseModelWrapper):
    """Wrapper autour du modèle Qwen3 VL en s'appuyant sur vLLM."""

    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def __init__(
        self,
        cache_dir,
        hf_token=None,
        preferred_device_ids: list[int] | None = None,
    ):
        metadata = ModelMetadata(
            identifier=self.model_id,
            task="chat-completion",
            description="Qwen3 VL 30B A3B Instruct model for multimodal chat completions",
            format="chatml",
        )
        super().__init__(metadata, cache_dir, hf_token, preferred_device_ids)
        self.tokenizer = None
        self._engine = None
        self._visible_devices_backup: str | None = None

    async def load(self) -> None:
        def _load():
            import torch
            import json
            from transformers import AutoConfig, AutoTokenizer
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            if not torch.cuda.is_available():  # pragma: no cover - dépend du matériel
                raise RuntimeError("CUDA est requis pour charger Qwen avec vLLM")

            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            preferred = self.preferred_device_ids
            tensor_parallel = max(1, len(preferred)) if preferred else 1

            visible = ",".join(str(idx) for idx in preferred) if preferred else None
            previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            self._visible_devices_backup = previous_visible
            os.environ.setdefault("VLLM_USE_TRUST_REMOTE_CODE", "1")

            try:
                if visible:
                    os.environ["CUDA_VISIBLE_DEVICES"] = visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

                if auth_token:
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", auth_token)
                    os.environ.setdefault("HUGGINGFACE_TOKEN", auth_token)

                self.update_runtime(
                    progress=12,
                    status="Préparation du cluster GPU",
                    details={
                        "tensor_parallel": tensor_parallel,
                        "preferred_device_ids": self.preferred_device_ids,
                    },
                )

                download_root = self._download_repo(auth_token)
                model_path = Path(download_root)

                # Certains configs Qwen3 VL ne définissent pas explicitement
                # vocab_size. vLLM s'attend à ce champ et plante sinon, on
                # l'injecte donc depuis text_config ou le tokenizer si besoin.
                def _ensure_vocab_size() -> None:
                    config_path = model_path / "config.json"
                    if not config_path.exists():
                        return
                    try:
                        with config_path.open("r", encoding="utf-8") as f:
                            cfg_dict = json.load(f)

                        if "vocab_size" in cfg_dict:
                            return

                        hf_config = AutoConfig.from_pretrained(
                            str(model_path),
                            trust_remote_code=True,
                            use_auth_token=auth_token,
                        )
                        vocab_size = getattr(hf_config, "vocab_size", None)
                        if vocab_size is None and hasattr(hf_config, "text_config"):
                            vocab_size = getattr(hf_config.text_config, "vocab_size", None)

                        if vocab_size is None:
                            return

                        cfg_dict["vocab_size"] = int(vocab_size)
                        with config_path.open("w", encoding="utf-8") as f:
                            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
                    except Exception:
                        # On ne bloque pas le chargement si l'injection échoue
                        # (vLLM plantera plus tard, mais on préfère laisser
                        # l'erreur originale remonter).
                        return

                _ensure_vocab_size()
                self.update_runtime(progress=68, status="Chargement du tokenizer", downloaded=True)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    cache_dir=str(self.cache_dir),
                    use_auth_token=auth_token,
                    trust_remote_code=True,
                )

                engine_kwargs = dict(
                    model=str(model_path),
                    tokenizer=str(model_path),
                    tensor_parallel_size=tensor_parallel,
                    dtype="half",
                    download_dir=str(self.cache_dir),
                    trust_remote_code=True,
                    gpu_memory_utilization=0.90,
                    max_model_len=8192,
                    enforce_eager=True,
                )

                import inspect

                signature = inspect.signature(AsyncEngineArgs)
                if "worker_use_ray" in signature.parameters:
                    engine_kwargs["worker_use_ray"] = False

                engine_args = AsyncEngineArgs(**engine_kwargs)

                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.update_runtime(
                    progress=95,
                    status="Moteur vLLM initialisé",
                    server=self.build_server_metadata(
                        endpoint="/v1/chat/completions",
                        type="vLLM",
                        tensor_parallel=tensor_parallel,
                        visible_gpus=visible or "auto",
                        model_path=str(model_path),
                    ),
                )
            except Exception:
                if previous_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
                self._visible_devices_backup = previous_visible
                raise

        await asyncio.to_thread(_load)

    def _download_repo(self, auth_token: str | None) -> Path:
        return self.download_snapshot(
            repo_id=self.model_id,
            auth_token=auth_token,
            status_prefix="Téléchargement des poids Qwen",
            progress_range=(24, 60),
            complete_status="Poids Qwen synchronisés",
            local_dir_use_symlinks=False,
        )

    async def download(self) -> None:
        def _download():
            auth_token = self.hf_token or os.getenv("HUGGINGFACE_TOKEN")
            self.download_snapshot(
                repo_id=self.model_id,
                auth_token=auth_token,
                status_prefix="Téléchargement des poids Qwen",
                progress_range=(18, 96),
                complete_status="Poids Qwen disponibles hors ligne",
                local_dir_use_symlinks=False,
            )

        await asyncio.to_thread(_download)

    async def _unload(self) -> None:
        if self._engine is not None:
            await self._engine.shutdown()
        self._engine = None
        self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no branch - libère la VRAM
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch peut être déchargé lors de l'arrêt
            pass
        if self._visible_devices_backup is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._visible_devices_backup
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        self._visible_devices_backup = None

    async def infer(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        await self.ensure_loaded()

        if self._engine is None or self.tokenizer is None:
            raise RuntimeError("Le moteur vLLM n'est pas initialisé")

        prompt = self._build_prompt(messages)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
        )
        request_id = str(uuid.uuid4())
        outputs = await self._engine.generate(prompt, sampling_params, request_id=request_id)
        if not outputs:
            return {"content": ""}
        first = outputs[0]
        if not first.outputs:
            return {"content": ""}
        response = first.outputs[0].text.strip()
        return {"content": response}

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Le tokenizer de Qwen n'est pas initialisé")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
