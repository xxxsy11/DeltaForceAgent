"""本地 Qwen 推理运行时（支持按需切换 LoRA 适配器）。"""

from __future__ import annotations

import gc
import logging
import os
import re
import threading
import atexit
import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - 运行环境可能未安装 peft
    PeftModel = None


def _normalize_device(device: str) -> str:
    value = str(device or "cpu").strip().lower()
    if value in {"cuda", "gpu"}:
        return "cuda"
    return value


@dataclass
class _RuntimeKey:
    base_model_path: str
    device: str
    max_new_tokens: int
    force_no_think: bool


class _LocalQwenRuntime:
    """单例运行时：同一 base model 只加载一次，适配器按需切换。"""

    DEFAULT_MAX_INSTANCES = 4
    _instances: Dict[Tuple[str, str, int, bool], "_LocalQwenRuntime"] = {}
    _instances_lock = threading.Lock()
    _max_instances = DEFAULT_MAX_INSTANCES

    def __init__(self, key: _RuntimeKey):
        self.key = key
        self._lock = threading.Lock()
        self._tokenizer = None
        self._base_model = None
        self._active_model = None
        self._active_adapter = ""

    @classmethod
    def get_or_create(
        cls,
        base_model_path: str,
        device: str,
        max_new_tokens: int,
        force_no_think: bool,
    ) -> "_LocalQwenRuntime":
        key = (str(base_model_path), _normalize_device(device), int(max_new_tokens), bool(force_no_think))
        with cls._instances_lock:
            runtime = cls._instances.get(key)
            if runtime is None:
                # 防止频繁切换配置造成运行时缓存无上界增长。
                if len(cls._instances) >= cls._max_instances:
                    evict_key, evict_runtime = next(iter(cls._instances.items()))
                    try:
                        evict_runtime._release()
                    finally:
                        cls._instances.pop(evict_key, None)
                runtime = _LocalQwenRuntime(
                    _RuntimeKey(
                        base_model_path=str(base_model_path),
                        device=_normalize_device(device),
                        max_new_tokens=int(max_new_tokens),
                        force_no_think=bool(force_no_think),
                    )
                )
                cls._instances[key] = runtime
            return runtime

    @classmethod
    def clear_all(cls):
        with cls._instances_lock:
            instances = list(cls._instances.values())
            cls._instances.clear()
        for runtime in instances:
            runtime._release()

    def _release(self):
        with self._lock:
            if self._active_model is not None and self._active_model is not self._base_model:
                del self._active_model
            self._active_model = None
            if self._base_model is not None:
                del self._base_model
            self._base_model = None
            self._tokenizer = None
            self._active_adapter = ""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _ensure_base_loaded(self):
        if self._base_model is not None and self._tokenizer is not None:
            return
        logger.info("local_qwen: loading base model from %s on %s", self.key.base_model_path, self.key.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.key.base_model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.key.device == "cpu":
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = None
        else:
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        self._base_model = AutoModelForCausalLM.from_pretrained(self.key.base_model_path, **model_kwargs)
        if self.key.device == "cpu":
            self._base_model = self._base_model.to("cpu")
        self._base_model.eval()
        self._active_model = self._base_model
        self._active_adapter = ""

    def _switch_adapter(self, adapter_path: str):
        normalized = str(adapter_path or "").strip()
        if normalized == self._active_adapter:
            return

        if self._active_model is not None and self._active_model is not self._base_model:
            del self._active_model
            self._active_model = None
            gc.collect()
            # 清理 base 上可能残留的 peft 标记，避免重复切换时报“multiple adapters”告警。
            try:
                if hasattr(self._base_model, "peft_config"):
                    delattr(self._base_model, "peft_config")
            except Exception:
                pass
            try:
                if hasattr(self._base_model, "_hf_peft_config_loaded"):
                    setattr(self._base_model, "_hf_peft_config_loaded", False)
            except Exception:
                pass

        if not normalized:
            self._active_model = self._base_model
            self._active_adapter = ""
            return

        if not os.path.isdir(normalized):
            logger.warning("local_qwen: adapter path not found, fallback to base model: %s", normalized)
            self._active_model = self._base_model
            self._active_adapter = ""
            return
        if not os.path.isfile(os.path.join(normalized, "adapter_config.json")):
            logger.warning("local_qwen: adapter files incomplete, fallback to base model: %s", normalized)
            self._active_model = self._base_model
            self._active_adapter = ""
            return
        if PeftModel is None:
            logger.warning("local_qwen: peft unavailable, fallback to base model for adapter: %s", normalized)
            self._active_model = self._base_model
            self._active_adapter = ""
            return

        logger.info("local_qwen: switching adapter -> %s", normalized)
        self._active_model = PeftModel.from_pretrained(self._base_model, normalized, is_trainable=False)
        if self.key.device == "cpu":
            self._active_model = self._active_model.to("cpu")
        self._active_model.eval()
        self._active_adapter = normalized

    def generate(self, prompt: str, adapter_path: str = "", max_new_tokens: Optional[int] = None) -> str:
        with self._lock:
            self._ensure_base_loaded()
            self._switch_adapter(adapter_path)

            user_prompt = str(prompt or "")
            if self.key.force_no_think:
                lowered = user_prompt.lstrip().lower()
                if not lowered.startswith("/no_think") and not lowered.startswith("/think"):
                    user_prompt = f"/no_think\n{user_prompt}"
            messages = [{"role": "user", "content": user_prompt}]
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=(not self.key.force_no_think),
                )
            except Exception:
                try:
                    text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = str(prompt or "")

            inputs = self._tokenizer(text, return_tensors="pt")
            if self.key.device == "cpu":
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self._active_model.device) for k, v in inputs.items()}

            generation_config = self._active_model.generation_config
            generation_config.do_sample = False
            for key in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, key):
                    try:
                        setattr(generation_config, key, None)
                    except Exception:
                        pass

            with torch.no_grad():
                output_ids = self._active_model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens or self.key.max_new_tokens),
                    do_sample=False,
                    generation_config=generation_config,
                )
            gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            decoded = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            decoded = re.sub(r"<think>[\s\S]*?</think>\s*", "", decoded).strip()
            return decoded


class LocalQwenChatModel:
    """提供与 `ChatOpenAI.invoke` 兼容的最小接口。"""

    DEFAULT_MAX_NEW_TOKENS = 384

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str = "",
        device: str = "cpu",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        force_no_think: bool = True,
    ):
        self.base_model_path = str(base_model_path or "").strip()
        self.adapter_path = str(adapter_path or "").strip()
        self.device = _normalize_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.force_no_think = bool(force_no_think)
        self._runtime = _LocalQwenRuntime.get_or_create(
            base_model_path=self.base_model_path,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            force_no_think=self.force_no_think,
        )

    def invoke(self, prompt: str):
        text = self._runtime.generate(
            prompt=prompt,
            adapter_path=self.adapter_path,
            max_new_tokens=self.max_new_tokens,
        )
        return SimpleNamespace(content=text)

    async def ainvoke(self, prompt: str):
        return await asyncio.to_thread(self.invoke, prompt)


atexit.register(_LocalQwenRuntime.clear_all)
