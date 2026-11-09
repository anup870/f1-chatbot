"""Thin wrapper around the local GGUF model (llama-cpp-python).

Everything here degrades gracefully: with no model file present the bot still
answers stats questions from the database and falls back to extractive answers
for text questions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MAX_TOKENS = int(os.getenv("F1_MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("F1_TEMPERATURE", "0.1"))
N_CTX = int(os.getenv("F1_N_CTX", "4096"))


class LocalLLM:
    def __init__(self) -> None:
        self._llm = None
        self._tried = False
        self.last_error: Optional[str] = None

    @property
    def path(self) -> Optional[Path]:
        env = os.getenv("F1_MODEL_PATH", "").strip()
        if env:
            return Path(env)
        files = sorted(MODEL_DIR.glob("*.gguf"))
        return files[0] if files else None

    def available(self) -> bool:
        self._load()
        return self._llm is not None

    def _load(self) -> None:
        if self._llm is not None or self._tried:
            return
        self._tried = True
        path = self.path
        if not path or not path.exists():
            return
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(path),
                n_ctx=N_CTX,
                n_threads=max((os.cpu_count() or 2) - 1, 1),
                verbose=False,
            )
            logger.info("Loaded GGUF model from %s.", path)
        except Exception as exc:
            self.last_error = f"model_load_error: {exc}"
            logger.exception("Model initialization failed.")

    def chat(self, system: str, user: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> Optional[str]:
        self._load()
        if self._llm is None:
            return None
        try:
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            self.last_error = f"generation_error: {exc}"
            logger.exception("Generation failed.")
            return None

    def chat_json(self, system: str, user: str, max_tokens: int = 160) -> Optional[dict[str, Any]]:
        """Ask for a JSON object. Uses llama.cpp's JSON grammar when available."""
        self._load()
        if self._llm is None:
            return None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = None
        try:
            out = self._llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = self.chat(system, user, max_tokens=max_tokens, temperature=0.0)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
