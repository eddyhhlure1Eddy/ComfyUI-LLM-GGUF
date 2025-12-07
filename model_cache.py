# -*- coding: utf-8 -*-
# Author: eddy
# Model cache for LLM GGUF inference

import os
import re
import subprocess
import tempfile
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Callable

# Regex to strip ANSI escape codes
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Regex to strip thinking chain tags
THINK_CHAIN = re.compile(r'<think>.*?</think>', re.DOTALL)

# Try to import llama-cpp-python binding
try:
    from llama_cpp import Llama
    USE_BINDING = True
    logging.info("llama-cpp-python binding available, using native inference")
except ImportError:
    USE_BINDING = False
    logging.warning("llama-cpp-python not found, falling back to subprocess mode")


class SubprocessModel:
    """Fallback model wrapper using llama-cli.exe subprocess."""

    def __init__(self, model_path: str, llama_cli_path: str = None, **kwargs):
        self.model_path = model_path
        self.llama_cli_path = llama_cli_path or self._find_llama_cli()
        self.n_gpu_layers = kwargs.get("n_gpu_layers", 99)
        self.n_ctx = kwargs.get("n_ctx", 32768)

    def _find_llama_cli(self) -> str:
        """Find llama-cli.exe in common locations."""
        possible_paths = [
            r"C:\Users\Administrator\Desktop\222\llama.cpp\build\bin\llama-cli.exe",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama-cli.exe"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("llama-cli.exe not found. Please specify llama_cli_path.")

    def __call__(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 40, repeat_penalty: float = 1.1,
                 stream: bool = False, callback: Callable = None, **kwargs) -> str:
        """Run inference using subprocess."""
        # Write prompt to temp file
        fd, temp_file = tempfile.mkstemp(suffix=".txt", prefix="llm_prompt_")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(prompt)
        except Exception:
            os.close(fd)
            raise

        try:
            cmd = [
                self.llama_cli_path,
                "-m", self.model_path,
                "-f", temp_file,
                "-n", str(max_tokens),
                "-ngl", str(self.n_gpu_layers),
                "-c", str(self.n_ctx),
                "--temp", str(temperature),
                "--top-p", str(top_p),
                "--top-k", str(top_k),
                "--repeat-penalty", str(repeat_penalty),
                "--no-display-prompt",
                "-no-cnv",
                "-e",
            ]

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                startupinfo=startupinfo,
            )

            result = []
            buffer = b""

            while True:
                chunk = process.stdout.read(64)
                if not chunk:
                    break

                buffer += chunk
                try:
                    text = buffer.decode("utf-8")
                    buffer = b""

                    # Check for end token
                    if "<|im_end|>" in text:
                        text = text.split("<|im_end|>")[0]
                        result.append(text)
                        if callback:
                            callback(text)
                        break

                    result.append(text)
                    if callback:
                        callback(text)

                except UnicodeDecodeError:
                    for i in range(min(4, len(buffer)), 0, -1):
                        try:
                            text = buffer[:-i].decode("utf-8")
                            buffer = buffer[-i:]
                            result.append(text)
                            if callback:
                                callback(text)
                            break
                        except UnicodeDecodeError:
                            continue

            # Flush remaining buffer
            if buffer:
                try:
                    text = buffer.decode("utf-8", errors="replace")
                    if "<|im_end|>" not in text:
                        result.append(text)
                except Exception:
                    pass

            process.wait()
            output = "".join(result).strip()
            # Strip ANSI escape codes
            output = ANSI_ESCAPE.sub('', output)
            # Strip thinking chain
            output = THINK_CHAIN.sub('', output)
            # Strip end of text marker
            output = output.replace('[end of text]', '').strip()
            return output

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class ModelCache:
    """Singleton cache for loaded LLM models."""

    _instance = None
    _store: OrderedDict = None
    _max_items: int = 2  # Keep max 2 models in memory

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._store = OrderedDict()
        return cls._instance

    def get(self, model_path: str, llama_cli_path: str = None, **kwargs) -> Any:
        """Get or load a model from cache."""
        # Create cache key from path and important parameters
        key = (model_path, kwargs.get("n_gpu_layers", 99), kwargs.get("n_ctx", 32768))

        if key in self._store:
            # Move to end (LRU)
            self._store.move_to_end(key)
            logging.info(f"Using cached model: {model_path}")
            return self._store[key]

        # Load new model
        logging.info(f"Loading model: {model_path}")

        if USE_BINDING:
            model = Llama(
                model_path=model_path,
                n_gpu_layers=kwargs.get("n_gpu_layers", 99),
                n_ctx=kwargs.get("n_ctx", 32768),
                verbose=False,
            )
        else:
            model = SubprocessModel(
                model_path=model_path,
                llama_cli_path=llama_cli_path,
                **kwargs
            )

        self._store[key] = model

        # Enforce LRU limit
        while len(self._store) > self._max_items:
            old_key, old_model = self._store.popitem(last=False)
            logging.info(f"Evicting cached model: {old_key[0]}")
            # Clean up old model if possible
            if hasattr(old_model, "close"):
                old_model.close()
            del old_model

        return model

    def clear(self):
        """Clear all cached models."""
        for key, model in self._store.items():
            if hasattr(model, "close"):
                model.close()
        self._store.clear()
        logging.info("Model cache cleared")


# Global cache instance
model_cache = ModelCache()
