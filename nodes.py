# -*- coding: utf-8 -*-
# Author: eddy
# ComfyUI nodes for LLM GGUF inference

import logging
import folder_paths

from .model_cache import model_cache, USE_BINDING

# Register LLM folder for GGUF models
LLM_FOLDER = "LLM"
if LLM_FOLDER not in folder_paths.folder_names_and_paths:
    llm_path = folder_paths.models_dir + "/LLM"
    folder_paths.folder_names_and_paths[LLM_FOLDER] = ([llm_path], {".gguf"})


# Global session storage (shared across all LLMChat instances)
_chat_sessions = {}


class LoadGGUFModel:
    """Load a GGUF LLM model for inference."""

    @classmethod
    def INPUT_TYPES(cls):
        model_list = folder_paths.get_filename_list(LLM_FOLDER)
        return {
            "required": {
                "model_name": (model_list, {"tooltip": "Select GGUF model file"}),
                "gpu_layers": ("INT", {
                    "default": 99,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": "Number of layers to offload to GPU (99 = all)"
                }),
                "context_size": ("INT", {
                    "default": 32768,
                    "min": 512,
                    "max": 131072,
                    "step": 512,
                    "tooltip": "Context window size in tokens"
                }),
            },
            "optional": {
                "llama_cli_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to llama-cli.exe (for subprocess fallback)"
                }),
            }
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LLM"
    TITLE = "Load GGUF Model"

    def load_model(self, model_name: str, gpu_layers: int = 99,
                   context_size: int = 32768, llama_cli_path: str = ""):
        model_path = folder_paths.get_full_path(LLM_FOLDER, model_name)

        model = model_cache.get(
            model_path=model_path,
            llama_cli_path=llama_cli_path if llama_cli_path else None,
            n_gpu_layers=gpu_layers,
            n_ctx=context_size,
        )

        return (model,)


class LLMChat:
    """Run LLM inference with chat support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLM_MODEL", {"tooltip": "LLM model from LoadGGUFModel"}),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User message to send to the model"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System prompt to set model behavior"
                }),
                "session_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Session ID for maintaining conversation history"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Maximum tokens to generate"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature (higher = more creative)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p sampling threshold"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-k sampling"
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Repetition penalty"
                }),
                "keep_history": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep conversation history for hot session"
                }),
                "output_condition": (["None", "Chinese_T5"], {
                    "default": "None",
                    "tooltip": "Output condition prefix (Chinese_T5: output in Chinese natural language T5 format)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "LLM"
    TITLE = "LLM Chat"

    def _build_prompt(self, system_prompt: str, history: list, user_prompt: str) -> str:
        """Build prompt using Qwen chat template."""
        parts = []

        # Add system prompt
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # Add conversation history
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add current user message
        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")

        # Add assistant start tag
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def chat(self, model, user_prompt: str, system_prompt: str = "",
             session_id: str = "default", max_tokens: int = 1024, temperature: float = 0.7,
             top_p: float = 0.9, top_k: int = 40, repeat_penalty: float = 1.1,
             keep_history: bool = True, output_condition: str = "None"):

        # Apply output condition prefix
        if output_condition == "Chinese_T5":
            user_prompt = "Do not write tags, use Chinese natural language T5 output.\n\n" + user_prompt

        # Get or create session history
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = []

        history = _chat_sessions[session_id] if keep_history else []

        # Build prompt
        prompt = self._build_prompt(system_prompt, history, user_prompt)

        # Run inference
        if USE_BINDING:
            # Use llama-cpp-python binding
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
            if isinstance(output, dict):
                response = output.get("choices", [{}])[0].get("text", "").strip()
            else:
                response = str(output).strip()
        else:
            # Use subprocess fallback
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
            )

        # Update history
        if keep_history:
            _chat_sessions[session_id].append({"role": "user", "content": user_prompt})
            _chat_sessions[session_id].append({"role": "assistant", "content": response})

        return (response,)


class ResetChat:
    """Reset chat session history."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Session ID to reset"
                }),
            },
            "optional": {
                "trigger": ("*", {"tooltip": "Connect any input to trigger reset"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "message")
    FUNCTION = "reset"
    CATEGORY = "LLM"
    TITLE = "Reset Chat Session"

    def reset(self, session_id: str = "default", trigger=None):
        if session_id in _chat_sessions:
            del _chat_sessions[session_id]
            return (True, f"Session '{session_id}' cleared")
        return (False, f"Session '{session_id}' not found")


class ClearModelCache:
    """Clear all cached LLM models to free memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Connect any input to trigger cache clear"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "message")
    FUNCTION = "clear"
    CATEGORY = "LLM"
    TITLE = "Clear Model Cache"

    def clear(self, trigger=None):
        model_cache.clear()
        return (True, "Model cache cleared")


class ListSessions:
    """List all active chat sessions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sessions",)
    FUNCTION = "list_sessions"
    CATEGORY = "LLM"
    TITLE = "List Chat Sessions"

    def list_sessions(self):
        if not _chat_sessions:
            return ("No active sessions",)

        lines = []
        for sid, history in _chat_sessions.items():
            lines.append(f"Session '{sid}': {len(history)} messages")
        return ("\n".join(lines),)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LoadGGUFModel": LoadGGUFModel,
    "LLMChat": LLMChat,
    "ResetChat": ResetChat,
    "ClearModelCache": ClearModelCache,
    "ListSessions": ListSessions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGGUFModel": "Load GGUF Model",
    "LLMChat": "LLM Chat",
    "ResetChat": "Reset Chat Session",
    "ClearModelCache": "Clear Model Cache",
    "ListSessions": "List Chat Sessions",
}
