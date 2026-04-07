"""
Unified provider package for Macos-Use.

Each provider lives in its own sub-package (e.g. ``providers.google``)
and exposes all capabilities (LLM, STT, TTS) it supports.

Shared base protocols and data models:
    - ``BaseChatLLM``  — LLM provider protocol
    - ``BaseSTT``      — Speech-to-Text provider protocol
    - ``BaseTTS``      — Text-to-Speech provider protocol
    - ``TokenUsage``, ``Metadata`` — LLM data models
"""

# Base protocols & data models
from macos_use.providers.base import BaseChatLLM, BaseSTT, BaseTTS
from macos_use.providers.views import TokenUsage, Metadata
from macos_use.providers.events import Thinking, LLMEvent, LLMStreamEvent, ToolCall

# LLM providers
from macos_use.providers.anthropic import ChatAnthropic
from macos_use.providers.google import ChatGoogle
from macos_use.providers.openai import ChatOpenAI
from macos_use.providers.ollama import ChatOllama
from macos_use.providers.groq import ChatGroq
from macos_use.providers.mistral import ChatMistral
from macos_use.providers.cerebras import ChatCerebras
from macos_use.providers.open_router import ChatOpenRouter
from macos_use.providers.azure_openai import ChatAzureOpenAI
from macos_use.providers.litellm import ChatLiteLLM
from macos_use.providers.vllm import ChatVLLM
from macos_use.providers.nvidia import ChatNvidia
from macos_use.providers.deepseek import ChatDeepSeek

# STT providers
from macos_use.providers.openai import STTOpenAI
from macos_use.providers.google import STTGoogle
from macos_use.providers.groq import STTGroq
try:
    from macos_use.providers.elevenlabs import STTElevenLabs
except ImportError:
    pass

try:
    from macos_use.providers.deepgram import STTDeepgram
except ImportError:
    pass

# TTS providers
from macos_use.providers.openai import TTSOpenAI
from macos_use.providers.google import TTSGoogle
from macos_use.providers.groq import TTSGroq

try:
    from macos_use.providers.elevenlabs import TTSElevenLabs
except ImportError:
    pass

try:
    from macos_use.providers.deepgram import TTSDeepgram
except ImportError:
    pass

# Misc
from macos_use.providers.google.tts import GOOGLE_TTS_VOICES

__all__ = [
    # Base
    "BaseChatLLM",
    "BaseSTT",
    "BaseTTS",
    "TokenUsage",
    "Metadata",
    "Thinking",
    "LLMEvent",
    "LLMStreamEvent",
    "ToolCall",
    # LLM providers
    "ChatAnthropic",
    "ChatGoogle",
    "ChatOpenAI",
    "ChatOllama",
    "ChatGroq",
    "ChatMistral",
    "ChatCerebras",
    "ChatOpenRouter",
    "ChatAzureOpenAI",
    "ChatLiteLLM",
    "ChatVLLM",
    "ChatNvidia",
    "ChatDeepSeek",
    # STT providers
    "STTOpenAI",
    "STTGoogle",
    "STTGroq",
    "STTElevenLabs",
    "STTDeepgram",
    # TTS providers
    "TTSOpenAI",
    "TTSGoogle",
    "TTSGroq",
    "TTSElevenLabs",
    "TTSDeepgram",
    "GOOGLE_TTS_VOICES",
]
