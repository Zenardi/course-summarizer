"""
Central configuration for course-summarizer.
All values here can be overridden via CLI flags in main.py.
"""

# ── Whisper transcription ────────────────────────────────────────────────────
# Options: "tiny", "base", "small", "medium", "large-v3"
# tiny/base = fastest, small = good balance, medium/large = most accurate
# For short Udacity-style lessons with a CUDA GPU: "large-v3" gives the best
# transcript quality with ~10s processing time per 45s chunk (well within real time).
# Without GPU, use "medium" instead.
WHISPER_MODEL: str = "large-v3"

# "auto" uses GPU if available, falls back to CPU
WHISPER_DEVICE: str = "auto"

# ── Audio capture ────────────────────────────────────────────────────────────
# Seconds of audio per chunk sent to Whisper.
# Longer chunks = more sentence context for Whisper = better accuracy.
# 45s is the sweet spot for short lessons: enough context without too much latency.
AUDIO_CHUNK_SECONDS: int = 45

# Sample rate expected by Whisper (do not change)
SAMPLE_RATE: int = 16000

# RMS amplitude below which a chunk is considered silence and skipped
SILENCE_THRESHOLD: float = 0.001

# ── Ollama / topic analysis ──────────────────────────────────────────────────
# Must match a model you have pulled locally: `ollama pull llama3.1:8b`
# llama3.1:8b is recommended for technical courses (AI, Kubernetes, DevOps):
# strong domain vocabulary, reliable structured output, fits in 8 GB VRAM (~5 GB Q4).
OLLAMA_MODEL: str = "llama3.1:8b"

# Ollama server URL (default local install)
OLLAMA_HOST: str = "http://localhost:11434"

# How many transcript segments to include in the topic-detection prompt.
# Higher = more context for Ollama = fewer false topic splits.
# For short lessons with 1-3 topics, use a large window to be conservative.
TOPIC_WINDOW_SEGMENTS: int = 10

# Minimum segments that must accumulate before topic detection even runs.
# For short lessons, wait for a solid block of content before considering a split.
TOPIC_MIN_SEGMENTS: int = 8

# ── Output ───────────────────────────────────────────────────────────────────
# Directory where .md files are written (created if missing)
OUTPUT_DIR: str = "output"
