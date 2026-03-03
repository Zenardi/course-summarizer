"""
Central configuration for course-summarizer.
All values here can be overridden via CLI flags in main.py.
"""

# ── Whisper transcription ────────────────────────────────────────────────────
# Options: "tiny", "base", "small", "medium", "large-v3"
# tiny/base = fastest, small = good balance, medium/large = most accurate
WHISPER_MODEL: str = "small"

# "auto" uses GPU if available, falls back to CPU
WHISPER_DEVICE: str = "auto"

# ── Audio capture ────────────────────────────────────────────────────────────
# Seconds of audio per chunk sent to Whisper
AUDIO_CHUNK_SECONDS: int = 30

# Sample rate expected by Whisper (do not change)
SAMPLE_RATE: int = 16000

# RMS amplitude below which a chunk is considered silence and skipped
SILENCE_THRESHOLD: float = 0.001

# ── Ollama / topic analysis ──────────────────────────────────────────────────
# Must match a model you have pulled locally: `ollama pull mistral`
OLLAMA_MODEL: str = "mistral"

# Ollama server URL (default local install)
OLLAMA_HOST: str = "http://localhost:11434"

# How many transcript segments (chunks) to include in topic-detection prompt
TOPIC_WINDOW_SEGMENTS: int = 5

# Minimum segments that must accumulate before topic detection runs
TOPIC_MIN_SEGMENTS: int = 3

# ── Output ───────────────────────────────────────────────────────────────────
# Directory where .md files are written (created if missing)
OUTPUT_DIR: str = "output"
