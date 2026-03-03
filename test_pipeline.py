"""
test_pipeline.py — Smoke test for the transcription + topic analysis pipeline.

Bypasses live audio capture by feeding a synthetic .wav file directly through
the Transcriber and TopicAnalyzer, then verifies a markdown file is produced.

Run with:
    python test_pipeline.py

Requirements: faster-whisper and ollama must be installed and Ollama must be
running locally with at least one model pulled.
"""

import os
import queue
import struct
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_sine_wav(path: str, duration_s: float = 5.0, freq: float = 440.0, rate: int = 16000) -> None:
    """Write a simple sine-wave .wav file to disk."""
    n_samples = int(rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        pcm16 = (audio * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())


def _load_wav_as_float32(path: str) -> np.ndarray:
    with wave.open(path, "r") as wf:
        raw = wf.readframes(wf.getnframes())
        pcm16 = np.frombuffer(raw, dtype=np.int16)
    return pcm16.astype(np.float32) / 32767.0


# ── mock Ollama so the test works without a running Ollama server ─────────────

_MOCK_OLLAMA_RESPONSE = {
    "message": {"content": "NO"},
}
_MOCK_SUMMARY_RESPONSE = {
    "message": {"content": "- Key point one\n- Key point two\n- Key point three"},
}


def _mock_ollama_chat(**kwargs):
    prompt = kwargs.get("messages", [{}])[-1].get("content", "")
    if "Has the topic clearly shifted" in prompt:
        return _MOCK_OLLAMA_RESPONSE
    return _MOCK_SUMMARY_RESPONSE


# ── test ─────────────────────────────────────────────────────────────────────

def test_markdown_file_is_created():
    import config
    from transcriber import Transcriber
    from topic_analyzer import TopicAnalyzer
    from markdown_writer import MarkdownWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        config.OUTPUT_DIR = tmpdir
        config.WHISPER_MODEL = "tiny"     # fastest model for testing
        config.WHISPER_DEVICE = "cpu"
        config.TOPIC_MIN_SEGMENTS = 1
        config.TOPIC_WINDOW_SEGMENTS = 1
        config.OLLAMA_MODEL = "mistral"

        audio_q: queue.Queue = queue.Queue()
        transcript_q: queue.Queue = queue.Queue()
        session_start = time.time()
        session_dt = datetime.now()

        writer = MarkdownWriter(topics=[], lock=threading.Lock(), session_start=session_dt)
        analyzer = TopicAnalyzer(transcript_queue=transcript_q, on_update=writer.notify)
        writer._topics = analyzer.topics
        writer._lock = analyzer._lock
        transcriber = Transcriber(audio_queue=audio_q, transcript_queue=transcript_q, session_start=session_start)

        # Create and load a short sine-wave audio file
        wav_path = os.path.join(tmpdir, "test.wav")
        _make_sine_wav(wav_path, duration_s=3.0)
        audio_chunk = _load_wav_as_float32(wav_path)

        with patch("ollama.chat", side_effect=lambda **kw: _mock_ollama_chat(**kw)):
            writer.start()
            analyzer.start()
            transcriber.start()

            # Feed one audio chunk directly
            audio_q.put(audio_chunk)

            # Give pipeline time to process
            time.sleep(15)

            transcriber.stop()
            analyzer.stop()
            writer.stop()

        # Verify markdown file was created
        md_files = list(Path(tmpdir).glob("*.md"))
        assert md_files, "No markdown file was produced!"
        content = md_files[0].read_text(encoding="utf-8")
        assert "Course Transcript" in content, "Missing title in markdown output"
        assert "Introduction" in content or "##" in content, "Missing topic section"
        print(f"\n✅ Test passed! Markdown file: {md_files[0].name}")
        print("-" * 40)
        print(content[:500])
        print("-" * 40)


if __name__ == "__main__":
    test_markdown_file_is_created()
