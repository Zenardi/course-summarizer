"""
transcriber.py — Chunked transcription using faster-whisper.

Consumes raw float32 PCM arrays from audio_queue, transcribes each chunk,
and pushes timestamped TranscriptSegment objects to transcript_queue.
"""

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

import config


@dataclass
class TranscriptSegment:
    """A single transcribed chunk with its wall-clock timestamp."""
    text: str
    start_time: float        # seconds since session start
    chunk_index: int


def _load_model() -> WhisperModel:
    device = config.WHISPER_DEVICE
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[Transcriber] Loading Whisper '{config.WHISPER_MODEL}' on {device} ({compute_type})")
    return WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type)


class Transcriber:
    """
    Reads PCM audio chunks from audio_queue, transcribes them with Whisper,
    and puts TranscriptSegment objects into transcript_queue.

    Usage:
        t = Transcriber(audio_queue, transcript_queue, session_start)
        t.start()
        ...
        t.stop()
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        transcript_queue: queue.Queue,
        session_start: float,
    ):
        self._audio_q = audio_queue
        self._transcript_q = transcript_queue
        self._session_start = session_start
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._chunk_index = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="Transcriber")
        self._thread.start()

    def stop(self) -> None:
        """Signal stop and wait for the current transcription to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=120)

    def _run(self) -> None:
        model = _load_model()
        print("[Transcriber] Ready.")

        while not self._stop_event.is_set():
            try:
                chunk: np.ndarray = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk_start = time.time() - self._session_start
            text = self._transcribe(model, chunk)
            if text:
                seg = TranscriptSegment(
                    text=text,
                    start_time=chunk_start,
                    chunk_index=self._chunk_index,
                )
                self._transcript_q.put(seg)
                print(f"[Transcriber] [{_fmt_time(chunk_start)}] {text[:80]}{'…' if len(text) > 80 else ''}")
            self._chunk_index += 1

        # Drain remaining audio after stop signal
        while not self._audio_q.empty():
            try:
                chunk = self._audio_q.get_nowait()
                chunk_start = time.time() - self._session_start
                text = self._transcribe(model, chunk)
                if text:
                    self._transcript_q.put(
                        TranscriptSegment(text=text, start_time=chunk_start, chunk_index=self._chunk_index)
                    )
                self._chunk_index += 1
            except queue.Empty:
                break

        print("[Transcriber] Stopped.")

    @staticmethod
    def _transcribe(model: WhisperModel, audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, beam_size=5, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
