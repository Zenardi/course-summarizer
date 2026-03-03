"""
pipeline.py — Wires all stages together and manages lifecycle.

Creates the thread-safe queues, instantiates all workers, and provides
a clean start/stop API. Handles graceful shutdown on SIGINT (Ctrl+C).
"""

import queue
import signal
import threading
import time
from datetime import datetime

import config
from audio_capture import AudioCapture
from transcriber import Transcriber
from topic_analyzer import TopicAnalyzer
from markdown_writer import MarkdownWriter


class Pipeline:
    """
    Orchestrates the full audio→transcript→topics→markdown pipeline.

    Usage:
        p = Pipeline()
        p.start()
        p.wait()   # blocks until Ctrl+C
    """

    def __init__(self, whisper_model: str | None = None, ollama_model: str | None = None, output_dir: str | None = None, lecture_title: str | None = None, module_name: str | None = None):
        # Allow CLI overrides
        if whisper_model:
            config.WHISPER_MODEL = whisper_model
        if ollama_model:
            config.OLLAMA_MODEL = ollama_model
        if output_dir:
            config.OUTPUT_DIR = output_dir

        self._lecture_title = lecture_title
        self._module_name = module_name

        self._session_start_dt = datetime.now()
        self._session_start_ts = time.time()

        # Thread-safe queues between stages
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self._transcript_queue: queue.Queue = queue.Queue(maxsize=100)

        # Markdown writer (needs to exist before TopicAnalyzer so we can pass the callback)
        self._writer = MarkdownWriter(
            topics=[],          # will be replaced after analyzer is created
            lock=threading.Lock(),
            session_start=self._session_start_dt,
            lecture_title=self._lecture_title,
            module_name=self._module_name,
        )

        self._analyzer = TopicAnalyzer(
            transcript_queue=self._transcript_queue,
            on_update=self._writer.notify,
            lecture_title=self._lecture_title,
            module_name=self._module_name,
        )

        # Wire the writer to the analyzer's shared topics list
        self._writer._topics = self._analyzer.topics
        self._writer._lock = self._analyzer._lock

        self._transcriber = Transcriber(
            audio_queue=self._audio_queue,
            transcript_queue=self._transcript_queue,
            session_start=self._session_start_ts,
        )

        self._capture = AudioCapture(audio_queue=self._audio_queue)

        self._running = False
        self._stop_event = threading.Event()

    def start(self) -> None:
        print("=" * 60)
        print("  course-summarizer starting…")
        if self._module_name:
            print(f"  Module        : {self._module_name}")
        if self._lecture_title:
            print(f"  Lecture       : {self._lecture_title}")
        print(f"  Whisper model : {config.WHISPER_MODEL}")
        print(f"  Ollama model  : {config.OLLAMA_MODEL}")
        print(f"  Output dir    : {config.OUTPUT_DIR}")
        print("  Press Ctrl+C to stop and finalize the file.")
        print("=" * 60)

        # Start in dependency order: writer first (passive), then analyzer,
        # then transcriber, then capture (source of data)
        self._writer.start()
        self._analyzer.start()
        self._transcriber.start()
        self._capture.start()
        self._running = True

        # Register signal handler for graceful Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

    def wait(self) -> None:
        """Block the main thread until a stop signal is received."""
        self._stop_event.wait()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        print("\n[Pipeline] Shutting down — flushing remaining audio…")
        self._capture.stop()
        self._transcriber.stop()
        self._analyzer.stop()
        self._writer.stop()
        print(f"[Pipeline] Done. Output saved to: {self._writer.output_path}")
        self._stop_event.set()

    def _handle_sigint(self, signum, frame) -> None:
        print()  # newline after ^C
        threading.Thread(target=self.stop, daemon=True).start()
