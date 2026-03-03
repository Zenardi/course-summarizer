"""
topic_analyzer.py — Topic detection and summarization via Ollama.

Consumes TranscriptSegment objects from transcript_queue.
Maintains a rolling window of recent segments and periodically asks the
local Ollama LLM whether the topic has shifted. When a shift is detected,
it summarizes the completed topic section. Updates a shared topics list
that the MarkdownWriter watches.
"""

import queue
import threading
from dataclasses import dataclass, field

import ollama

import config
from transcriber import TranscriptSegment, _fmt_time


@dataclass
class TopicSection:
    """Represents one coherent topic section with its transcript and summary."""
    title: str
    start_time: float
    segments: list[TranscriptSegment] = field(default_factory=list)
    summary: str = ""
    is_finalized: bool = False

    @property
    def transcript_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)


class TopicAnalyzer:
    """
    Reads TranscriptSegments, groups them into topic sections, and
    produces summaries per section via Ollama.

    Notifies a callback whenever topics are updated so the writer can
    re-render the markdown file.

    Usage:
        analyzer = TopicAnalyzer(transcript_queue, on_update=writer.render)
        analyzer.start()
        ...
        analyzer.stop()
    """

    def __init__(
        self,
        transcript_queue: queue.Queue,
        on_update,  # callable()
    ):
        self._transcript_q = transcript_queue
        self._on_update = on_update
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Shared state — also read by MarkdownWriter
        self.topics: list[TopicSection] = []
        self._lock = threading.Lock()

        # The current in-progress section
        self._current: TopicSection | None = None
        # Segments waiting for analysis
        self._pending_segments: list[TranscriptSegment] = []

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TopicAnalyzer")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=120)

    def _run(self) -> None:
        print("[TopicAnalyzer] Ready.")
        while not self._stop_event.is_set():
            try:
                seg: TranscriptSegment = self._transcript_q.get(timeout=1.0)
            except queue.Empty:
                continue
            self._process_segment(seg)

        # Drain on shutdown
        while not self._transcript_q.empty():
            try:
                seg = self._transcript_q.get_nowait()
                self._process_segment(seg)
            except queue.Empty:
                break

        # Finalize the last open section
        if self._current and self._current.segments:
            self._finalize_current()

        print("[TopicAnalyzer] Stopped.")

    def _process_segment(self, seg: TranscriptSegment) -> None:
        # Bootstrap the first section
        if self._current is None:
            self._current = TopicSection(title="Introduction", start_time=seg.start_time)
            with self._lock:
                self.topics.append(self._current)

        self._current.segments.append(seg)
        self._pending_segments.append(seg)

        # Notify writer so transcript lines appear immediately (no summary yet)
        self._on_update()

        # Only run topic detection once we have enough segments
        total = len(self._current.segments)
        if total >= config.TOPIC_MIN_SEGMENTS and len(self._pending_segments) >= config.TOPIC_WINDOW_SEGMENTS:
            self._check_topic_shift()
            self._pending_segments.clear()

    def _check_topic_shift(self) -> None:
        recent_text = " ".join(s.text for s in self._pending_segments)
        context_text = self._current.transcript_text[-2000:]  # last ~2000 chars for context

        prompt = (
            f"You are analyzing a transcript of an educational video course.\n\n"
            f"Current topic: \"{self._current.title}\"\n\n"
            f"Recent transcript:\n{recent_text}\n\n"
            f"Has the topic clearly shifted to something new and distinct? "
            f"Reply with ONLY one of:\n"
            f"  NO\n"
            f"  YES: <new topic title in 3-7 words>\n"
        )

        try:
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            answer = response["message"]["content"].strip()
        except Exception as e:
            print(f"[TopicAnalyzer] Ollama error during topic detection: {e}")
            return

        if answer.upper().startswith("YES"):
            new_title = answer.split(":", 1)[-1].strip() if ":" in answer else "New Topic"
            print(f"[TopicAnalyzer] Topic shift → '{new_title}'")
            self._finalize_current()
            new_section = TopicSection(
                title=new_title,
                start_time=self._pending_segments[0].start_time if self._pending_segments else 0,
            )
            # Move pending segments to new section
            new_section.segments.extend(self._pending_segments)
            self._current = new_section
            with self._lock:
                self.topics.append(self._current)
            self._on_update()
        else:
            print(f"[TopicAnalyzer] Same topic: '{self._current.title}'")

    def _finalize_current(self) -> None:
        if not self._current or not self._current.segments:
            return
        section = self._current
        print(f"[TopicAnalyzer] Summarizing '{section.title}'…")
        summary = self._summarize(section)
        with self._lock:
            section.summary = summary
            section.is_finalized = True
        self._on_update()

    @staticmethod
    def _summarize(section: TopicSection) -> str:
        transcript = section.transcript_text
        if not transcript.strip():
            return "_No content to summarize._"

        prompt = (
            f"You are creating study documentation for a section of an educational video course.\n\n"
            f"Topic: \"{section.title}\"\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Write comprehensive study notes that a student can use to fully understand and "
            f"review this topic without watching the video again. Structure your response in "
            f"markdown using the following format:\n\n"
            f"#### Overview\n"
            f"2-3 sentences explaining what this section is about and why it matters.\n\n"
            f"#### Key Concepts\n"
            f"For each important concept, write a bold heading followed by a clear explanation "
            f"of 2-4 sentences. Include examples or analogies if they were mentioned.\n\n"
            f"#### How It Works\n"
            f"A step-by-step or narrative explanation of the main process, technique, or idea "
            f"covered. Use numbered steps if applicable.\n\n"
            f"#### Key Takeaways\n"
            f"3-5 sentences summarizing the most important things to remember from this section.\n\n"
            f"Write in clear, educational prose. Do not just list words — explain ideas fully."
        )
        try:
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"_Summary unavailable: {e}_"
