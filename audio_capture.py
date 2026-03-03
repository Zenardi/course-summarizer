"""
audio_capture.py — System audio loopback capture, cross-platform.

Supports three backends chosen automatically at runtime:
  • Windows (native)  → WASAPI loopback via pyaudiowpatch
  • WSL / Linux       → PulseAudio monitor source via parec
  • macOS (Darwin)    → Virtual loopback device via sounddevice
                        (requires BlackHole: brew install blackhole-2ch)

All backends push fixed-length float32 mono PCM chunks (16 kHz) to a
thread-safe queue, so the rest of the pipeline is identical regardless
of platform.
"""

import os
import platform
import queue
import subprocess
import threading

import numpy as np

import config


# ── Platform detection ────────────────────────────────────────────────────────

def _running_on_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")


def _running_on_windows_native() -> bool:
    return platform.system() == "Windows"


def _running_on_macos() -> bool:
    return platform.system() == "Darwin"


# ── Shared silence helper ─────────────────────────────────────────────────────

def _is_silent(pcm: np.ndarray, threshold: float) -> bool:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    return rms < threshold


# ── Backend: WASAPI loopback (Windows native) ─────────────────────────────────

class _WasapiCapture:
    """Captures system audio via WASAPI loopback (Windows only)."""

    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        self._queue = audio_queue
        self._stop = stop_event

    def run(self) -> None:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            device = self._find_loopback_device(pa)
            device_index = int(device["index"])
            n_channels = int(device["maxInputChannels"])
            src_rate = int(device["defaultSampleRate"])

            frames_per_chunk = int(src_rate * config.AUDIO_CHUNK_SECONDS)
            read_block = int(src_rate * 0.1)  # 100 ms per read

            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=n_channels,
                rate=src_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=read_block,
            )

            accumulated: list[np.ndarray] = []
            frames_acc = 0
            print(f"[AudioCapture] WASAPI loopback: {device['name']} ({src_rate} Hz, {n_channels}ch)")

            while not self._stop.is_set():
                raw = stream.read(read_block, exception_on_overflow=False)
                block = np.frombuffer(raw, dtype=np.float32)
                accumulated.append(block)
                frames_acc += read_block

                if frames_acc >= frames_per_chunk:
                    chunk = _to_mono_16k(np.concatenate(accumulated), n_channels, src_rate)
                    accumulated, frames_acc = [], 0
                    self._emit(chunk)

            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()

    def _emit(self, chunk: np.ndarray) -> None:
        if _is_silent(chunk, config.SILENCE_THRESHOLD):
            print("[AudioCapture] Silent chunk skipped.")
        else:
            self._queue.put(chunk)

    @staticmethod
    def _find_loopback_device(pa) -> dict:
        import pyaudiowpatch as pyaudio
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_idx = wasapi_info["defaultOutputDevice"]
        default_name = pa.get_device_info_by_index(default_idx)["name"]
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False) and default_name in dev["name"]:
                return dev
        raise RuntimeError(
            "No WASAPI loopback device found. "
            "Ensure your audio output device supports loopback capture."
        )


# ── Backend: PulseAudio monitor source (WSL / Linux) ─────────────────────────

class _PulseAudioCapture:
    """
    Captures system audio via a PulseAudio monitor source (WSL2 / Linux).

    Uses `parec` (ships with pulseaudio) to read raw PCM from the monitor
    source via subprocess — bypasses PortAudio entirely, so it works even
    when PortAudio is not compiled with PulseAudio support (e.g. Arch Linux).
    """

    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        self._queue = audio_queue
        self._stop = stop_event

    def run(self) -> None:
        source = self._find_monitor_source()
        frames_per_chunk = config.SAMPLE_RATE * config.AUDIO_CHUNK_SECONDS
        bytes_per_chunk = frames_per_chunk * 4  # float32 = 4 bytes per sample

        cmd = [
            "parec",
            f"--device={source}",
            "--format=float32le",
            f"--rate={config.SAMPLE_RATE}",
            "--channels=1",
            "--latency-msec=100",
        ]

        print(f"[AudioCapture] PulseAudio monitor: {source} ({config.SAMPLE_RATE} Hz, mono)")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        try:
            while not self._stop.is_set():
                raw = proc.stdout.read(bytes_per_chunk)
                if not raw:
                    break
                if len(raw) < bytes_per_chunk:
                    continue  # partial read at shutdown, discard
                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                if _is_silent(chunk, config.SILENCE_THRESHOLD):
                    print("[AudioCapture] Silent chunk skipped.")
                else:
                    self._queue.put(chunk)
        finally:
            proc.terminate()
            proc.wait()

    @staticmethod
    def _find_monitor_source() -> str:
        """Use pactl to find the first available monitor source."""
        try:
            out = subprocess.check_output(
                ["pactl", "list", "short", "sources"], text=True
            )
        except FileNotFoundError:
            raise RuntimeError(
                "pactl not found. Install PulseAudio: yay -S pulseaudio"
            )

        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].endswith(".monitor"):
                return parts[1]

        raise RuntimeError(
            "No PulseAudio monitor source found.\n"
            "Run: pactl list short sources\n"
            "Expected a source ending in .monitor (e.g. RDPSink.monitor)"
        )



# ── Backend: Virtual loopback device (macOS) ──────────────────────────────────

class _MacOSCapture:
    """
    Captures system audio on macOS via a virtual loopback input device.

    Requires BlackHole (or Soundflower) installed and set as audio output:
        brew install blackhole-2ch
    Then in System Settings → Sound → Output, select "BlackHole 2ch".

    To hear audio while capturing, create a Multi-Output Device in
    Audio MIDI Setup that routes to both BlackHole and your speakers.
    """

    #: Device name substrings to recognise as loopback devices (case-insensitive)
    _LOOPBACK_KEYWORDS = ["blackhole", "soundflower", "loopback", "ishowu"]

    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        self._queue = audio_queue
        self._stop = stop_event

    def run(self) -> None:
        import time
        import sounddevice as sd

        device_index = self._find_loopback_device(sd)
        device_info = sd.query_devices(device_index)
        frames_per_chunk = config.SAMPLE_RATE * config.AUDIO_CHUNK_SECONDS

        accumulated: list[np.ndarray] = []
        frames_acc = 0

        def _callback(indata: np.ndarray, frames: int, _time, _status) -> None:
            nonlocal accumulated, frames_acc
            accumulated.append(indata[:, 0].copy())
            frames_acc += frames
            if frames_acc >= frames_per_chunk:
                chunk = np.concatenate(accumulated).astype(np.float32)
                accumulated.clear()
                frames_acc = 0
                if _is_silent(chunk, config.SILENCE_THRESHOLD):
                    print("[AudioCapture] Silent chunk skipped.")
                else:
                    self._queue.put(chunk)

        print(
            f"[AudioCapture] macOS loopback: {device_info['name']} "
            f"({config.SAMPLE_RATE} Hz, mono)"
        )

        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=config.SAMPLE_RATE,
            dtype="float32",
            blocksize=int(config.SAMPLE_RATE * 0.1),  # 100 ms blocks
            callback=_callback,
        ):
            while not self._stop.is_set():
                time.sleep(0.1)

    @classmethod
    def _find_loopback_device(cls, sd) -> int:
        """Return the index of the first recognised loopback input device."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                if any(kw in dev["name"].lower() for kw in cls._LOOPBACK_KEYWORDS):
                    return i

        input_devs = "\n".join(
            f"  [{i}] {d['name']}"
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        )
        raise RuntimeError(
            "No loopback audio device found on macOS.\n\n"
            "Install BlackHole to capture system audio:\n"
            "  brew install blackhole-2ch\n\n"
            "Then set BlackHole as your audio output in:\n"
            "  System Settings → Sound → Output → BlackHole 2ch\n\n"
            "To hear audio while capturing, open Audio MIDI Setup and create\n"
            "a Multi-Output Device combining BlackHole and your speakers.\n\n"
            f"Available input devices:\n{input_devs}"
        )




def _to_mono_16k(audio: np.ndarray, n_channels: int, src_rate: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz (Whisper's expected rate)."""
    if audio.ndim == 2 and n_channels > 1:
        audio = audio.mean(axis=1)
    elif audio.ndim == 2:
        audio = audio[:, 0]

    if src_rate != config.SAMPLE_RATE:
        target_len = int(len(audio) * config.SAMPLE_RATE / src_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio.astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

class AudioCapture:
    """
    Captures system audio output in fixed-length chunks.
    Automatically selects the right backend for the current platform.

    Usage:
        capture = AudioCapture(audio_queue)
        capture.start()
        ...
        capture.stop()
    """

    def __init__(self, audio_queue: queue.Queue):
        self._queue = audio_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AudioCapture"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        print("[AudioCapture] Stopped.")

    def _run(self) -> None:
        if _running_on_windows_native():
            backend = _WasapiCapture(self._queue, self._stop_event)
        elif _running_on_macos():
            backend = _MacOSCapture(self._queue, self._stop_event)
        else:
            # WSL2 or plain Linux
            backend = _PulseAudioCapture(self._queue, self._stop_event)
        backend.run()

