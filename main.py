"""
main.py — CLI entry point for course-summarizer.

Usage:
    python main.py start
    python main.py start --whisper-model small --ollama-model mistral --output-dir output
    python main.py devices     # list available WASAPI loopback devices
"""

import argparse
import sys


def cmd_start(args) -> None:
    from pipeline import Pipeline
    p = Pipeline(
        whisper_model=args.whisper_model,
        ollama_model=args.ollama_model,
        output_dir=args.output_dir,
        lecture_title=args.title,
        module_name=args.module,
    )
    p.start()
    p.wait()


def cmd_devices(args) -> None:
    """List available audio loopback/input devices."""
    import platform
    if platform.system() == "Darwin":
        try:
            import sounddevice as sd
        except ImportError:
            print("ERROR: sounddevice is not installed. Run: pip install sounddevice")
            sys.exit(1)
        print("\nAvailable audio input devices (macOS):\n")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                marker = " ← loopback" if any(
                    kw in dev["name"].lower()
                    for kw in ["blackhole", "soundflower", "loopback", "ishowu"]
                ) else ""
                print(f"  [{i}] {dev['name']}{marker}")
                print(f"       Channels: {dev['max_input_channels']}  "
                      f"Sample rate: {int(dev['default_samplerate'])} Hz")
        return

    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("ERROR: pyaudiowpatch is not installed. Run: pip install pyaudiowpatch")
        sys.exit(1)

    pa = pyaudio.PyAudio()
    print("\nAvailable WASAPI loopback devices:\n")
    found = False
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("isLoopbackDevice", False):
            print(f"  [{i}] {info['name']}")
            print(f"       Channels: {info['maxInputChannels']}  "
                  f"Sample rate: {info['defaultSampleRate']} Hz")
            found = True
    if not found:
        print("  No loopback devices found.")
    pa.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="course-summarizer",
        description="Capture system audio, transcribe, detect topics, and produce a Markdown summary.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── start ────────────────────────────────────────────────────────────────
    start_parser = subparsers.add_parser("start", help="Start capturing and summarizing audio.")
    start_parser.add_argument(
        "--whisper-model",
        default=None,
        metavar="MODEL",
        help="Whisper model size: tiny, base, small (default), medium, large-v3",
    )
    start_parser.add_argument(
        "--ollama-model",
        default=None,
        metavar="MODEL",
        help="Ollama model name (must be pulled locally, e.g. mistral, llama3)",
    )
    start_parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write the markdown file (default: output/)",
    )
    start_parser.add_argument(
        "--title",
        default=None,
        metavar="TITLE",
        help="Lecture title — gives the LLM context for better summaries (e.g. 'Docker Networking')",
    )
    start_parser.add_argument(
        "--module",
        default=None,
        metavar="MODULE",
        help="Course or module name — gives the LLM domain context (e.g. 'Docker for Developers')",
    )
    start_parser.set_defaults(func=cmd_start)

    # ── devices ──────────────────────────────────────────────────────────────
    devices_parser = subparsers.add_parser("devices", help="List available WASAPI loopback devices.")
    devices_parser.set_defaults(func=cmd_devices)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
