# 📚 course-summarizer

Capture system audio while watching a video course, transcribe it locally with Whisper, detect topic shifts with Ollama, and produce a live-updated Markdown file — all running silently in the background.

## How it works

```
[Windows Speaker Output]
  ↓ WASAPI loopback (pyaudiowpatch) — no virtual cable needed
AudioCapture thread → audio_queue (30s PCM chunks)
  ↓
Transcriber thread (faster-whisper) → transcript_queue (timestamped text)
  ↓
TopicAnalyzer thread (Ollama) → detects topic shifts, summarizes each section
  ↓
MarkdownWriter → rewrites output/course_2026-03-03_13-00.md live
```

---

## Getting Started

Follow these steps in order to get the app running on **WSL2 (Arch Linux)**.

### Step 1 — Install the NVIDIA driver on Windows (optional, for GPU)

Download and install the NVIDIA Game Ready Driver on Windows (not inside WSL):
**[https://www.nvidia.com/en-us/drivers/details/235904/](https://www.nvidia.com/en-us/drivers/details/235904/)**

> [!NOTE]
>  **Do NOT install an NVIDIA driver inside WSL.** The Windows driver provides GPU passthrough automatically.

Verify the GPU is visible from WSL after installing:
```zsh
ls /dev/dxg      # ✔ should exist
nvidia-smi       # ✔ should show your GPU model and driver version
```

---

### Step 2 — Install the CUDA toolkit in WSL (optional, for GPU)

```zsh
yay -S cuda
```

Add CUDA to your shell:
```zsh
echo 'export PATH=/opt/cuda/bin:$PATH' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

Verify:
```zsh
nvcc --version   # ✔ should print CUDA version
```

---

### Step 3 — Verify PulseAudio in WSL

WSLg (Windows 11) already runs a PulseAudio server automatically at `/mnt/wslg/PulseServer` — no need to start one manually. Just install the client tools for `pactl`:

```zsh
yay -S --noconfirm pulseaudio
```

Verify the monitor source exists:
```zsh
pactl list short sources
# ✔ expected output on WSLg:
# 1  RDPSink.monitor  module-rdp-sink.c   s16le 2ch 44100Hz  SUSPENDED
# 2  RDPSource        module-rdp-source.c s16le 1ch 44100Hz  SUSPENDED
```

`RDPSink.monitor` is the system audio loopback — the app detects it automatically. `SUSPENDED` is normal when nothing is playing.

> [!NOTE] If `pactl` can't connect, ensure the `PULSE_SERVER` env var is set:
> ```zsh
> echo $PULSE_SERVER   # should print unix:/mnt/wslg/PulseServer
> # If empty, add to ~/.zshrc:
> echo 'export PULSE_SERVER=unix:/mnt/wslg/PulseServer' >> ~/.zshrc
> source ~/.zshrc
> ```

---

### Step 4 — Install and start Ollama in WSL

```zsh
# Install Ollama (official Linux installer)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (mistral is a good default, ~4 GB)
ollama pull mistral

# Start the server (runs in the background)
ollama serve &
```

Verify it is running:
```zsh
curl http://localhost:11434   # ✔ should return "Ollama is running"
```

> [!NOTE]
> **GPU note:** If you completed Steps 1–2 (NVIDIA driver + CUDA), Ollama will automatically use your GPU for inference. You'll see GPU utilisation in `nvidia-smi`.

---

### Step 5 — Set up Python and install dependencies

```zsh
# Clone / enter the project directory
cd course-summarizer

# Create a virtual environment (required on Arch)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify core packages installed:
```zsh
python -c "import faster_whisper; print('✔ faster-whisper OK')"
python -c "import ollama; print('✔ ollama OK')"
python -c "import sounddevice; print('✔ sounddevice OK')"
```

---

### Step 6 — Install PyTorch with CUDA (optional, for GPU)

```zsh
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:
```zsh
python -c "import torch; print('✔ CUDA available:', torch.cuda.is_available())"
```

When enabled, Whisper logs:
```
[Transcriber] Loading Whisper 'small' on cuda (float16)
```

---

### Step 7 — Run the smoke test

Verifies the full pipeline (transcription + topic analysis + markdown output) without needing live audio:

```zsh
python test_pipeline.py
# ✔ Test passed! Markdown file: course_2026-03-03_13-00.md
```

---

### Step 8 — Start the app

> **Important:** The app captures audio from **Linux apps running inside WSL** (via WSLg's RDP audio bridge). Windows apps (Chrome, video players, etc.) output audio directly to Windows and are not captured.
>
> **Watch your video course in a Linux browser inside WSL:**
> ```zsh
> yay -S brave-bin   # one-time install
> brave
> ```

Open your course in Firefox, then in a separate terminal start the app:

```zsh
python main.py start
```

Press **Ctrl+C** to stop. The app flushes remaining audio, finalizes summaries, and closes the file. Output is saved to `output/course_YYYY-MM-DD_HH-MM.md`.

---

## Prerequisites summary

| Requirement | Platform | Notes |
|-------------|----------|-------|
| Python 3.10+ | WSL | |
| PulseAudio | WSL | For audio loopback capture |
| [Ollama](https://ollama.com) | WSL | Run `ollama serve &` before starting the app |
| NVIDIA driver 465+ | Windows | Only needed for GPU acceleration |
| CUDA toolkit | WSL | Only needed for GPU acceleration |
| PyTorch (CUDA) | WSL venv | Only needed for GPU acceleration |

## Usage

### Start capturing

```zsh
python main.py start
```

Press **Ctrl+C** to stop. The app will flush remaining audio, finalize summaries, and close the file.

### With options

```zsh
python main.py start --whisper-model small --ollama-model mistral --output-dir output
```

| Flag | Default | Options |
|------|---------|---------|
| `--whisper-model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--ollama-model` | `mistral` | any model you have pulled via `ollama pull` |
| `--output-dir` | `output/` | any directory path |

### List audio devices

```zsh
python main.py devices
```

### Run the smoke test

```zsh
python test_pipeline.py
```

## Output format

Files are saved to `output/course_YYYY-MM-DD_HH-MM.md`:

```markdown
# 📚 Course Transcript — 2026-03-03 13:00

---

## ✅ Introduction to Docker
*Started at 00:00:00*

### Summary

#### Overview
Docker is a containerization platform that allows developers to package applications
and their dependencies into isolated units called containers...

#### Key Concepts
**Container**
A container is a lightweight, isolated process that shares the host OS kernel but
runs in its own environment. Unlike virtual machines, containers start instantly...

**Docker Image**
An image is a read-only template used to create containers. It contains the application
code, runtime, libraries, and configuration needed to run...

#### How It Works
1. You write a Dockerfile describing your application environment...
2. Docker builds an image from that file...
3. You run a container from the image...

#### Key Takeaways
Containers solve the "works on my machine" problem by bundling everything an application
needs. Docker makes it easy to build, ship, and run applications consistently...

### Transcript

**`00:00:05`**

Today we'll cover what Docker is and why it matters for modern development...

**`00:00:35`**

A container is an isolated process that shares the OS kernel but has its own filesystem...

---

## 🔄 Container Networking
*Started at 00:08:12*
...
```

The `🔄` icon means the section is still being captured. `✅` means it's been summarized.

## Configuration

Edit `config.py` to change defaults:

```python
WHISPER_MODEL = "small"        # tiny/base/small/medium/large-v3
AUDIO_CHUNK_SECONDS = 30       # seconds per transcription chunk
SILENCE_THRESHOLD = 0.01       # skip chunks quieter than this RMS
OLLAMA_MODEL = "mistral"       # must be pulled locally
TOPIC_WINDOW_SEGMENTS = 5      # segments to analyze for topic shifts
OUTPUT_DIR = "output"          # where .md files are written
```

## File structure

```
course-summarizer/
├── main.py             # CLI entry point
├── config.py           # All settings
├── audio_capture.py    # WASAPI loopback audio capture
├── transcriber.py      # faster-whisper transcription
├── topic_analyzer.py   # Ollama topic detection + summarization
├── markdown_writer.py  # Live markdown file builder
├── pipeline.py         # Wires all stages together
├── test_pipeline.py    # Smoke test (no live audio needed)
├── requirements.txt
└── output/             # Generated markdown files (created at runtime)
```
