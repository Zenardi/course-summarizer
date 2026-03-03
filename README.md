# 📚 course-summarizer

Capture system audio while watching a video course, transcribe it locally with Whisper, detect topic shifts with Ollama, and produce a live-updated Markdown file — all running silently in the background.

- [📚 course-summarizer](#-course-summarizer)
  - [How it works](#how-it-works)
  - [Getting Started](#getting-started)
    - [WSL2 (Arch Linux)](#wsl2-arch-linux)
      - [Step 1 — Install the NVIDIA driver on Windows (optional, for GPU)](#step-1--install-the-nvidia-driver-on-windows-optional-for-gpu)
      - [Step 2 — Install the CUDA toolkit in WSL (optional, for GPU)](#step-2--install-the-cuda-toolkit-in-wsl-optional-for-gpu)
      - [Step 3 — Verify PulseAudio in WSL](#step-3--verify-pulseaudio-in-wsl)
      - [Step 4 — Install and start Ollama in WSL](#step-4--install-and-start-ollama-in-wsl)
      - [Step 5 — Set up Python and install dependencies](#step-5--set-up-python-and-install-dependencies)
      - [Step 6 — Install PyTorch with CUDA (optional, for GPU)](#step-6--install-pytorch-with-cuda-optional-for-gpu)
      - [Step 7 — Run the smoke test](#step-7--run-the-smoke-test)
      - [Step 8 — Start the app](#step-8--start-the-app)
    - [macOS (Apple Silicon / Intel)](#macos-apple-silicon--intel)
      - [Step 1 — Install BlackHole (audio loopback)](#step-1--install-blackhole-audio-loopback)
      - [Step 2 — Create a Multi-Output Device](#step-2--create-a-multi-output-device)
      - [Step 3 — Install and start Ollama](#step-3--install-and-start-ollama)
      - [Step 4 — Set up Python and install dependencies](#step-4--set-up-python-and-install-dependencies)
      - [Step 5 — Run the smoke test](#step-5--run-the-smoke-test)
      - [Step 6 — Start the app](#step-6--start-the-app)
  - [Prerequisites summary](#prerequisites-summary)
  - [Usage](#usage)
    - [Start capturing](#start-capturing)
    - [With all options](#with-all-options)
    - [List audio devices](#list-audio-devices)
    - [Run the smoke test](#run-the-smoke-test)
  - [Output format](#output-format)
  - [Configuration](#configuration)
    - [Whisper model](#whisper-model)
    - [Ollama model selection](#ollama-model-selection)
    - [Topic detection tuning (`TOPIC_WINDOW_SEGMENTS` / `TOPIC_MIN_SEGMENTS`)](#topic-detection-tuning-topic_window_segments--topic_min_segments)
  - [File structure](#file-structure)


## How it works

```
[System Audio Output]
  ↓ WASAPI loopback (Windows) · PulseAudio monitor (WSL/Linux) · BlackHole (macOS)
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

---

## WSL2 (Arch Linux)

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

# Pull the model — llama3.1:8b is recommended for technical courses (~5 GB)
ollama pull llama3.1:8b

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

Open your course in Brave, then in a separate terminal start the app:

```zsh
# Basic — no context
python main.py start

# With lecture context (recommended — gives the LLM better accuracy)
python main.py start --title "Docker Networking" --module "Docker for Developers"
```

Press **Ctrl+C** to stop. The app flushes remaining audio, finalizes summaries, and closes the file. Output is saved to `output/course_YYYY-MM-DD_HH-MM.md` (or `output/<lecture-title-slug>_YYYY-MM-DD_HH-MM.md` when `--title` is provided).

---

## macOS (Apple Silicon / Intel)

macOS does not expose a built-in audio loopback API, so a free virtual audio driver (**BlackHole**) is required to capture system output.

### Step 1 — Install BlackHole (audio loopback)

```zsh
brew install blackhole-2ch
```

BlackHole creates a virtual audio device that the app reads as an input source — no paid virtual cable needed.

---

### Step 2 — Create a Multi-Output Device

To hear audio through your speakers **and** capture it simultaneously:

1. Open **Audio MIDI Setup** (search in Spotlight).
2. Click **`+`** at the bottom-left → **Create Multi-Output Device**.
3. Check both **BlackHole 2ch** and your speakers (e.g. *MacBook Pro Speakers*).
4. Right-click the new **Multi-Output Device** → **Use This Device for Sound Output**.

Then in **System Settings → Sound → Output**, select **Multi-Output Device**.

> [!NOTE]
> If you only need to capture (no monitoring), you can skip the Multi-Output Device and set **BlackHole 2ch** directly as your output device.

---

### Step 3 — Install and start Ollama

```zsh
# Install Ollama (macOS pkg or Homebrew)
brew install ollama

# Pull the recommended model (~5 GB)
ollama pull llama3.1:8b

# Start the server
ollama serve &
```

Verify it is running:
```zsh
curl http://localhost:11434   # ✔ should return "Ollama is running"
```

---

### Step 4 — Set up Python and install dependencies

```zsh
cd course-summarizer

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Verify core packages:
```zsh
python -c "import faster_whisper; print('✔ faster-whisper OK')"
python -c "import ollama; print('✔ ollama OK')"
python -c "import sounddevice; print('✔ sounddevice OK')"
```

---

### Step 5 — Run the smoke test

```zsh
python test_pipeline.py
# ✔ Test passed! Markdown file: course_2026-03-03_13-00.md
```

---

### Step 6 — Start the app

Verify BlackHole is detected:
```zsh
python main.py devices
# ✔ BlackHole 2ch should appear with "← loopback"
```

Start the app:
```zsh
# Basic
python main.py start

# With lecture context (recommended)
python main.py start --title "Docker Networking" --module "Docker for Developers"
```

Press **Ctrl+C** to stop.

---

## Prerequisites summary

| Requirement | Platform | Notes |
|-------------|----------|-------|
| Python 3.10+ | WSL / macOS | |
| PulseAudio | WSL | For audio loopback capture |
| [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) | macOS | `brew install blackhole-2ch` |
| sounddevice | macOS | Installed via `requirements.txt` |
| [Ollama](https://ollama.com) | WSL / macOS | Run `ollama serve &` before starting the app |
| NVIDIA driver 465+ | Windows | Only needed for GPU acceleration |
| CUDA toolkit | WSL | Only needed for GPU acceleration |
| PyTorch (CUDA) | WSL venv | Only needed for GPU acceleration |

## Usage

### Start capturing

```zsh
# Minimal
python main.py start

# With lecture context (recommended)
python main.py start --title "Kubernetes Pods and Services" --module "Kubernetes Fundamentals"
```

Press **Ctrl+C** to stop. The app will flush remaining audio, finalize summaries, and close the file.

### With all options

```zsh
python main.py start --whisper-model large-v3 --ollama-model llama3.1:8b --output-dir output \
  --title "Docker Networking" --module "Docker for Developers"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--whisper-model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--ollama-model` | `llama3.1:8b` | any model you have pulled via `ollama pull` |
| `--output-dir` | `output/` | directory to write the markdown file |
| `--title` | *(none)* | lecture title — improves LLM topic detection and summaries |
| `--module` | *(none)* | course/module name — gives the LLM domain context |

### List audio devices

```zsh
python main.py devices
```

### Run the smoke test

```zsh
python test_pipeline.py
```

## Output format

Files are saved to `output/course_YYYY-MM-DD_HH-MM.md`, or `output/<title-slug>_YYYY-MM-DD_HH-MM.md` when `--title` is provided.

```markdown
# 📚 Docker for Developers — Docker Networking

*Recorded on 2026-03-03 13:00*

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
WHISPER_MODEL = "large-v3"     # tiny/base/small/medium/large-v3
AUDIO_CHUNK_SECONDS = 45       # seconds per transcription chunk
SILENCE_THRESHOLD = 0.001      # skip chunks quieter than this RMS
OLLAMA_MODEL = "llama3.1:8b"   # must be pulled locally
TOPIC_WINDOW_SEGMENTS = 10     # segments to analyze for topic shifts
TOPIC_MIN_SEGMENTS = 8         # minimum segments before a shift is considered
OUTPUT_DIR = "output"          # where .md files are written
```

### Whisper model

| Model | Speed | Accuracy | Recommended for |
|-------|-------|----------|-----------------|
| `tiny` / `base` | fastest | low | quick tests only |
| `small` | fast | good | CPU without GPU |
| `medium` | moderate | best CPU balance | CPU without GPU |
| `large-v3` | slow on CPU / fast on GPU | highest | **CUDA GPU** ✅ |

### Ollama model selection

The Ollama model handles both topic detection and summary generation. Model choice has a big impact on summary quality for technical content.

| Model | VRAM (Q4) | Technical knowledge | Summary quality | Best for |
|-------|-----------|--------------------|-----------------|----|
| `mistral` | ~5 GB | ⭐⭐⭐ | ⭐⭐⭐ | general content |
| `llama3.1:8b` | ~5 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **technical courses, 8 GB VRAM** ✅ |
| `qwen2.5:7b` | ~5 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ML/AI-heavy content |
| `mixtral:8x7b` | ~5.5 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | best quality, tight on 8 GB |

**`llama3.1:8b`** is the default. Compared to Mistral 7B it has significantly better knowledge of AI, Kubernetes, DevOps, and autonomous systems terminology, and follows complex structured prompts (Overview / Key Concepts / How It Works) more reliably.

To switch models at any time:
```zsh
ollama pull qwen2.5:7b
python main.py start --ollama-model qwen2.5:7b
```



Whisper transcribes more accurately with longer audio chunks — it can resolve ambiguous words using surrounding sentence context. `45s` is the sweet spot for short lessons: enough context without too much startup latency.

### Topic detection tuning (`TOPIC_WINDOW_SEGMENTS` / `TOPIC_MIN_SEGMENTS`)

These two settings control how aggressively the app splits the transcript into separate topic sections.

- **`TOPIC_MIN_SEGMENTS`** — minimum segments a topic must accumulate before a split is even considered. Acts as a guard against splitting on the very first sentence of a new topic.
- **`TOPIC_WINDOW_SEGMENTS`** — how many recent segments are sent to Ollama each time it checks for a topic shift. More segments = more context = fewer false splits.

**Recommended presets:**

| Lesson type | `TOPIC_MIN_SEGMENTS` | `TOPIC_WINDOW_SEGMENTS` |
|---|---|---|
| Short / focused (Udacity, ~5–15 min, 1–3 topics) | `8` | `10` ✅ |
| Medium (30–45 min course, 3–6 topics) | `5` | `7` |
| Long / dense (1h+, many topic changes) | `3` | `4` |

With the **Udacity preset** (`8` / `10`), a typical 8-minute lesson produces a single deep summary section. Genuine topic shifts (e.g. moving from theory to a hands-on demo) are still detected because the large window gives Ollama enough contrast to recognise the change.

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
