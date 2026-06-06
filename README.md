# AudioNorm & AudioScribe

Command line audio processing and transcription tools combining FFmpeg,
AI-based denoising, and speech recognition. Can be used with TTS-generated audio, podcasts, and voice recordings.

## What These Tools Do

### AudioNorm - Audio Enhancement & Cleanup

- **AI-powered denoising** using SpeechBrain MTL-MIMIC and Demucs models
- **Professional normalization** with EBU R128 loudness standards
- **Vocal/background separation** with stem export functionality

### AudioScribe - Audio Transcription

- **Whisper-based transcription** using Faster-Whisper models
- **Multi-language support** with auto-detection
- **Timestamps and formatting** options

## System Requirements

- **Python 3.8+**
- **FFmpeg** (must be in system PATH)

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg

# Verify installation
ffmpeg -version
```

### Clone Repo and setup Python Environment

## Option 1: Python conda environment

```bash
# Create conda virtual environment
conda create -n audionorm-env python=3.12
conda activate audionorm-env

git clone https://github.com/bradsec/audionorm.git
cd audionorm

# Install dependencies
pip install -r requirements.txt

# Verify installation
python audionorm.py --help
python audioscribe.py --help
```

## Option 2: Python venv environment

```bash
# Create virtual environment
python -m venv audionorm-env
source audionorm-env/bin/activate  # Linux/macOS
# audionorm-env\Scripts\activate   # Windows

git clone https://github.com/bradsec/audionorm.git
cd audionorm

# Install dependencies
pip install -r requirements.txt

# Verify installation
python audionorm.py --help
python audioscribe.py --help
```

## AudioNorm - Audio Enhancement

### AudioNorm Features

- **SpeechBrain Enhancement**: MTL-MIMIC model for voice quality improvement
- **Demucs AI Denoising**: htdemucs_ft model optimized for vocal separation
- **Stem Separation**: Extract vocals and background tracks separately
- **Professional Normalization**: EBU R128 loudness standards
- **Intelligent Processing**: Auto-detects optimal processing pipeline
- **Batch Processing**: Handle entire directories recursively

### AudioNorm Examples

```bash
usage: audionorm.py [-h] [--target-lufs TARGET_LUFS] [--model MODEL] [--device {cpu,cuda}] [--recursive] [--keep-temp] [--skip-demucs] [--enhanced-cleaning]
                    [--basic-cleaning] [--python-restoration] [--verbose] [--quiet] [--trim-silence] [--silence-threshold SILENCE_THRESHOLD]
                    [--intensive-cleanup] [--use-loudnorm] [--overwrite] [--format {wav,mp3}] [--two-pass-loudnorm] [--single-pass-loudnorm]
                    [--no-speechbrain] [--save-stems] [--stems-only] [--voice-consistent] [--separator {demucs,roformer,mel-roformer}]
                    input

Audio normalization and cleanup for TTS-generated audio

positional arguments:
  input                 Input audio file or directory

options:
  -h, --help            show this help message and exit
  --target-lufs TARGET_LUFS
                        Target loudness in LUFS (default: -18.0 optimized for voice content)
  --model MODEL         Demucs model (default: htdemucs_ft for better vocal separation)
  --device {cpu,cuda}   Processing device (auto-detect if not specified)
  --recursive, -r       Process directories recursively
  --keep-temp           Keep temporary files for debugging
  --skip-demucs         Skip Demucs denoising (FFmpeg normalization only)
  --enhanced-cleaning   Enable enhanced FFmpeg cleaning (gentle)
  --basic-cleaning      Use basic cleaning only (faster processing)
  --python-restoration  Use Python-based restoration (librosa + noisereduce) instead of Demucs
  --verbose, -v         Enable verbose logging
  --quiet, -q           Minimal output mode
  --trim-silence        Trim extended silences (>1s) to natural pause lengths (0.2-0.8s)
  --silence-threshold SILENCE_THRESHOLD
                        Silence detection threshold in dB (default: -30, lower=more sensitive)
  --intensive-cleanup   Enable intensive post-AI cleanup (may cause reverb artifacts, disabled by default)
  --use-loudnorm        Use loudnorm instead of dynaudnorm (may sound over-processed)
  --overwrite           Overwrite existing output files (default: skip existing files)
  --format {wav,mp3}, -f {wav,mp3}
                        Output audio format (default: wav)
  --two-pass-loudnorm   Use professional two-pass loudnorm for superior volume consistency (default: enabled)
  --single-pass-loudnorm
                        Use single-pass loudnorm instead of two-pass (faster but less accurate)
  --no-speechbrain      Disable SpeechBrain MTL-MIMIC voice enhancement (default: enabled)
  --save-stems          Save all Demucs stems (vocals, drums, bass, other) when using Demucs pipeline
  --stems-only          Only separate into vocal and background stems without any processing or normalization
  --voice-consistent    Use voice-consistent normalization for uniform voice levels throughout audio (fixes level variations)
  --separator {demucs,roformer,mel-roformer}
                        Vocal separator: roformer (default, BS-RoFormer SDR 12.97), demucs, or mel-roformer (Mel-Band-RoFormer, SDR 11.4). roformer/mel-
                        roformer require: pip install audio-separator. --save-stems and --stems-only produce vocals + background (2 stems, not 4).

Examples:
  audionorm.py audio.wav
  audionorm.py /path/to/audio/folder --recursive
  audionorm.py audio.wav --target-lufs -23 --model htdemucs
  audionorm.py folder/ --device cuda --keep-temp --quiet
```

## AudioScribe - Speech Transcription

### Core Features

- **Faster-Whisper Models**: Optimized Whisper implementation
- **Multi-language Support**: 60+ languages with auto-detection
- **Flexible Output**: Plain text, timestamps, or structured formats
- **Batch Transcription**: Process entire directories
- **Model Selection**: From tiny (fast) to large-v3 (most accurate)

### Usage

```bash
usage: audioscribe.py [-h]
                      [--model {tiny,tiny.en,base,base.en,small,small.en,medium,medium.en,large-v1,large-v2,large-v3,distil-large-v2,distil-large-v3}]
                      [--language LANGUAGE] [--task {transcribe,translate}] [--device {cpu,cuda}]
                      [--beam-size BEAM_SIZE] [--recursive] [--no-timestamps] [--keep-temp] [--verbose] [--quiet]
                      [--list-models] [--overwrite]
                      input

High-quality audio transcription with Faster-Whisper

positional arguments:
  input                 Input audio file or directory

options:
  -h, --help            show this help message and exit
  --model {tiny,tiny.en,base,base.en,small,small.en,medium,medium.en,large-v1,large-v2,large-v3,distil-large-v2,distil-large-v3}
                        Whisper model size (default: large-v2 for best balance)
  --language LANGUAGE   Source language code (auto-detect if not specified). Examples: en, es, fr, de, it, pt, ru, ja,
                        ko, zh
  --task {transcribe,translate}
                        Task to perform (default: transcribe)
  --device {cpu,cuda}   Processing device (auto-detect if not specified)
  --beam-size BEAM_SIZE
                        Beam size for decoding (higher = more accurate but slower, default: 5)
  --recursive, -r       Process directories recursively
  --no-timestamps       Disable timestamps in output (plain text only)
  --keep-temp           Keep temporary files for debugging
  --verbose, -v         Enable verbose logging
  --quiet, -q           Minimal output mode
  --list-models         List available models and exit
  --overwrite           Overwrite existing transcript files (default: skip existing files)

Examples:
  audioscribe.py audio.wav
  audioscribe.py /path/to/audio/folder --recursive
  audioscribe.py audio.wav --model large-v3 --language en
  audioscribe.py folder/ --device cuda --no-timestamps --quiet
```


