#!/usr/bin/env python3
"""
audionorm.py - Audio normalization and cleanup for TTS-generated audio

Combines FFmpeg normalization with Demucs AI-based denoising to clean and
normalize audio files, specifically optimized for TTS-generated content.

Usage:
    python audionorm.py file.wav
    python audionorm.py /path/to/folder
    python audionorm.py file.wav --target-lufs -23 --denoise-model htdemucs_ft
"""

import argparse
import importlib.util
import logging
import secrets
import shutil
import subprocess  # nosec B404 - Used for FFmpeg calls with validated inputs
import sys
import warnings
from pathlib import Path
from typing import List, Optional

# Third-party imports
import numpy as np
import noisereduce as nr
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from scipy import signal
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Check for optional dependencies
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logging.warning("soundfile not available, using torchaudio for audio I/O")

HAS_SPEECHBRAIN = importlib.util.find_spec("speechbrain") is not None

# Global console for rich output
console = Console()

# Configure logging with suppressed output by default
logging.basicConfig(
    level=logging.ERROR,  # Default to ERROR to suppress all debug/info/warning
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Custom filter to suppress SpeechBrain debug logs
class SpeechBrainFilter(logging.Filter):
    def filter(self, record):
        # Always block specific debug messages that clutter output
        if record.levelno <= logging.INFO:
            message = getattr(record, 'getMessage', lambda: str(record.msg))()
            # Block specific annoying messages regardless of verbose mode
            block_patterns = [
                'registered checkpoint save hook',
                'registered checkpoint load hook',
                'fetch hyperparams.yaml',
                'fetch enhance_model.ckpt',
                'collecting files',
                'set local path in self.paths',
                'loading pretrained files',
                'redirecting (loading from local path)'
            ]
            if any(pattern in message.lower() for pattern in block_patterns):
                return False

            # Block DEBUG and INFO messages from SpeechBrain unless in verbose mode
            if not VERBOSE_MODE:
                # Check for SpeechBrain-related messages by content or module name
                if any(keyword in message.lower() for keyword in
                       ['speechbrain', 'checkpoint', 'pretrained', 'fetch']):
                    return False
                # Also check the logger name
                if hasattr(record, 'name') and any(keyword in record.name.lower() for keyword in
                       ['speechbrain', 'checkpoint', 'pretrained']):
                    return False
        return True

# Set global logging level to WARNING to prevent any third-party DEBUG logs
logging.getLogger().setLevel(logging.WARNING)

# Add the filter to the root logger
logging.getLogger().addFilter(SpeechBrainFilter())

# Suppress SpeechBrain debug logs globally (they show before VERBOSE_MODE is set)
try:
    import importlib.util
    speechbrain_spec = importlib.util.find_spec("speechbrain")
    if speechbrain_spec is not None:
        # Set the root speechbrain logger to WARNING level
        logging.getLogger('speechbrain').setLevel(logging.WARNING)

        # Also suppress all known speechbrain submodules
        speechbrain_loggers = [
            'speechbrain.utils.checkpoints',
            'speechbrain.pretrained',
            'speechbrain.utils.parameter_transfer',
            'speechbrain.utils.fetching',
            'speechbrain.utils.seed',
            'speechbrain.utils',
            'speechbrain.utils.torch_audio_backend',
            'speechbrain.dataio.dataio',
            'speechbrain.dataio',
            'speechbrain.dataio.dataset',
            'speechbrain.dataio.sampler',
            'speechbrain.dataio.dataloader',
            'speechbrain.utils.quirks',
            'speechbrain.core'
        ]
        for logger_name in speechbrain_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Set propagate to False to prevent logs from bubbling up
        logging.getLogger('speechbrain').propagate = False
except ImportError:
    pass  # SpeechBrain not available, no need to suppress

# Global flags for output control
QUIET_MODE = False
VERBOSE_MODE = False

# Audio processing constants
CLICK_REMOVAL_FILTER = "adeclick=t=2:w=10"
SIBILANT_REDUCTION_FILTER = "equalizer=f=4000:width_type=o:width=3:g=-1.5"
HISS_REDUCTION_7K_FILTER = "equalizer=f=7000:width_type=o:width=2:g=-2"
HISS_REDUCTION_10K_FILTER = "equalizer=f=10000:width_type=o:width=2:g=-1"
VOICE_WARMTH_FILTER = "equalizer=f=200:width_type=o:width=2:g=1"
VOICE_CLARITY_FILTER = "equalizer=f=2500:width_type=o:width=2:g=0.5"

def run_neural_noise_reduction(input_file: Path, output_file: Path, voice_optimized: bool = True) -> bool:
    """Apply FFmpeg's AI-based neural noise reduction (arnndn filter).

    Uses the 'cb' (conjoined-burgers) model which provides high-quality
    noise reduction with CPU efficiency comparable to loudnorm.
    Enhanced with voice-specific frequency optimization.
    Based on 2024-2025 research showing superior performance for voice.

    Args:
        input_file: Input audio file
        output_file: Output file with neural noise reduction applied
        voice_optimized: Apply voice frequency optimization (200-3000Hz)

    Returns:
        bool: Success status
    """
    try:
        # Build filter chain with optional voice optimization
        filters = []

        if voice_optimized:
            # Voice frequency optimization from research
            filters.extend([
                "highpass=f=200",      # Remove frequencies below voice range
                "lowpass=f=3000",      # Remove frequencies above voice range
            ])

        # Add neural noise reduction (try arnndn, fallback to afwtdn)
        # Note: arnndn may require model files not available on all systems
        filters.append("afwtdn")  # Wavelet-based denoising (more universally available)

        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-af", ",".join(filters),
            "-c:a", "pcm_s16le",
            "-ar", "24000",  # Voice-optimized sample rate
            "-ac", "1",      # Mono output
            str(output_file)
        ]

        if VERBOSE_MODE:
            mode = "voice-optimized" if voice_optimized else "standard"
            logging.info(f"Applying {mode} neural noise reduction (arnndn with cb model)")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
            if result.stderr:
                logging.info(f"Neural noise reduction stderr: {result.stderr}")
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # nosec B603 - FFmpeg with validated paths

        return True

    except subprocess.CalledProcessError as e:
        if VERBOSE_MODE:
            logging.warning(f"Neural noise reduction failed (falling back to adaptive): {e}")
        return False
    except Exception as e:
        if VERBOSE_MODE:
            logging.warning(f"Neural noise reduction error: {e}")
        return False

def run_adaptive_noise_reduction(input_file: Path, output_file: Path) -> bool:
    """Apply adaptive noise reduction that profiles noise from quiet sections.
    
    Uses FFmpeg's afftdn with adaptive noise tracking to identify and remove
    consistent background noise patterns, especially effective for low-level
    reverb and ambient noise in quiet sections.
    
    Args:
        input_file: Input audio file
        output_file: Output file with noise reduction applied
        
    Returns:
        bool: Success status
    """
    try:
        # Use afftdn with balanced parameters to reduce wind noise without muffling
        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-af", (
                # Gentle but targeted noise reduction
                "afftdn="
                "nr=8:"           # Moderate noise reduction (less aggressive)
                "nf=-32:"         # Balanced noise floor detection
                "nt=w:"           # White noise type
                "tn=1:"           # Track noise continuously
                "om=o,"           # Output mode: original (preserves dynamics)
                # Voice-optimized frequency shaping
                "highpass=f=80,"  # Remove rumble below voice range
                "lowpass=f=8000," # Remove harsh frequencies above voice range
                "equalizer=f=4500:width_type=h:width=1500:g=-0.8" # Gentle wind noise reduction
            ),
            "-c:a", "pcm_s16le",
            "-ar", "24000",  # Voice-optimized sample rate
            "-ac", "1",      # Mono output
            str(output_file)
        ]
        
        if VERBOSE_MODE:
            logging.info("Applying adaptive noise profiling and reduction")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
            if result.stderr:
                logging.info(f"Adaptive noise reduction stderr: {result.stderr}")
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # nosec B603 - FFmpeg with validated paths
            
        return True
        
    except subprocess.CalledProcessError as e:
        if VERBOSE_MODE:
            logging.warning(f"Adaptive noise reduction failed: {e}")
        return False
    except Exception as e:
        if VERBOSE_MODE:
            logging.warning(f"Noise reduction error: {e}")
        return False

def _get_voice_enhancement_filters() -> list:
    """Get standard voice enhancement filter chain."""
    return [
        CLICK_REMOVAL_FILTER,        # Remove digital clicks/pops
        # Anti-hissing for speech pronunciation
        SIBILANT_REDUCTION_FILTER,   # Reduce harsh sibilant hiss
        HISS_REDUCTION_7K_FILTER,    # Target pronunciation hiss frequencies
        HISS_REDUCTION_10K_FILTER,   # Gentle high-freq hiss reduction
        # Voice enhancement
        VOICE_WARMTH_FILTER,         # Warm up voice
        VOICE_CLARITY_FILTER,        # Enhance clarity without harshness
    ]

def _get_standard_cleanup_filters() -> list:
    """Get standard intensive cleanup filter chain."""
    return [
        # Stage 1: Spectral cleanup
        "afftdn=nf=-25:nt=w:tn=1",  # Remove residual spectral noise
        CLICK_REMOVAL_FILTER,        # Remove digital clicks/pops
        # Stage 2: Anti-hissing for speech pronunciation
        SIBILANT_REDUCTION_FILTER,   # Reduce harsh sibilant hiss
        HISS_REDUCTION_7K_FILTER,    # Target pronunciation hiss frequencies
        HISS_REDUCTION_10K_FILTER,   # Gentle high-freq hiss reduction
        # Stage 3: Voice-optimized frequency filtering
        "highpass=f=80",            # Remove rumble below voice range
        "lowpass=f=8000",           # Remove harsh frequencies above voice range
        # Stage 4: Voice enhancement
        VOICE_WARMTH_FILTER,         # Warm up voice
        VOICE_CLARITY_FILTER,        # Enhance clarity without harshness
    ]

def _cleanup_temp_files(temp_files: list) -> None:
    """Clean up temporary files safely."""
    for temp_file in temp_files:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except OSError as e:
                # Log specific cleanup errors but continue processing
                logging.debug(f"Could not remove temp file {temp_file}: {e}")

def run_post_ai_cleanup(
    input_file: Path,
    temp_file: Path,
    intensive_cleanup: bool = False,
) -> bool:
    """Apply post-AI cleanup to remove residual artifacts.

    This stage specifically targets artifacts left after AI denoising (Demucs):
    - Residual static and noise
    - Digital clicks and pops
    - Echo/reverb artifacts
    - Frequency imbalances

    Args:
        input_file: Input file (post-Demucs)
        temp_file: Output cleaned file
        intensive_cleanup: Enable more aggressive artifact removal

    Returns:
        bool: Success status
    """
    try:
        if intensive_cleanup:
            cleanup_input, filters, temp_files = _apply_intensive_cleanup(input_file)
        else:
            # Ultra-minimal cleanup - only remove clicks
            filters = [CLICK_REMOVAL_FILTER]
            cleanup_input = input_file
            temp_files = []

        # Execute FFmpeg cleanup
        cmd = [
            "ffmpeg", "-y", "-i", str(cleanup_input),
            "-af", ",".join(filters),
            "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1",
            str(temp_file),
        ]

        if VERBOSE_MODE:
            logging.info(f"Post-AI cleanup command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
            if result.stderr:
                logging.info(f"FFmpeg cleanup stderr: {result.stderr}")
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # nosec B603 - FFmpeg with validated paths

        # Clean up temporary files
        _cleanup_temp_files(temp_files)
        return True

    except subprocess.CalledProcessError as e:
        stderr_msg = getattr(e, 'stderr', '') or str(e)
        logging.error(f"Post-AI cleanup failed for {input_file}: {stderr_msg}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during post-AI cleanup: {e}")
        return False

def _apply_intensive_cleanup(input_file: Path) -> tuple:
    """Apply intensive cleanup with neural noise reduction fallback.

    Returns:
        tuple: (cleanup_input_file, filters, temp_files_to_cleanup)
    """
    temp_neural_file = input_file.parent / f"{input_file.stem}_temp_neural.wav"
    temp_adaptive_file = input_file.parent / f"{input_file.stem}_temp_adaptive.wav"

    # Try neural noise reduction first
    if run_neural_noise_reduction(input_file, temp_neural_file, voice_optimized=True):
        if VERBOSE_MODE:
            logging.info("Applied voice-optimized neural noise reduction, adding final cleanup")
        return temp_neural_file, _get_voice_enhancement_filters(), [temp_neural_file, temp_adaptive_file]

    # Try adaptive noise reduction
    if run_adaptive_noise_reduction(input_file, temp_adaptive_file):
        if VERBOSE_MODE:
            logging.info("Applied adaptive noise profiling, adding final cleanup")
        return temp_adaptive_file, _get_voice_enhancement_filters(), [temp_neural_file, temp_adaptive_file]

    # Fall back to standard intensive cleanup
    if VERBOSE_MODE:
        logging.info("Neural/adaptive noise reduction failed, using standard intensive cleanup")
    return input_file, _get_standard_cleanup_filters(), [temp_neural_file, temp_adaptive_file]

def run_speechbrain_enhance(input_file: Path, output_file: Path, model_name: str = "speechbrain/mtl-mimic-voicebank") -> bool:
    """Apply SpeechBrain MTL-MIMIC enhancement for superior voice quality.

    MTL-MIMIC-Voicebank provides better voice enhancement than MetricGAN+ and
    handles the API correctly with proper lengths parameter management.

    Args:
        input_file: Input audio file
        output_file: Output enhanced file
        model_name: SpeechBrain model to use

    Returns:
        bool: Success status
    """
    # Suppress SpeechBrain debug logs if not in verbose mode
    if not VERBOSE_MODE:
        speechbrain_logger = logging.getLogger('speechbrain')
        speechbrain_logger.setLevel(logging.WARNING)
        # Suppress all related loggers
        for logger_name in ['speechbrain.utils.checkpoints', 'speechbrain.pretrained',
                           'speechbrain.utils.parameter_transfer', 'speechbrain.utils.fetching']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    if not HAS_SPEECHBRAIN:
        if VERBOSE_MODE:
            logging.warning("SpeechBrain not available, skipping voice enhancement")
        return False

    try:
        if VERBOSE_MODE:
            logging.info(f"Loading SpeechBrain model: {model_name}")

        # Import using correct SpeechBrain 1.0+ API
        from speechbrain.inference.enhancement import WaveformEnhancement
        import torchaudio
        import sys
        import os

        # Suppress stderr during entire SpeechBrain process if not in verbose mode
        if not VERBOSE_MODE:
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

        try:
            # Load the enhancement model with GPU support if available
            try:
                model = WaveformEnhancement.from_hparams(
                    source=model_name,
                    savedir=f"pretrained_models/{model_name.split('/')[-1]}",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
            except Exception:
                # Fallback without GPU options if CUDA unavailable
                model = WaveformEnhancement.from_hparams(source=model_name)

            if VERBOSE_MODE:
                logging.info(f"Enhancing audio with SpeechBrain: {input_file.name}")

            # Load and enhance the audio
            waveform, sample_rate = torchaudio.load(input_file)

            # Process with SpeechBrain enhancement
            enhanced = model.enhance_batch(waveform, lengths=torch.tensor([1.0]))

            # Save enhanced audio
            torchaudio.save(output_file, enhanced.cpu(), sample_rate)

            if VERBOSE_MODE:
                logging.info(f"SpeechBrain enhancement complete: {output_file.name}")

            return True

        finally:
            # Restore stderr
            if not VERBOSE_MODE:
                sys.stderr.close()
                sys.stderr = original_stderr

    except Exception as e:
        if VERBOSE_MODE:
            logging.error(f"SpeechBrain enhancement failed for {input_file}: {e}")
        return False

def _parse_loudness_measurements(stderr_output: str) -> dict:
    """Parse loudness measurements from FFmpeg stderr output.

    Args:
        stderr_output: FFmpeg stderr output containing loudness measurements

    Returns:
        dict: Parsed loudness measurements
    """
    measurements = {}
    for line in stderr_output.split('\n'):
        if 'Input Integrated:' in line:
            measurements['input_i'] = float(line.split()[-2])
        elif 'Input True Peak:' in line:
            measurements['input_tp'] = float(line.split()[-2])
        elif 'Input LRA:' in line:
            measurements['input_lra'] = float(line.split()[-2])
        elif 'Input Threshold:' in line:
            measurements['input_thresh'] = float(line.split()[-2])
        elif 'Target Offset:' in line:
            measurements['target_offset'] = float(line.split()[-2])
    return measurements

def _build_normalize_filter(measurements: dict, target_lufs: float, target_tp: float, target_lra: float) -> str:
    """Build loudnorm filter string based on measurements.

    Args:
        measurements: Parsed loudness measurements
        target_lufs: Target loudness in LUFS
        target_tp: Target true peak in dBFS
        target_lra: Target loudness range in LU

    Returns:
        str: FFmpeg filter string
    """
    if len(measurements) >= 5:  # All measurements available
        normalize_filter = (
            f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
            f"measured_I={measurements['input_i']}:"
            f"measured_TP={measurements['input_tp']}:"
            f"measured_LRA={measurements['input_lra']}:"
            f"measured_thresh={measurements['input_thresh']}:"
            f"offset={measurements['target_offset']}"
        )
        if VERBOSE_MODE:
            logging.info("Applying two-pass normalization with measured values")
    else:
        # Fallback to single-pass if measurements failed
        normalize_filter = f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}"
        if VERBOSE_MODE:
            logging.warning("Measurements incomplete, falling back to single-pass loudnorm")
    return normalize_filter

def _run_loudness_analysis(input_file: Path, target_lufs: float, target_tp: float, target_lra: float) -> dict:
    """Run first pass loudness analysis.

    Args:
        input_file: Input audio file
        target_lufs: Target loudness in LUFS
        target_tp: Target true peak in dBFS
        target_lra: Target loudness range in LU

    Returns:
        dict: Parsed loudness measurements
    """
    analyze_cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:print_format=summary",
        "-f", "null", "-"
    ]

    result = subprocess.run(analyze_cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
    measurements = _parse_loudness_measurements(result.stderr)

    if VERBOSE_MODE:
        logging.info(f"Measured: I={measurements.get('input_i', 'N/A')} LUFS, "
                    f"TP={measurements.get('input_tp', 'N/A')} dBFS, "
                    f"LRA={measurements.get('input_lra', 'N/A')} LU")

    return measurements

def _apply_loudness_normalization(input_file: Path, output_file: Path, normalize_filter: str) -> bool:
    """Apply loudness normalization using the given filter.

    Args:
        input_file: Input audio file
        output_file: Output normalized file
        normalize_filter: FFmpeg filter string

    Returns:
        bool: Success status
    """
    normalize_cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-af", normalize_filter,
        "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1",
        str(output_file)
    ]

    if VERBOSE_MODE:
        logging.info(f"Two-pass loudnorm command: {' '.join(normalize_cmd)}")
        result = subprocess.run(normalize_cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
        if result.stderr:
            logging.info(f"Two-pass loudnorm stderr: {result.stderr}")
    else:
        subprocess.run(normalize_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # nosec B603 - FFmpeg with validated paths

    return True

def run_two_pass_loudnorm(
    input_file: Path,
    output_file: Path,
    target_lufs: float = -18.0,
    target_tp: float = -1.5,
    target_lra: float = 3.0,
) -> bool:
    """Professional two-pass loudness normalization for consistent volume.

    First pass analyzes the audio to measure loudness parameters,
    second pass applies normalization using these measurements for
    superior accuracy and consistency compared to single-pass loudnorm.

    Args:
        input_file: Input audio file
        output_file: Output normalized file
        target_lufs: Target loudness in LUFS (default -23 EBU R128)
        target_tp: Target true peak in dBFS (default -1.5)
        target_lra: Target loudness range in LU (default 7)

    Returns:
        bool: Success status
    """
    try:
        if VERBOSE_MODE:
            logging.info(f"Two-pass loudnorm: analyzing {input_file.name}")

        # Pass 1: Analyze audio to get loudness measurements
        measurements = _run_loudness_analysis(input_file, target_lufs, target_tp, target_lra)

        # Pass 2: Apply normalization using measured values
        normalize_filter = _build_normalize_filter(measurements, target_lufs, target_tp, target_lra)

        # Apply normalization
        return _apply_loudness_normalization(input_file, output_file, normalize_filter)

    except subprocess.CalledProcessError as e:
        stderr_msg = getattr(e, 'stderr', '') or str(e)
        logging.error(f"Two-pass loudnorm failed for {input_file}: {stderr_msg}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during two-pass loudnorm: {e}")
        return False

def _get_dynaudnorm_filters(target_lufs: float) -> list:
    """Get balanced dynamic normalization filters.

    Args:
        target_lufs: Target loudness in LUFS

    Returns:
        list: FFmpeg filter strings
    """
    return [
        # Gentle compression to even out levels safely
        "compand=attacks=0.15:decays=0.4:points=-80/-80|-40/-30|-20/-15|-10/-10|-5/-5|0/-5",
        # Safe loudness normalization with conservative peak limit
        f"loudnorm=I={target_lufs}:TP=-3.0:LRA=9",
        # Conservative limiter to completely prevent clipping
        "alimiter=level_in=1:level_out=0.85:limit=0.85:attack=3:release=75",
    ]

def _get_voice_consistent_filters(target_lufs: float) -> list:
    """Get voice-consistent normalization filters for uniform voice levels.

    Args:
        target_lufs: Target loudness in LUFS

    Returns:
        list: FFmpeg filter strings optimized for voice consistency
    """
    return [
        # Stage 1: Voice-specific dynamic range compression
        "compand=attacks=0.05:decays=0.2:points=-80/-80|-50/-40|-30/-25|-20/-18|-15/-15|-10/-12|-5/-8|0/-5",
        # Stage 2: Tight loudness normalization for voice consistency
        f"loudnorm=I={target_lufs}:TP=-2.0:LRA=3.0",  # Much tighter LRA for voice
        # Stage 3: Voice-optimized limiter for consistent peaks
        "alimiter=level_in=1:level_out=0.9:limit=0.9:attack=1:release=50",
        # Stage 4: Final voice leveling with AGC-like behavior
        "dynaudnorm=framelen=500:gausssize=31:peak=0.9:maxgain=10:targetrms=0.25",
    ]

def _get_enhanced_cleaning_filters(target_lufs: float) -> list:
    """Get legacy enhanced cleaning filters.

    Args:
        target_lufs: Target loudness in LUFS

    Returns:
        list: FFmpeg filter strings
    """
    return [
        "highpass=f=80",  # Voice-optimized low cut
        "lowpass=f=8000", # Voice-optimized high cut
        "agate=threshold=0.001:ratio=3:attack=2:release=100:makeup=1.0",
        "compand=attacks=0.1:decays=0.3:points=-80/-80|-40/-35|-25/-20|-15/-12|-5/-3|0/-1",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=3.0",
    ]

def _get_minimal_filters(target_lufs: float) -> list:
    """Get minimal loudnorm filters.

    Args:
        target_lufs: Target loudness in LUFS

    Returns:
        list: FFmpeg filter strings
    """
    return [f"loudnorm=I={target_lufs}:TP=-1.5:LRA=3.0"]

def _get_normalization_filters(target_lufs: float, use_dynaudnorm: bool, enhanced_cleaning: bool, voice_consistent: bool = False) -> list:
    """Get appropriate normalization filters based on mode.

    Args:
        target_lufs: Target loudness in LUFS
        use_dynaudnorm: Use dynamic normalization
        enhanced_cleaning: Enable legacy enhanced cleaning
        voice_consistent: Use voice-consistent normalization for uniform voice levels

    Returns:
        list: FFmpeg filter strings
    """
    if voice_consistent:
        if VERBOSE_MODE:
            logging.info("Using voice-consistent normalization for uniform voice levels throughout audio")
        return _get_voice_consistent_filters(target_lufs)
    elif use_dynaudnorm:
        if VERBOSE_MODE:
            logging.info("Using balanced normalization for consistent speech levels")
        return _get_dynaudnorm_filters(target_lufs)
    elif enhanced_cleaning:
        if VERBOSE_MODE:
            logging.info("Using legacy loudnorm approach (may sound over-processed)")
        return _get_enhanced_cleaning_filters(target_lufs)
    else:
        return _get_minimal_filters(target_lufs)

def _add_format_encoding_params(cmd: list, output_format: str) -> None:
    """Add format-specific encoding parameters to FFmpeg command.

    Args:
        cmd: FFmpeg command list to extend
        output_format: Output format (wav or mp3)
    """
    if output_format == "mp3":
        cmd.extend([
            "-c:a", "libmp3lame",  # MP3 encoder
            "-b:a", "128k",        # 128kbps bitrate (sufficient for voice)
            "-ar", "24000",        # 24kHz sample rate for voice
            "-ac", "1",            # Mono output
        ])
    else:  # wav (default)
        cmd.extend([
            "-c:a", "pcm_s16le",   # 16-bit PCM for WAV
            "-ar", "24000",        # 24kHz sample rate for voice
            "-ac", "1",            # Mono output
        ])

def run_ffmpeg_normalize(
    input_file: Path,
    temp_file: Path,
    target_lufs: float = -23.0,  # EBU R128 standard for broadcast content
    use_dynaudnorm: bool = True,
    enhanced_cleaning: bool = False,
    output_format: str = "wav",
    use_two_pass: bool = False,
    voice_consistent: bool = False,
) -> bool:
    """Normalize volume with ffmpeg optimized for natural speech dynamics.

    Args:
        input_file: Input audio file path
        temp_file: Output normalized file path
        target_lufs: Target loudness in LUFS (default -23 EBU R128 broadcast standard)
        use_dynaudnorm: Use dynamic normalization (preserves speech dynamics)
        enhanced_cleaning: Enable legacy loudnorm with cleaning (not recommended)
        use_two_pass: Use two-pass loudnorm for superior consistency
        voice_consistent: Use voice-consistent normalization for uniform voice levels

    Returns:
        bool: Success status
    """
    try:
        if use_two_pass and not voice_consistent:
            # Use professional two-pass loudnorm for best consistency (unless voice_consistent requested)
            if VERBOSE_MODE:
                logging.info("Using two-pass loudnorm for superior volume consistency")
            return run_two_pass_loudnorm(input_file, temp_file, target_lufs)
        elif voice_consistent:
            # Voice-consistent mode overrides two-pass
            if VERBOSE_MODE:
                logging.info("Using voice-consistent normalization for uniform voice levels")
            pass  # Continue to filter-based approach below

        # Get appropriate filters based on mode
        filters = _get_normalization_filters(target_lufs, use_dynaudnorm, enhanced_cleaning, voice_consistent)

        # Build FFmpeg command with format-specific encoding
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_file),
            "-af", ",".join(filters),
        ]

        # Add format-specific encoding parameters (mono 24kHz for voice)
        _add_format_encoding_params(cmd, output_format)
        cmd.append(str(temp_file))

        if VERBOSE_MODE:
            logging.info(f"FFmpeg normalization command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
            if result.stderr:
                logging.info(f"FFmpeg normalization stderr: {result.stderr}")
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # nosec B603 - FFmpeg with validated paths
        return True
    except subprocess.CalledProcessError as e:
        stderr_msg = getattr(e, 'stderr', '') or str(e)
        logging.error(f"FFmpeg normalization failed for {input_file}: {stderr_msg}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during normalization: {e}")
        return False

def _load_audio_file(input_file: Path) -> tuple:
    """Load audio file using best available library.

    Returns:
        tuple: (audio_data, sample_rate)
    """
    if HAS_SOUNDFILE:
        y, sr = sf.read(str(input_file))
        return y.astype(np.float64), sr
    else:
        # Fallback to torchaudio
        wav, sr = torchaudio.load(str(input_file))
        y = wav.numpy()
        if y.ndim > 1:
            y = y[0]  # Take first channel for mono processing
        return y.astype(np.float64), sr

def _apply_noise_reduction(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply gentle noise reduction to audio."""
    try:
        # Only apply noise reduction if there's significant quiet background
        y_denoised = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=True,  # More conservative for TTS
            prop_decrease=0.2,  # Very gentle - only 20% reduction
            n_fft=512,  # Smaller window
            chunk_size=sr // 4,  # Process in smaller chunks
        )
        if VERBOSE_MODE:
            logging.info("Applied gentle noise reduction")
        return y_denoised
    except Exception:
        # If noise reduction fails, just use original
        if VERBOSE_MODE:
            logging.info("Skipped noise reduction")
        return y.copy()

def _remove_clicks(y: np.ndarray) -> np.ndarray:
    """Remove extreme clicks using median filtering."""
    y_smooth = signal.medfilt(y, kernel_size=3)
    diff = np.abs(y - y_smooth)
    threshold = np.percentile(diff, 99.8)  # Only top 0.2% outliers

    y_restored = y.copy()
    extreme_clicks = diff > threshold
    if np.any(extreme_clicks):
        y_restored[extreme_clicks] = y_smooth[extreme_clicks]
        click_count = np.sum(extreme_clicks)
        if VERBOSE_MODE:
            logging.info(f"Removed {click_count} extreme clicks")
    return y_restored

def _preserve_audio_level(y_restored: np.ndarray, original_peak: float) -> np.ndarray:
    """Preserve original audio level and apply safety limiting."""
    current_peak = np.max(np.abs(y_restored))

    if current_peak > 0 and original_peak > 0:
        # Maintain the original peak level
        level_ratio = original_peak / current_peak
        y_restored = y_restored * level_ratio
        if VERBOSE_MODE:
            logging.info(f"Preserved original level (ratio: {level_ratio:.3f})")
    elif current_peak == 0:
        logging.error("Audio processing resulted in silent output - restoration failed")
        return None
    elif original_peak == 0:
        logging.warning("Original audio was silent - keeping processed output")

    # Final safety check
    final_peak = np.max(np.abs(y_restored))
    if final_peak > 0.95:
        y_restored = y_restored * 0.95 / final_peak

    return y_restored

def _save_audio_file(y_restored: np.ndarray, sr: int, output_file: Path) -> bool:
    """Save processed audio using best available library."""
    if HAS_SOUNDFILE:
        sf.write(str(output_file), y_restored, sr)
    else:
        # Fallback to torchaudio
        y_tensor = torch.from_numpy(y_restored).float()
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)
        torchaudio.save(str(output_file), y_tensor, sr)
    return True

def python_audio_restoration(input_file: Path, output_file: Path) -> bool:
    """Apply gentle Python-based audio restoration without over-processing.

    Args:
        input_file: Input audio file path
        output_file: Output restored file path

    Returns:
        bool: Success status
    """
    try:
        if not QUIET_MODE:
            console.print("  [yellow]Processing: Applying Python audio restoration...[/yellow]")

        # Load audio file
        y, sr = _load_audio_file(input_file)

        # Ensure mono processing
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Store original peak for reference
        original_peak = np.max(np.abs(y))
        logging.info(f"Original peak level: {original_peak:.6f}")

        # Apply restoration steps
        y_denoised = _apply_noise_reduction(y, sr)
        y_restored = _remove_clicks(y_denoised)
        y_final = _preserve_audio_level(y_restored, original_peak)

        if y_final is None:
            return False

        # Save processed audio
        _save_audio_file(y_final, sr, output_file)

        # Verify output level
        verify_y, _ = _load_audio_file(output_file)
        if verify_y.ndim > 1:
            verify_y = verify_y[0]  # Take first channel
        verify_peak = np.max(np.abs(verify_y))

        if VERBOSE_MODE:
            logging.info(f"Output peak level: {verify_peak:.6f}")

        if verify_peak < 0.001:
            logging.error("Output level too low - restoration failed")
            return False

        return True

    except Exception as e:
        logging.error(f"Python audio restoration failed for {input_file}: {e}")
        return False

def run_silence_trimmer(
    input_file: Path, 
    output_file: Path, 
    min_silence_duration: float = 1.0,
    min_pause_length: float = 0.2,
    max_pause_length: float = 0.8,
    silence_threshold: float = -30.0,
    output_format: str = "wav"
) -> bool:
    """Trim extended silences to natural pause lengths using FFmpeg.
    
    This function detects silences longer than min_silence_duration and reduces
    them to a random length between min_pause_length and max_pause_length,
    creating more natural sounding pauses in speech audio.
    
    Args:
        input_file: Input audio file path
        output_file: Output file with trimmed silences
        min_silence_duration: Minimum silence length to trim (seconds)
        min_pause_length: Minimum natural pause length (seconds)  
        max_pause_length: Maximum natural pause length (seconds)
        silence_threshold: dB threshold for silence detection
        
    Returns:
        bool: Success status
    """
    try:
        # Generate random pause length for natural variation
        # Use secrets.randbelow for cryptographically secure randomness
        range_ms = int((max_pause_length - min_pause_length) * 1000)
        random_ms = secrets.randbelow(range_ms + 1)
        target_pause = min_pause_length + (random_ms / 1000.0)
        
        if VERBOSE_MODE:
            logging.info(f"Trimming silences >{min_silence_duration}s to {target_pause:.2f}s at {silence_threshold}dB threshold")
        
        # Build FFmpeg command for silence removal optimized for speech
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-i", str(input_file),
            "-af", (
                f"silenceremove="
                f"start_periods=1:"  # Remove silence from beginning
                f"stop_periods=-1:"  # Remove all middle/end silences  
                f"stop_duration={target_pause}:"  # Keep this much silence
                f"start_threshold={silence_threshold}dB:"  # Beginning threshold
                f"stop_threshold={silence_threshold}dB:"  # Middle/end threshold
                f"detection=peak"  # Use peak detection for speech
            ),
        ]
        
        # Add format-specific encoding parameters (mono 24kHz for voice)
        if output_format == "mp3":
            cmd.extend([
                "-c:a", "libmp3lame",  # MP3 encoder
                "-b:a", "128k",        # 128kbps bitrate (sufficient for voice)
                "-ar", "24000",        # 24kHz sample rate for voice
                "-ac", "1",            # Mono output
            ])
        else:  # wav (default)
            cmd.extend([
                "-c:a", "pcm_s16le",   # 16-bit PCM for WAV
                "-ar", "24000",        # 24kHz sample rate for voice
                "-ac", "1",            # Mono output
            ])
        
        cmd.append(str(output_file))
        
        if VERBOSE_MODE:
            logging.info(f"Silence trimming command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603 - FFmpeg with validated paths
            if result.stderr:
                logging.info(f"FFmpeg silence trimming stderr: {result.stderr}")
        else:
            # Suppress FFmpeg output in normal mode
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # nosec B603 - FFmpeg with validated paths
        
        # Verify output file was created and has content
        if not output_file.exists() or output_file.stat().st_size == 0:
            logging.error(f"Silence trimming produced empty output: {output_file}")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        stderr_msg = getattr(e, 'stderr', '') or str(e)
        logging.error(f"Silence trimming failed for {input_file}: {stderr_msg}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during silence trimming: {e}")
        return False

def load_demucs_model(model_name: str = "htdemucs_ft", device: Optional[str] = None):
    """Load Demucs model once for reuse across multiple files.
    
    Args:
        model_name: Demucs model to use (htdemucs_ft better for speech)
        device: Processing device (auto-detected if None)
        
    Returns:
        tuple: (model, device) for reuse
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not QUIET_MODE:
        console.print(f"  [cyan]Loading Demucs model '{model_name}' on {device}...[/cyan]")
    # Load model
    try:
        model = get_model(model_name)
    except Exception as e:
        if not QUIET_MODE:
            console.print(f"  [yellow]Warning: Model '{model_name}' not found, using 'htdemucs'[/yellow]")
        if VERBOSE_MODE:
            logging.warning(f"Model '{model_name}' not found, falling back to 'htdemucs': {e}")
        model = get_model("htdemucs")
    model.eval()
    if device == "cuda":
        model = model.to(device)
    return model, device

def _load_audio_for_demucs(input_file: Path) -> tuple[torch.Tensor, int]:
    """Load audio file and prepare it for Demucs processing."""
    if HAS_SOUNDFILE:
        wav, sr = sf.read(str(input_file))
        wav = torch.from_numpy(wav).float()
        # Soundfile returns (samples, channels), need (channels, samples) for Demucs
        if wav.dim() == 2:
            wav = wav.transpose(0, 1)  # Convert from (samples, channels) to (channels, samples)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension for mono
        if VERBOSE_MODE:
            logging.info(f"Loaded audio with soundfile, shape: {wav.shape}")
    else:
        wav, sr = torchaudio.load(str(input_file))
        if VERBOSE_MODE:
            logging.info(f"Loaded audio with torchaudio, shape: {wav.shape}")
    return wav, sr

def _prepare_audio_channels(wav: torch.Tensor) -> torch.Tensor:
    """Prepare audio channels for Demucs processing (requires stereo)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    # Demucs expects stereo input - convert mono to stereo if needed
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # Duplicate mono to both channels
        if VERBOSE_MODE:
            logging.info("Converted mono to stereo for Demucs processing")
    elif wav.shape[0] > 2:
        # If more than 2 channels, take first 2
        wav = wav[:2]
        if VERBOSE_MODE:
            logging.info("Reduced to stereo from multi-channel")
    return wav

def _apply_demucs_separation(model, wav: torch.Tensor, device: str, sr: int) -> torch.Tensor:
    """Apply Demucs separation and extract vocals."""
    chunk_size = sr * 30  # 30 second chunks
    if VERBOSE_MODE:
        logging.info(f"Audio length: {wav.shape[-1]} samples, chunk size: {chunk_size}, will chunk: {wav.shape[-1] > chunk_size}")

    if wav.shape[-1] <= chunk_size:
        # Separate into stems and extract vocals only
        separated = apply_model(model, wav[None], device=device, progress=False)
        # Extract vocals stem (index 3) for voice/music separation
        # htdemucs model outputs: [0] drums, [1] bass, [2] other, [3] vocals
        denoised = separated[0][3]  # Extract vocals stem only
        if VERBOSE_MODE:
            logging.info(f"Extracted vocals stem from Demucs separation, shape: {denoised.shape}")
    else:
        # Process long files in chunks
        chunks = []
        for i in range(0, wav.shape[-1], chunk_size):
            chunk = wav[:, i : i + chunk_size]
            separated = apply_model(model, chunk[None], device=device, progress=False)
            chunk_vocals = separated[0][3]  # Extract vocals stem only
            chunks.append(chunk_vocals.cpu())
        denoised = torch.cat(chunks, dim=-1)
        if VERBOSE_MODE:
            logging.info(f"Extracted vocals stem from Demucs separation (chunked), shape: {denoised.shape}")

    return denoised

def _convert_to_mono(denoised: torch.Tensor) -> torch.Tensor:
    """Convert stereo to mono if needed."""
    if VERBOSE_MODE:
        logging.info(f"Demucs output tensor shape: {denoised.shape}")

    # Convert stereo to mono if needed (proper dimension handling)
    if denoised.dim() == 2 and denoised.shape[0] == 2:
        # Convert stereo to mono by averaging channels
        denoised = torch.mean(denoised, dim=0, keepdim=True)
        if VERBOSE_MODE:
            logging.info(f"Converted stereo to mono, new shape: {denoised.shape}")
    elif denoised.dim() == 2 and denoised.shape[0] == 1:
        # Already mono, keep as is
        if VERBOSE_MODE:
            logging.info("Output already mono")
    elif denoised.dim() == 1:
        # 1D tensor, add channel dimension
        denoised = denoised.unsqueeze(0)
        if VERBOSE_MODE:
            logging.info(f"Added channel dimension to 1D tensor, new shape: {denoised.shape}")

    return denoised

def _save_demucs_output(denoised: torch.Tensor, output_file: Path, sr: int) -> None:
    """Save the processed audio to file."""
    if HAS_SOUNDFILE:
        # Convert to numpy and ensure correct shape for soundfile
        audio_data = denoised.numpy()

        # soundfile expects (samples,) for mono or (samples, channels) for multi-channel
        if audio_data.ndim == 2:
            if audio_data.shape[0] == 1:
                # Convert from (1, samples) to (samples,) for mono
                audio_data = audio_data[0]
            else:
                # Convert from (channels, samples) to (samples, channels)
                audio_data = audio_data.T

        if VERBOSE_MODE:
            logging.info(f"Saving audio data with shape: {audio_data.shape}, dtype: {audio_data.dtype}")

        sf.write(str(output_file), audio_data, sr)
    else:
        # Ensure proper tensor shape for torchaudio: (channels, samples)
        if denoised.dim() == 1:
            denoised = denoised.unsqueeze(0)
        torchaudio.save(str(output_file), denoised, sr)

def _save_vocals_and_background(
    input_file: Path, output_dir: Path, base_name: str, model, device: str, vocals_temp_file: Path
) -> bool:
    """Extract vocals and save background music as separate files.

    Creates two files:
    - {base_name}_vocals.wav - Isolated vocals/speech
    - {base_name}_background.wav - Everything else (music, instruments, etc.)

    Args:
        input_file: Input audio file
        output_dir: Directory to save stems
        base_name: Base filename for stems
        model: Pre-loaded Demucs model
        device: Processing device
        vocals_temp_file: Where to save vocals for further processing

    Returns:
        bool: Success status
    """
    try:
        logging.info(f"DEBUG: _save_vocals_and_background called for {input_file}")
        # Load and prepare audio
        wav, sr = _load_audio_for_demucs(input_file)
        wav = _prepare_audio_channels(wav)
        logging.info(f"DEBUG: Audio loaded - shape: {wav.shape}, sample_rate: {sr}")

        # Apply model to get all stems
        with torch.no_grad():
            if device == "cuda":
                wav = wav.to(device)

            # Apply Demucs separation
            chunk_size = sr * 30  # 30 second chunks
            if wav.shape[-1] <= chunk_size:
                separated = apply_model(model, wav[None], device=device, progress=False)
                stems = separated[0]  # [drums, bass, other, vocals]
            else:
                # Process long files in chunks
                all_chunks = [[], [], [], []]  # For each stem
                for i in range(0, wav.shape[-1], chunk_size):
                    chunk = wav[:, i : i + chunk_size]
                    separated = apply_model(model, chunk[None], device=device, progress=False)
                    chunk_stems = separated[0]
                    for stem_idx in range(4):
                        all_chunks[stem_idx].append(chunk_stems[stem_idx].cpu())

                # Concatenate chunks for each stem
                stems = [torch.cat(chunks, dim=-1) for chunks in all_chunks]

        # Move to CPU
        stems = [stem.cpu() for stem in stems]

        # Extract vocals (index 3) and create background (everything else)
        vocals_stem = _convert_to_mono(stems[3])  # Vocals

        # Combine drums, bass, and other for background
        background = stems[0] + stems[1] + stems[2]  # drums + bass + other
        background_stem = _convert_to_mono(background)

        # Save vocals stem for processing pipeline
        _save_demucs_output(vocals_stem, vocals_temp_file, sr)

        # Save both stems as final outputs
        vocals_output = output_dir / f"{base_name}_vocals_raw.wav"
        background_output = output_dir / f"{base_name}_background_raw.wav"

        logging.info(f"DEBUG: Saving vocals to: {vocals_output}")
        _save_demucs_output(vocals_stem, vocals_output, sr)
        logging.info(f"DEBUG: Saving background to: {background_output}")
        _save_demucs_output(background_stem, background_output, sr)

        if VERBOSE_MODE:
            logging.info(f"Saved vocals: {vocals_output.name}")
            logging.info(f"Saved background: {background_output.name}")

        return True

    except Exception as e:
        logging.error(f"Vocals/background separation failed for {input_file}: {e}")
        return False

def run_demucs_with_all_stems(
    input_file: Path, output_dir: Path, base_name: str, model, device: str
) -> dict:
    """Apply Demucs separation and save all stems (vocals, drums, bass, other).

    Args:
        input_file: Input audio file
        output_dir: Directory to save stems
        base_name: Base filename for stems
        model: Pre-loaded Demucs model
        device: Processing device

    Returns:
        dict: Paths to saved stems {"vocals": path, "drums": path, "bass": path, "other": path}
    """
    try:
        # Load and prepare audio
        wav, sr = _load_audio_for_demucs(input_file)
        wav = _prepare_audio_channels(wav)

        # Apply model
        with torch.no_grad():
            if device == "cuda":
                wav = wav.to(device)

            # Apply Demucs separation and get all stems
            chunk_size = sr * 30  # 30 second chunks
            if wav.shape[-1] <= chunk_size:
                separated = apply_model(model, wav[None], device=device, progress=False)
                stems = separated[0]  # [drums, bass, other, vocals]
            else:
                # Process long files in chunks
                all_chunks = [[], [], [], []]  # For each stem
                for i in range(0, wav.shape[-1], chunk_size):
                    chunk = wav[:, i : i + chunk_size]
                    separated = apply_model(model, chunk[None], device=device, progress=False)
                    chunk_stems = separated[0]
                    for stem_idx in range(4):
                        all_chunks[stem_idx].append(chunk_stems[stem_idx].cpu())

                # Concatenate chunks for each stem
                stems = [torch.cat(chunks, dim=-1) for chunks in all_chunks]

        # Move to CPU
        stems = [stem.cpu() for stem in stems]

        # Save all stems
        stem_names = ["drums", "bass", "other", "vocals"]
        stem_paths = {}

        for idx, (name, stem) in enumerate(zip(stem_names, stems)):
            # Convert to mono
            processed_stem = _convert_to_mono(stem)

            # Save stem
            output_path = output_dir / f"{base_name}_{name}.wav"
            _save_demucs_output(processed_stem, output_path, sr)
            stem_paths[name] = output_path

            if VERBOSE_MODE:
                logging.info(f"Saved {name} stem: {output_path.name}")

        return stem_paths

    except Exception as e:
        logging.error(f"Demucs stem separation failed for {input_file}: {e}")
        return {}

def run_demucs_denoise(
    input_file: Path, output_file: Path, model, device: str
) -> bool:
    """Apply Demucs denoiser optimized for TTS artifacts removal.

    Args:
        input_file: Input normalized audio file
        output_file: Output denoised file path
        model: Pre-loaded Demucs model
        device: Processing device

    Returns:
        bool: Success status
    """
    try:
        # Load and preprocess audio
        wav, sr = _load_audio_for_demucs(input_file)

        # Prepare audio channels for Demucs processing
        wav = _prepare_audio_channels(wav)

        # Apply model with optimized settings for TTS
        with torch.no_grad():
            if device == "cuda":
                wav = wav.to(device)

            # Apply Demucs separation and extract vocals
            denoised = _apply_demucs_separation(model, wav, device, sr)

        # Move to CPU for processing
        denoised = denoised.cpu()

        # Convert to mono if needed
        denoised = _convert_to_mono(denoised)

        # Save the processed audio
        _save_demucs_output(denoised, output_file, sr)
        return True
        
    except Exception as e:
        logging.error(f"Demucs processing failed for {input_file}: {e}")
        return False

def _process_skip_demucs_pipeline(
    file_path: Path,
    output_file: Path,
    temp_normalized_file: Path,
    target_lufs: float,
    enhanced_cleaning: bool,
    use_two_pass_loudnorm: bool,
    trim_silence: bool,
    silence_threshold: float,
    output_format: str,
    voice_consistent: bool = False,
) -> bool:
    """Handle the skip-demucs processing pipeline."""
    # Skip vocal separation - normalize the original audio
    normalize_output = temp_normalized_file if trim_silence else output_file
    if not run_ffmpeg_normalize(
        file_path, normalize_output, target_lufs, enhanced_cleaning=enhanced_cleaning,
        use_two_pass=use_two_pass_loudnorm, voice_consistent=voice_consistent
    ):
        return False
    if not QUIET_MODE:
        console.print("  [blue]Info: Normalized audio without vocal separation[/blue]")
    
    # Apply silence trimming if requested
    if trim_silence:
        if not run_silence_trimmer(temp_normalized_file, output_file, silence_threshold=silence_threshold, output_format=output_format):
            return False
        if not QUIET_MODE:
            console.print("  [green]Step 2: Trimmed extended silences[/green]")
    return True

def _process_python_restoration_pipeline(
    file_path: Path,
    output_file: Path,
    temp_vocals_file: Path,
    temp_normalized_file: Path,
    target_lufs: float,
    enhanced_cleaning: bool,
    use_two_pass_loudnorm: bool,
    trim_silence: bool,
    silence_threshold: float,
    output_format: str,
    voice_consistent: bool = False,
) -> bool:
    """Handle the Python restoration processing pipeline."""
    # Use Python-based restoration instead of Demucs
    if not python_audio_restoration(file_path, temp_vocals_file):
        return False
    # Then normalize the restored audio
    normalize_output = temp_normalized_file if trim_silence else output_file
    if not run_ffmpeg_normalize(
        temp_vocals_file, normalize_output, target_lufs, enhanced_cleaning=enhanced_cleaning,
        use_two_pass=use_two_pass_loudnorm, voice_consistent=voice_consistent
    ):
        return False
    
    # Apply silence trimming if requested
    if trim_silence:
        if not run_silence_trimmer(temp_normalized_file, output_file, silence_threshold=silence_threshold, output_format=output_format):
            return False
        if not QUIET_MODE:
            console.print("  [green]Step 3: Trimmed extended silences[/green]")
    return True

def _process_demucs_pipeline(
    file_path: Path,
    output_file: Path,
    temp_files: dict,  # Contains temp_vocals_file, temp_cleaned_file, temp_normalized_file
    model,
    device: str,
    processing_config: dict,  # Contains target_lufs, enhanced_cleaning, intensive_cleanup, use_dynaudnorm, use_two_pass_loudnorm, trim_silence, silence_threshold, output_format
) -> bool:
    """Handle the improved Demucs processing pipeline with post-AI cleanup."""
    # Extract parameters from config and temp_files
    temp_vocals_file = temp_files['temp_vocals_file']
    temp_cleaned_file = temp_files['temp_cleaned_file']
    temp_normalized_file = temp_files['temp_normalized_file']

    target_lufs = processing_config['target_lufs']
    enhanced_cleaning = processing_config['enhanced_cleaning']
    intensive_cleanup = processing_config['intensive_cleanup']
    use_dynaudnorm = processing_config['use_dynaudnorm']
    use_two_pass_loudnorm = processing_config['use_two_pass_loudnorm']
    trim_silence = processing_config['trim_silence']
    silence_threshold = processing_config['silence_threshold']
    output_format = processing_config['output_format']
    voice_consistent = processing_config.get('voice_consistent', False)

    # Step 1: Extract vocals (and optionally all stems) using Demucs
    save_stems = processing_config.get('save_stems', False)

    if VERBOSE_MODE:
        logging.info(f"save_stems flag: {save_stems}")

    if save_stems:
        # Save vocals and background (everything else) when requested
        base_name = file_path.stem
        output_dir = file_path.parent

        # Run Demucs separation and get vocals + background
        if _save_vocals_and_background(file_path, output_dir, base_name, model, device, temp_vocals_file):
            if not QUIET_MODE:
                console.print("  [green]Step 1: Extracted vocals and saved background music[/green]")
        else:
            return False
    else:
        # Standard vocals-only extraction
        if not run_demucs_denoise(file_path, temp_vocals_file, model, device):
            return False
        if not QUIET_MODE:
            console.print("  [green]Step 1: Extracted vocals from audio[/green]")
    
    # Step 2: NEW - Post-AI cleanup to remove residual artifacts
    if not run_post_ai_cleanup(temp_vocals_file, temp_cleaned_file, intensive_cleanup):
        return False
    if not QUIET_MODE:
        console.print("  [green]Step 2: Cleaned up AI artifacts[/green]")
    
    # Step 3: Normalize the cleaned vocals (now using better algorithm)
    normalize_input = temp_cleaned_file
    normalize_output = temp_normalized_file if trim_silence else output_file
    
    if not run_ffmpeg_normalize(
        normalize_input, normalize_output, target_lufs,
        use_dynaudnorm=use_dynaudnorm, enhanced_cleaning=enhanced_cleaning,
        output_format=output_format, use_two_pass=use_two_pass_loudnorm,
        voice_consistent=voice_consistent
    ):
        return False
    if not QUIET_MODE:
        method = "dynaudnorm" if use_dynaudnorm else "loudnorm"
        console.print(f"  [green]Step 3: Normalized with {method}[/green]")
    
    # Step 4: Trim silences if requested
    if trim_silence:
        if not run_silence_trimmer(temp_normalized_file, output_file, silence_threshold=silence_threshold, output_format=output_format):
            return False
        if not QUIET_MODE:
            console.print("  [green]Step 4: Trimmed extended silences[/green]")
    return True

def _process_speechbrain_pipeline(
    file_path: Path,
    output_file: Path,
    temp_enhanced_file: Path,
    temp_normalized_file: Path,
    target_lufs: float,
    enhanced_cleaning: bool,
    use_two_pass_loudnorm: bool,
    trim_silence: bool,
    silence_threshold: float,
    output_format: str,
    save_stems: bool = False,
    model=None,
    device: Optional[str] = None,
    voice_consistent: bool = False,
) -> bool:
    """Handle the SpeechBrain MetricGAN+ processing pipeline."""
    logging.info(f"DEBUG: _process_speechbrain_pipeline called with save_stems={save_stems}, model={model is not None}, device={device}")

    # Handle save_stems request before SpeechBrain processing
    if save_stems:
        logging.info("DEBUG: Entering save_stems block in _process_speechbrain_pipeline")
        if model is None or device is None:
            logging.info(f"DEBUG: save_stems failed - model is None: {model is None}, device is None: {device is None}")
            if not QUIET_MODE:
                console.print("  [yellow]Warning: save_stems requires model and device parameters[/yellow]")
        else:
            logging.info("DEBUG: save_stems has valid model and device - proceeding with stem separation")
            base_name = file_path.stem
            output_dir = file_path.parent
            temp_vocals_file = output_dir / f"{base_name}_temp_vocals.wav"

            logging.info(f"DEBUG: Calling _save_vocals_and_background with base_name='{base_name}', output_dir='{output_dir}'")
            if _save_vocals_and_background(file_path, output_dir, base_name, model, device, temp_vocals_file):
                logging.info("DEBUG: _save_vocals_and_background returned True - stem separation succeeded")
                if not QUIET_MODE:
                    console.print("  [green]Saved vocals and background stems[/green]")
            else:
                logging.info("DEBUG: _save_vocals_and_background returned False - stem separation failed")

    # Step 1: SpeechBrain MetricGAN+ enhancement
    if not HAS_SPEECHBRAIN:
        if not QUIET_MODE:
            console.print("  [yellow]Warning: SpeechBrain not available. Install with: pip install speechbrain[/yellow]")
            console.print("  [blue]Falling back to standard Demucs pipeline...[/blue]")
        return False

    # Step 1: Normalize the audio first for better SpeechBrain performance
    temp_prenorm_file = file_path.parent / f"{file_path.stem}_temp_prenorm.wav"
    if not run_ffmpeg_normalize(
        file_path, temp_prenorm_file, target_lufs,
        enhanced_cleaning=enhanced_cleaning, use_two_pass=use_two_pass_loudnorm,
        voice_consistent=voice_consistent
    ):
        return False
    if not QUIET_MODE:
        console.print("  [green]Step 1: Pre-normalized audio for optimal enhancement[/green]")

    # Step 2: Enhance the normalized audio with SpeechBrain
    if not run_speechbrain_enhance(temp_prenorm_file, temp_enhanced_file):
        if not QUIET_MODE:
            console.print("  [yellow]SpeechBrain enhancement failed, falling back to standard pipeline[/yellow]")
        # Clean up temp file
        if temp_prenorm_file.exists():
            temp_prenorm_file.unlink()
        return False
    if not QUIET_MODE:
        console.print("  [green]Step 2: Enhanced pre-normalized audio with SpeechBrain MetricGAN+[/green]")

    # Clean up intermediate file
    if temp_prenorm_file.exists():
        temp_prenorm_file.unlink()

    # Step 3: Final output preparation
    normalize_output = temp_normalized_file if trim_silence else output_file
    # Copy enhanced file to final output location (may need light post-processing)
    if temp_enhanced_file != normalize_output:
        try:
            shutil.copy2(temp_enhanced_file, normalize_output)
        except (OSError, PermissionError) as e:
            logging.error(f"Failed to copy enhanced file: {e}")
            return False
    if not QUIET_MODE:
        console.print("  [green]Step 3: Prepared final enhanced audio[/green]")

    # Step 3: Apply silence trimming if requested
    if trim_silence:
        if not run_silence_trimmer(temp_normalized_file, output_file, silence_threshold=silence_threshold, output_format=output_format):
            return False
        if not QUIET_MODE:
            console.print("  [green]Step 3: Trimmed extended silences[/green]")

    return True

def process_file_with_model(
    file_path: Path,
    target_lufs: float = -23.0,  # EBU R128 standard
    model=None,
    device: Optional[str] = None,
    keep_temp: bool = False,
    skip_demucs: bool = False,
    enhanced_cleaning: bool = False,
    python_restoration: bool = False,
    trim_silence: bool = False,
    silence_threshold: float = -30.0,
    intensive_cleanup: bool = False,
    use_dynaudnorm: bool = True,
    use_two_pass_loudnorm: bool = False,
    speechbrain_enhance: bool = True,
    save_stems: bool = False,
    stems_only: bool = False,
    overwrite: bool = False,
    output_format: str = "wav",
    voice_consistent: bool = False,
) -> bool:
    """Process a single audio file through the improved voice-first pipeline.

    IMPROVED PIPELINE ORDER:
    1. Extract vocals (Demucs) - separates voice from music/noise
    2. Post-AI cleanup - removes residual static, clicks, echo artifacts
    3. Smart normalization (dynaudnorm/voice-consistent) - preserves speech dynamics
    4. Silence trimming (optional) - natural pause lengths

    Key improvements address static, over-loud volume, and echo issues.

    Args:
        file_path: Input audio file path
        target_lufs: Target loudness for normalization (default -23 LUFS EBU R128 standard)
        model: Pre-loaded Demucs model (if using Demucs)
        device: Processing device
        keep_temp: Whether to keep temporary files for debugging
        skip_demucs: Skip vocal separation (normalize original audio)
        enhanced_cleaning: Enable legacy loudnorm cleaning (not recommended)
        python_restoration: Use Python-based restoration instead of Demucs
        trim_silence: Enable silence trimming after processing
        intensive_cleanup: Enable more aggressive post-AI artifact removal
        use_dynaudnorm: Use dynaudnorm instead of loudnorm (recommended)
        voice_consistent: Use voice-consistent normalization for uniform voice levels throughout audio

    Returns:
        bool: Success status
    """
    logging.info(f"DEBUG: audionorm.py process_file_with_model called with save_stems={save_stems}, speechbrain_enhance={speechbrain_enhance}")
    if not file_path.exists():
        logging.error(f"Input file not found: {file_path}")
        return False

    if file_path.suffix.lower() not in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
        logging.warning(f"Unsupported file format: {file_path.suffix}")
        return False

    base = file_path.with_suffix("")
    temp_vocals_file = base.with_name(base.name + "_temp_vocals.wav")
    temp_cleaned_file = base.with_name(base.name + "_temp_cleaned.wav")
    temp_enhanced_file = base.with_name(base.name + "_temp_enhanced.wav")
    temp_normalized_file = base.with_name(base.name + "_temp_normalized.wav")
    temp_silence_trimmed_file = base.with_name(base.name + "_temp_silence_trimmed.wav")

    # Choose output filename based on processing type and format
    if stems_only:
        # For stems-only mode, check if vocal and background stems already exist
        vocals_file = base.with_name(base.name + "_vocals.wav")
        background_file = base.with_name(base.name + "_background.wav")
        if vocals_file.exists() and background_file.exists() and not overwrite:
            if not QUIET_MODE:
                console.print(f"[yellow]Skipping {file_path.name} (stems already exist, use --overwrite to replace)[/yellow]")
            return "skipped"
        output_file = None  # No single output file for stems-only mode
    elif speechbrain_enhance:
        suffix = f"_speechbrain_enhanced.{output_format}"
        output_file = base.with_name(base.name + suffix)
    elif skip_demucs:
        suffix = f"_normalized.{output_format}"
        output_file = base.with_name(base.name + suffix)
    elif python_restoration:
        suffix = f"_restored_normalized.{output_format}"
        output_file = base.with_name(base.name + suffix)
    else:
        suffix = f"_voice_normalized.{output_format}"  # More accurate naming
        output_file = base.with_name(base.name + suffix)

    # Check if output already exists (skip for stems-only mode as it's handled above)
    if not stems_only and output_file.exists() and not overwrite:
        if not QUIET_MODE:
            console.print(f"[yellow]Skipping {file_path.name} (output exists, use --overwrite to replace)[/yellow]")
        return "skipped"

    try:
        # PIPELINE SELECTION: Choose the best enhancement method

        if VERBOSE_MODE or save_stems or stems_only:
            logging.info(f"Pipeline selection - speechbrain_enhance: {speechbrain_enhance}, skip_demucs: {skip_demucs}, python_restoration: {python_restoration}, save_stems: {save_stems}, stems_only: {stems_only}")
            logging.info(f"Model and device status: model={model is not None}, device={device}")
            if save_stems:
                logging.info("DEBUG: save_stems=True - will attempt stem separation")
            if stems_only:
                logging.info("DEBUG: stems_only=True - will only separate stems without processing")

        if stems_only:
            # STEMS-ONLY MODE: Only separate vocals and background, no processing
            logging.info("DEBUG: Entering stems-only pipeline")
            if model is None or device is None:
                logging.error("Stems-only mode requires Demucs model - loading model")
                model, device = load_demucs_model('htdemucs_ft')

            base_name = file_path.stem
            output_dir = file_path.parent

            # Use Demucs to separate all stems
            stem_paths = run_demucs_with_all_stems(file_path, output_dir, base_name, model, device)

            if stem_paths and "vocals" in stem_paths and "other" in stem_paths:
                # Create combined background (drums + bass + other)
                background_path = output_dir / f"{base_name}_background.wav"

                # Load drums, bass, and other stems to combine them
                try:
                    drums_wav, sr = _load_audio_for_demucs(stem_paths.get("drums", stem_paths["other"]))
                    bass_wav, _ = _load_audio_for_demucs(stem_paths.get("bass", stem_paths["other"]))
                    other_wav, _ = _load_audio_for_demucs(stem_paths["other"])

                    # Combine all non-vocal stems
                    background_combined = drums_wav + bass_wav + other_wav
                    background_mono = _convert_to_mono(background_combined)
                    _save_demucs_output(background_mono, background_path, sr)

                    if not QUIET_MODE:
                        console.print(f"  [green]Stems-only: Saved {base_name}_vocals.wav and {base_name}_background.wav[/green]")
                        console.print("  [green]Stems-only: Also saved individual stems (drums, bass, other)[/green]")

                    return True
                except Exception as e:
                    logging.error(f"Failed to combine background stems: {e}")
                    if not QUIET_MODE:
                        console.print("  [green]Stems-only: Saved individual stems only[/green]")
                    return True
            else:
                logging.error("Stems separation failed")
                return False
        elif speechbrain_enhance:
            logging.info("DEBUG: Entering SpeechBrain pipeline block")
            # Try SpeechBrain first, fallback to Demucs if it fails
            if _process_speechbrain_pipeline(
                file_path, output_file, temp_enhanced_file, temp_normalized_file,
                target_lufs, enhanced_cleaning, use_two_pass_loudnorm, trim_silence,
                silence_threshold, output_format, save_stems, model, device, voice_consistent
            ):
                logging.info("DEBUG: SpeechBrain pipeline completed successfully")
                pass  # SpeechBrain pipeline completed successfully
            else:
                logging.info("DEBUG: SpeechBrain pipeline failed, falling back to Demucs")
                # Fallback to Demucs pipeline
                if not QUIET_MODE:
                    console.print("  [blue]Falling back to Demucs pipeline...[/blue]")
                temp_files_dict = {
                    'temp_vocals_file': temp_vocals_file,
                    'temp_cleaned_file': temp_cleaned_file,
                    'temp_normalized_file': temp_normalized_file
                }
                processing_config_dict = {
                    'target_lufs': target_lufs,
                    'enhanced_cleaning': enhanced_cleaning,
                    'intensive_cleanup': intensive_cleanup,
                    'use_dynaudnorm': use_dynaudnorm,
                    'use_two_pass_loudnorm': use_two_pass_loudnorm,
                    'trim_silence': trim_silence,
                    'silence_threshold': silence_threshold,
                    'output_format': output_format,
                    'save_stems': save_stems,
                    'voice_consistent': voice_consistent
                }
                if not _process_demucs_pipeline(
                    file_path, output_file, temp_files_dict, model, device, processing_config_dict
                ):
                    return False
        elif skip_demucs:
            if not _process_skip_demucs_pipeline(
                file_path, output_file, temp_normalized_file, target_lufs,
                enhanced_cleaning, use_two_pass_loudnorm, trim_silence, silence_threshold, output_format,
                voice_consistent
            ):
                return False
        elif python_restoration:
            if not _process_python_restoration_pipeline(
                file_path, output_file, temp_vocals_file, temp_normalized_file,
                target_lufs, enhanced_cleaning, use_two_pass_loudnorm, trim_silence, silence_threshold, output_format,
                voice_consistent
            ):
                return False
        else:
            temp_files_dict = {
                'temp_vocals_file': temp_vocals_file,
                'temp_cleaned_file': temp_cleaned_file,
                'temp_normalized_file': temp_normalized_file
            }
            processing_config_dict = {
                'target_lufs': target_lufs,
                'enhanced_cleaning': enhanced_cleaning,
                'intensive_cleanup': intensive_cleanup,
                'use_dynaudnorm': use_dynaudnorm,
                'use_two_pass_loudnorm': use_two_pass_loudnorm,
                'trim_silence': trim_silence,
                'silence_threshold': silence_threshold,
                'output_format': output_format,
                'save_stems': save_stems,
                'voice_consistent': voice_consistent
            }
            if not _process_demucs_pipeline(
                file_path, output_file, temp_files_dict, model, device, processing_config_dict
            ):
                return False

        # Cleanup temporary files unless debugging
        if not keep_temp:
            for temp_file in [temp_vocals_file, temp_cleaned_file, temp_enhanced_file, temp_normalized_file, temp_silence_trimmed_file]:
                if temp_file.exists():
                    temp_file.unlink()

        # Verify output file was created (skip for stems-only mode)
        if stems_only:
            # For stems-only mode, verification was already done in the stems-only logic
            if not QUIET_MODE:
                console.print(f"  [green]Complete: Stems separated for {file_path.name}[/green]")
        else:
            if not output_file.exists():
                logging.error(f"Output file was not created: {output_file}")
                return False

            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            if not QUIET_MODE:
                console.print(f"  [green]Complete: {output_file.name} ({file_size_mb:.1f}MB)[/green]")
        return True
    except Exception as e:
        logging.error(f"Processing failed for {file_path}: {e}")
        # Cleanup on failure
        for temp_file in [temp_vocals_file, temp_cleaned_file, temp_enhanced_file, temp_normalized_file, temp_silence_trimmed_file, output_file]:
            if temp_file.exists():
                temp_file.unlink()
        return False

def find_audio_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find audio files in the given path."""
    extensions = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    audio_files = []

    if path.is_file():
        if path.suffix.lower() in extensions:
            audio_files.append(path)
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        for ext in extensions:
            audio_files.extend(path.glob(f"{pattern}{ext}"))
            audio_files.extend(path.glob(f"{pattern}{ext.upper()}"))

    return sorted(audio_files)


def process_files_with_progress(audio_files: List[Path], **kwargs) -> tuple[int, int]:
    """Process files with rich progress bars and status updates.
    
    Returns:
        tuple: (success_count, skipped_count)
    """
    global QUIET_MODE
    
    success_count = 0
    skipped_count = 0
    skip_demucs = kwargs.get('skip_demucs', False)
    python_restoration = kwargs.get('python_restoration', False)
    
    # Load Demucs model once if needed
    model = None
    device = None
    save_stems = kwargs.get('save_stems', False)
    stems_only = kwargs.get('stems_only', False)
    if (not skip_demucs and not python_restoration) or save_stems or stems_only:
        if save_stems:
            logging.info("DEBUG: Loading Demucs model because save_stems=True")
        elif stems_only:
            logging.info("DEBUG: Loading Demucs model because stems_only=True")
        model, device = load_demucs_model(kwargs.get('model_name', 'htdemucs_ft'), kwargs.get('device'))
        kwargs['model'] = model
        kwargs['device'] = device
        logging.info(f"DEBUG: Model loaded - model={model is not None}, device={device}")
    else:
        logging.info(f"DEBUG: Skipping model load - skip_demucs={skip_demucs}, python_restoration={python_restoration}, save_stems={save_stems}, stems_only={stems_only}")
    
    if QUIET_MODE:
        # Minimal output mode
        for file_path in audio_files:
            # Create clean kwargs for process_file_with_model
            process_kwargs = {k: v for k, v in kwargs.items() if k != 'model_name'}
            result = process_file_with_model(file_path, **process_kwargs)
            if result == "skipped":
                skipped_count += 1
                console.print(f"Skipped: {file_path.name} (output exists)")
            elif result:
                success_count += 1
                console.print(f"Complete: {file_path.name}")
            else:
                console.print(f"Failed: {file_path.name}")
    else:
        # Rich progress bar mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[filename]}"),
            BarColumn(complete_style="green"),
            TaskProgressColumn(),
            TextColumn("[bold green]{task.fields[stage]}"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        ) as progress:
            overall_task = progress.add_task(
                "Processing files...", 
                total=len(audio_files), 
                filename="Overall Progress",
                stage=""
            )
            
            for i, file_path in enumerate(audio_files, 1):
                # Update overall progress
                progress.update(
                    overall_task, 
                    completed=i-1,
                    filename=f"File {i}/{len(audio_files)}: {file_path.name[:30]}...",
                    stage="Audio: Starting..."
                )
                
                # Add individual file task
                file_task = progress.add_task(
                    f"Processing {file_path.name}",
                    total=100,
                    filename=file_path.name,
                    stage="Voice: Extracting"
                )
                
                # Simulate stages for progress visualization
                progress.update(file_task, completed=10, stage="Voice: Starting")
                
                # Process the file
                # Create clean kwargs for process_single_file_with_progress  
                process_kwargs = {k: v for k, v in kwargs.items() if k != 'model_name'}
                result = process_single_file_with_progress(file_path, progress, file_task, **process_kwargs)
                
                if result == "skipped":
                    skipped_count += 1
                    progress.update(file_task, completed=100, stage="Skipped")
                elif result:
                    success_count += 1
                    progress.update(file_task, completed=100, stage="Complete")
                else:
                    progress.update(file_task, completed=100, stage="Failed")
                
                # Small delay to show progress
                import time
                time.sleep(0.1)
                
                # Remove individual file task after completion
                progress.remove_task(file_task)
            
            # Final update
            processed_count = success_count + skipped_count
            progress.update(
                overall_task, 
                completed=len(audio_files),
                filename=f"Processed {processed_count}/{len(audio_files)} files ({success_count} new, {skipped_count} skipped)",
                stage="Done"
            )
    
    return success_count, skipped_count

def process_single_file_with_progress(file_path: Path, progress, task_id, **kwargs) -> bool:
    """Process a single file with progress updates using voice-first pipeline."""
    try:
        progress.update(task_id, completed=10, stage="Starting")

        # Call the main processing function with all the pipeline logic
        result = process_file_with_model(file_path, **kwargs)

        if result:
            progress.update(task_id, completed=100, stage="Complete")
        else:
            progress.update(task_id, completed=100, stage="Failed")

        return result

    except Exception as e:
        progress.update(task_id, completed=100, stage="Failed")
        if VERBOSE_MODE:
            logging.error(f"Error processing {file_path}: {e}")
        return False


def _process_single_file_with_progress_old(file_path: Path, progress, task_id, **kwargs) -> bool:
    """OLD IMPLEMENTATION - Process a single file with progress updates using voice-first pipeline."""
    try:
        # Check if output already exists
        base = file_path.with_suffix("")
        skip_demucs = kwargs.get('skip_demucs', False)
        python_restoration = kwargs.get('python_restoration', False)
        output_format = kwargs.get('output_format', 'wav')

        # Choose output filename based on processing type and format
        if kwargs.get('speechbrain_enhance', False):
            suffix = f"_speechbrain_enhanced.{output_format}"
        elif skip_demucs:
            suffix = f"_normalized.{output_format}"
        elif python_restoration:
            suffix = f"_restored_normalized.{output_format}"
        else:
            suffix = f"_voice_normalized.{output_format}"  # Updated to match new naming
        
        output_file = base.with_name(base.name + suffix)
        
        if output_file.exists() and not kwargs.get('overwrite', False):
            progress.update(task_id, completed=100, stage="Skipped (exists)")
            return "skipped"
        
        # NEW PIPELINE: Voice extraction first, then normalization
        temp_vocals_file = base.with_name(base.name + "_temp_vocals.wav")
        temp_normalized_file = base.with_name(base.name + "_temp_normalized.wav")
        
        if skip_demucs:
            # Stage 1: Direct normalization (20-80% or 20-90% if no silence trimming)
            progress.update(task_id, completed=30, stage="Normalizing")
            
            # Determine output file based on silence trimming
            trim_silence = kwargs.get('trim_silence', False)
            normalize_output = temp_normalized_file if trim_silence else output_file
            
            if not run_ffmpeg_normalize(
                file_path, normalize_output,
                kwargs.get('target_lufs', -18.0),
                enhanced_cleaning=kwargs.get('enhanced_cleaning', False),
                use_two_pass=kwargs.get('use_two_pass_loudnorm', False)
            ):
                progress.update(task_id, completed=100, stage="Failed: Norm Failed")
                return False
            
            if trim_silence:
                progress.update(task_id, completed=70, stage="Normalized")
                # Stage 2: Trim silences (70-90%)
                progress.update(task_id, completed=80, stage="Trimming Silence")
                if not run_silence_trimmer(temp_normalized_file, output_file, silence_threshold=kwargs.get('silence_threshold', -30.0), output_format=output_format):
                    progress.update(task_id, completed=100, stage="Failed: Silence Failed")
                    return False
                progress.update(task_id, completed=90, stage="Silence Trimmed")
            else:
                progress.update(task_id, completed=90, stage="Normalized")
        elif python_restoration:
            # Stage 1: Python restoration (20-60%)
            progress.update(task_id, completed=30, stage="Restoring")
            if not python_audio_restoration(file_path, temp_vocals_file):
                progress.update(task_id, completed=100, stage="Failed: Restore Failed")
                return False
            progress.update(task_id, completed=60, stage="Restored")
            # Stage 2: Normalize restored audio (60-90%)
            progress.update(task_id, completed=70, stage="Normalizing")
            if not run_ffmpeg_normalize(
                temp_vocals_file, output_file,
                kwargs.get('target_lufs', -18.0),
                enhanced_cleaning=kwargs.get('enhanced_cleaning', False),
                use_two_pass=kwargs.get('use_two_pass_loudnorm', False)
            ):
                progress.update(task_id, completed=100, stage="Failed: Norm Failed")
                return False
            progress.update(task_id, completed=90, stage="Complete")
        else:
            # NEW PIPELINE: Voice first, then normalize
            # Stage 1: Extract vocals (20-60%)
            progress.update(task_id, completed=30, stage="Extracting Voice")
            if not run_demucs_denoise(
                file_path, temp_vocals_file, 
                kwargs.get('model'),
                kwargs.get('device')
            ):
                progress.update(task_id, completed=100, stage="Failed: Voice Failed")
                return False
            progress.update(task_id, completed=60, stage="Voice Extracted")
            
            # Stage 2: NEW - Post-AI cleanup (60-70%)
            temp_cleaned_file = base.with_name(base.name + "_temp_cleaned.wav")
            progress.update(task_id, completed=65, stage="Cleaning Artifacts")
            
            if not run_post_ai_cleanup(
                temp_vocals_file, temp_cleaned_file, 
                kwargs.get('intensive_cleanup', True)
            ):
                progress.update(task_id, completed=100, stage="Failed: Cleanup Failed")
                return False
            progress.update(task_id, completed=70, stage="Artifacts Cleaned")
            
            # Stage 3: Normalize cleaned voice (70-85% or 70-90% if no silence trimming)
            progress.update(task_id, completed=75, stage="Normalizing Voice")
            
            # Determine output file based on silence trimming
            trim_silence = kwargs.get('trim_silence', False)
            normalize_output = temp_normalized_file if trim_silence else output_file
            
            if not run_ffmpeg_normalize(
                temp_cleaned_file, normalize_output,
                kwargs.get('target_lufs', -18.0),
                use_dynaudnorm=kwargs.get('use_dynaudnorm', True),
                enhanced_cleaning=kwargs.get('enhanced_cleaning', False),
                use_two_pass=kwargs.get('use_two_pass_loudnorm', False)
            ):
                progress.update(task_id, completed=100, stage="Failed: Norm Failed")
                return False
            
            if trim_silence:
                progress.update(task_id, completed=85, stage="Voice Normalized")
                # Stage 4: Trim silences (85-90%)
                progress.update(task_id, completed=87, stage="Trimming Silence")
                if not run_silence_trimmer(temp_normalized_file, output_file, silence_threshold=kwargs.get('silence_threshold', -30.0), output_format=output_format):
                    progress.update(task_id, completed=100, stage="Failed: Silence Failed")
                    return False
                progress.update(task_id, completed=90, stage="Silence Trimmed")
            else:
                progress.update(task_id, completed=90, stage="Voice Normalized")
        
        # Cleanup
        if not kwargs.get('keep_temp', False):
            for temp_file in [temp_vocals_file, temp_cleaned_file, temp_normalized_file]:
                if temp_file.exists():
                    temp_file.unlink()
        
        # Verify output
        if not output_file.exists():
            progress.update(task_id, completed=100, stage="Failed: No Output")
            return False
        
        progress.update(task_id, completed=95, stage="Verifying")
        return True
        
    except Exception as e:
        progress.update(task_id, completed=100, stage="Failed: Error")
        if not QUIET_MODE:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")
        return False

def main():
    global QUIET_MODE, VERBOSE_MODE
    
    parser = argparse.ArgumentParser(
        description="Audio normalization and cleanup for TTS-generated audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s /path/to/audio/folder --recursive
  %(prog)s audio.wav --target-lufs -23 --model htdemucs
  %(prog)s folder/ --device cuda --keep-temp --quiet
        """,
    )

    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-18.0,
        help="Target loudness in LUFS (default: -18.0 optimized for voice content)",
    )
    parser.add_argument(
        "--model", default="htdemucs_ft", help="Demucs model (default: htdemucs_ft for better vocal separation)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Processing device (auto-detect if not specified)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Process directories recursively"
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary files for debugging"
    )
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        help="Skip Demucs denoising (FFmpeg normalization only)",
    )
    parser.add_argument(
        "--enhanced-cleaning",
        action="store_true",
        default=False,
        help="Enable enhanced FFmpeg cleaning (gentle)",
    )
    parser.add_argument(
        "--basic-cleaning",
        dest="enhanced_cleaning",
        action="store_false",
        help="Use basic cleaning only (faster processing)",
    )
    parser.add_argument(
        "--python-restoration",
        action="store_true",
        help="Use Python-based restoration (librosa + noisereduce) instead of Demucs",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output mode"
    )
    parser.add_argument(
        "--trim-silence", 
        action="store_true", 
        help="Trim extended silences (>1s) to natural pause lengths (0.2-0.8s)"
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-30.0,
        help="Silence detection threshold in dB (default: -30, lower=more sensitive)"
    )
    parser.add_argument(
        "--intensive-cleanup",
        dest="intensive_cleanup",
        action="store_true",
        help="Enable intensive post-AI cleanup (may cause reverb artifacts, disabled by default)"
    )
    parser.add_argument(
        "--use-loudnorm",
        dest="use_dynaudnorm",
        action="store_false",
        help="Use loudnorm instead of dynaudnorm (may sound over-processed)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip existing files)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["wav", "mp3"],
        default="wav",
        help="Output audio format (default: wav)"
    )
    parser.add_argument(
        "--two-pass-loudnorm",
        action="store_true",
        default=True,
        help="Use professional two-pass loudnorm for superior volume consistency (default: enabled)"
    )
    parser.add_argument(
        "--single-pass-loudnorm",
        dest="two_pass_loudnorm",
        action="store_false",
        help="Use single-pass loudnorm instead of two-pass (faster but less accurate)"
    )
    parser.add_argument(
        "--no-speechbrain",
        action="store_true",
        help="Disable SpeechBrain MTL-MIMIC voice enhancement (default: enabled)"
    )
    parser.add_argument(
        "--save-stems",
        action="store_true",
        help="Save all Demucs stems (vocals, drums, bass, other) when using Demucs pipeline"
    )
    parser.add_argument(
        "--stems-only",
        action="store_true",
        help="Only separate into vocal and background stems without any processing or normalization"
    )
    parser.add_argument(
        "--voice-consistent",
        action="store_true",
        help="Use voice-consistent normalization for uniform voice levels throughout audio (fixes level variations)"
    )

    args = parser.parse_args()

    # Set output modes
    QUIET_MODE = args.quiet
    VERBOSE_MODE = args.verbose
    
    # Set logging level
    if args.verbose:
        # Only enable DEBUG for our own logger, not third-party libraries
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Find audio files
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input path does not exist: {input_path}[/red]")
        sys.exit(1)

    audio_files = find_audio_files(input_path, args.recursive)

    if not audio_files:
        console.print(f"[red]Error: No audio files found in: {input_path}[/red]")
        sys.exit(1)

    # Display header with file count
    if not QUIET_MODE:
        console.print(Panel.fit(
            f"[bold cyan]Audio Normalization & Cleanup[/bold cyan]\n"
            f"Found {len(audio_files)} audio file(s) to process",
            border_style="blue"
        ))

    # Process files with progress bars
    success_count, skipped_count = process_files_with_progress(
        audio_files,
        target_lufs=args.target_lufs,
        model_name=args.model,
        device=args.device,
        keep_temp=args.keep_temp,
        skip_demucs=args.skip_demucs,
        enhanced_cleaning=args.enhanced_cleaning,
        python_restoration=args.python_restoration,
        trim_silence=args.trim_silence,
        silence_threshold=args.silence_threshold,
        intensive_cleanup=args.intensive_cleanup,
        use_dynaudnorm=args.use_dynaudnorm,
        use_two_pass_loudnorm=args.two_pass_loudnorm,
        speechbrain_enhance=not args.no_speechbrain,
        save_stems=args.save_stems,
        stems_only=args.stems_only,
        overwrite=args.overwrite,
        output_format=args.format,
        voice_consistent=args.voice_consistent,
    )

    # Summary
    failed_count = len(audio_files) - success_count - skipped_count
    
    if not QUIET_MODE:
        # Create summary table
        table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Successfully processed", str(success_count))
        table.add_row("Skipped (already exist)", str(skipped_count))
        table.add_row("Failed", str(failed_count))
        table.add_row("Total", str(len(audio_files)))
        
        console.print(table)
        
        if failed_count == 0 and success_count > 0:
            console.print("[bold green]All new files processed successfully![/bold green]")
        elif failed_count == 0 and skipped_count > 0 and success_count == 0:
            console.print("[bold blue]All files already processed (use --overwrite to reprocess)[/bold blue]")
        elif failed_count == 0:
            console.print("[bold green]Processing completed successfully![/bold green]")
        else:
            console.print(f"[bold yellow]Warning: {failed_count} files failed to process[/bold yellow]")
    else:
        console.print(f"Results: {success_count} processed, {skipped_count} skipped, {failed_count} failed")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    import atexit
    import os

    # Suppress any logging during exit cleanup if not in verbose mode
    def suppress_exit_logs():
        if not VERBOSE_MODE:
            # Redirect stderr to devnull during cleanup
            try:
                sys.stderr = open(os.devnull, 'w')
            except:
                pass

    atexit.register(suppress_exit_logs)
    main()
