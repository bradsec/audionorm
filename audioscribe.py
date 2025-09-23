#!/usr/bin/env python3
"""
audioscribe.py - High-quality audio transcription with Faster-Whisper

Processes individual audio files or entire directories to generate accurate
text transcriptions using state-of-the-art Faster-Whisper models.

Usage:
    python audioscribe.py file.wav
    python audioscribe.py /path/to/folder
    python audioscribe.py file.wav --model large-v2 --language en
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# Check for required dependencies
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    logging.error("faster-whisper not available. Install with: pip install faster-whisper")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table

# Global console for rich output
console = Console()

# Configure logging with suppressed output by default
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING to reduce noise
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Global flags for output control
QUIET_MODE = False
VERBOSE_MODE = False

# Available Faster-Whisper models
AVAILABLE_MODELS = {
    'tiny': 'Fastest, least accurate (~39 MB)',
    'tiny.en': 'Tiny English-only (~39 MB)', 
    'base': 'Fast, good accuracy (~74 MB)',
    'base.en': 'Base English-only (~74 MB)',
    'small': 'Balanced speed/quality (~244 MB)',
    'small.en': 'Small English-only (~244 MB)',
    'medium': 'High quality (~769 MB)',
    'medium.en': 'Medium English-only (~769 MB)',
    'large-v1': 'Highest quality v1 (~1550 MB)',
    'large-v2': 'Best overall choice (~1550 MB)',
    'large-v3': 'Latest model (~1550 MB)',
    'distil-large-v2': 'Fast large model (~756 MB)',
    'distil-large-v3': 'Fastest large model (~756 MB)'
}

def load_whisper_model(model_name: str = "large-v2", device: Optional[str] = None) -> WhisperModel:
    """Load Faster-Whisper model once for reuse across multiple files.
    
    Args:
        model_name: Whisper model to use
        device: Processing device (auto-detected if None)
        
    Returns:
        WhisperModel: Loaded model for reuse
    """
    if not HAS_FASTER_WHISPER:
        raise ImportError("faster-whisper not available. Install with: pip install faster-whisper")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        
    compute_type = "float16" if device == "cuda" else "int8"
    
    if not QUIET_MODE:
        console.print(f"  [cyan]Loading Whisper model '{model_name}' on {device}...[/cyan]")
        
    try:
        model = WhisperModel(
            model_name, 
            device=device, 
            compute_type=compute_type,
            download_root=None,  # Use default cache location
            local_files_only=False  # Allow downloading if not cached
        )
        if VERBOSE_MODE:
            logging.info(f"Successfully loaded {model_name} model on {device} with {compute_type}")
        return model
    except Exception as e:
        console.print(f"  [red]Error loading model '{model_name}': {e}[/red]")
        if model_name != "base":
            console.print("  [yellow]Falling back to 'base' model...[/yellow]")
            return WhisperModel("base", device=device, compute_type=compute_type)
        else:
            raise

def transcribe_audio_file(
    input_file: Path,
    output_file: Path, 
    model: WhisperModel,
    language: Optional[str] = None,
    task: str = "transcribe",
    include_timestamps: bool = True,
    beam_size: int = 5
) -> bool:
    """Transcribe a single audio file using Faster-Whisper.
    
    Args:
        input_file: Input audio file path
        output_file: Output text file path
        model: Pre-loaded Whisper model
        language: Target language (auto-detect if None)
        task: 'transcribe' or 'translate'
        include_timestamps: Whether to include timestamps in output
        beam_size: Beam search size for better accuracy
        
    Returns:
        bool: Success status
    """
    try:
        if VERBOSE_MODE:
            console.print(f"  [blue]Processing: {input_file.name}[/blue]")
            
        # Transcribe the audio file
        segments, info = model.transcribe(
            str(input_file),
            beam_size=beam_size,
            language=language,
            task=task,
            condition_on_previous_text=False,  # Better for varied content
            temperature=0.0,  # Deterministic output
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        if VERBOSE_MODE:
            logging.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            logging.info(f"Audio duration: {info.duration:.2f} seconds")
        
        # Write transcription to file
        transcription_lines = []
        full_text = ""
        
        for segment in segments:
            if include_timestamps:
                timestamp_line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                transcription_lines.append(timestamp_line)
            full_text += segment.text + " "
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            if include_timestamps:
                # Write timestamped version
                f.write(f"# Transcription of {input_file.name}\n")
                f.write(f"# Language: {info.language} (confidence: {info.language_probability:.2f})\n") 
                f.write(f"# Duration: {info.duration:.2f} seconds\n\n")
                f.write("## Timestamped Transcription\n\n")
                for line in transcription_lines:
                    f.write(line + '\n')
                f.write("\n## Full Text\n\n")
                f.write(full_text.strip())
            else:
                # Write plain text only
                f.write(full_text.strip())
        
        if VERBOSE_MODE:
            logging.info(f"Transcription saved to: {output_file}")
            
        return True
        
    except Exception as e:
        logging.error(f"Transcription failed for {input_file}: {e}")
        return False

def process_file_with_model(
    file_path: Path,
    model: WhisperModel,
    language: Optional[str] = None,
    task: str = "transcribe", 
    include_timestamps: bool = True,
    beam_size: int = 5,
    overwrite: bool = False
) -> str:
    """Process a single audio file through the transcription pipeline.

    Args:
        file_path: Input audio file path
        model: Pre-loaded Whisper model
        language: Target language (auto-detect if None)
        task: 'transcribe' or 'translate'
        include_timestamps: Whether to include timestamps in output
        beam_size: Beam search size for accuracy

    Returns:
        bool: Success status
    """
    if not file_path.exists():
        logging.error(f"Input file not found: {file_path}")
        return "failed"

    if file_path.suffix.lower() not in [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".mp4"]:
        logging.warning(f"Unsupported file format: {file_path.suffix}")
        return "failed"

    # Generate output filename: test.wav -> test.txt
    base = file_path.with_suffix("")
    output_file = base.with_suffix(".txt")

    # Check if output already exists
    if output_file.exists() and not overwrite:
        if not QUIET_MODE:
            console.print(f"[yellow]Skipping {file_path.name} (transcript exists, use --overwrite to replace)[/yellow]")
        return "skipped"

    try:
        # Transcribe the audio file
        success = transcribe_audio_file(
            file_path, 
            output_file, 
            model,
            language=language,
            task=task,
            include_timestamps=include_timestamps,
            beam_size=beam_size
        )
        
        if success:
            # Verify output file was created
            if not output_file.exists():
                logging.error(f"Output file was not created: {output_file}")
                return "failed"
            
            file_size_kb = output_file.stat().st_size / 1024
            if not QUIET_MODE:
                console.print(f"  [green]Complete: {output_file.name} ({file_size_kb:.1f}KB)[/green]")
            return "success"
        else:
            return "failed"
        
    except Exception as e:
        logging.error(f"Processing failed for {file_path}: {e}")
        return "failed"

def find_audio_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find audio files in the given path."""
    extensions = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".mp4")
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
    
    # Load Whisper model once
    model = load_whisper_model(
        kwargs.get('model_name', 'large-v2'), 
        kwargs.get('device')
    )
    kwargs['model'] = model
    # Remove parameters that process_file_with_model doesn't need
    kwargs.pop('model_name', None)
    kwargs.pop('device', None)
    kwargs.pop('keep_temp', None)
    
    if QUIET_MODE:
        # Minimal output mode
        for file_path in audio_files:
            result = process_file_with_model(file_path, **kwargs)
            if result == "skipped":
                skipped_count += 1
                console.print(f"Skipped: {file_path.name} (transcript exists)")
            elif result == "success":
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
                    stage="Speech: Starting..."
                )
                
                # Add individual file task
                file_task = progress.add_task(
                    f"Processing {file_path.name}",
                    total=100,
                    filename=file_path.name,
                    stage="Speech: Loading"
                )
                
                # Process the file
                progress.update(file_task, completed=20, stage="Speech: Transcribing")
                
                result = process_single_file_with_progress(file_path, progress, file_task, **kwargs)
                
                if result == "skipped":
                    skipped_count += 1
                    progress.update(file_task, completed=100, stage="Skipped")
                elif result == "success":
                    success_count += 1
                    progress.update(file_task, completed=100, stage="Complete")
                else:
                    progress.update(file_task, completed=100, stage="Failed")
                
                # Small delay to show progress
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

def _check_output_exists(file_path: Path, progress, task_id) -> bool:
    """Check if output file already exists."""
    base = file_path.with_suffix("")
    output_file = base.with_suffix(".txt")
    
    if output_file.exists():
        progress.update(task_id, completed=100, stage="Skipped")
        return True
    return False

def _verify_output_file(file_path: Path, progress, task_id) -> bool:
    """Verify output file was created successfully."""
    base = file_path.with_suffix("")
    output_file = base.with_suffix(".txt")
    
    if not output_file.exists():
        progress.update(task_id, completed=100, stage="Failed: No Output")
        return False
    
    progress.update(task_id, completed=95, stage="Verifying")
    return True

def process_single_file_with_progress(file_path: Path, progress, task_id, **kwargs) -> str:
    """Process a single file with progress updates."""
    try:
        # Check if output already exists
        if _check_output_exists(file_path, progress, task_id):
            return "skipped"
        
        # Stage 1: Load and process (20-90%)
        progress.update(task_id, completed=40, stage="Speech: Processing")
        
        # Process the file
        result = process_file_with_model(file_path, **kwargs)
        
        if result == "failed":
            progress.update(task_id, completed=100, stage="Failed")
            return "failed"
        elif result == "skipped":
            return "skipped"
            
        progress.update(task_id, completed=90, stage="Speech: Complete")
        
        # Verify output
        if _verify_output_file(file_path, progress, task_id):
            return "success"
        else:
            return "failed"
        
    except Exception as e:
        progress.update(task_id, completed=100, stage="Failed: Error")
        if not QUIET_MODE:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")
        return "failed"

def _setup_logging_and_modes(args):
    """Configure logging and output modes."""
    global QUIET_MODE, VERBOSE_MODE
    QUIET_MODE = args.quiet
    VERBOSE_MODE = args.verbose
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.WARNING)

def _display_results_summary(audio_files, success_count, skipped_count):
    """Display processing results summary."""
    failed_count = len(audio_files) - success_count - skipped_count
    
    if not QUIET_MODE:
        table = Table(title="Transcription Summary", show_header=True, header_style="bold magenta")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Successfully transcribed", str(success_count))
        table.add_row("Skipped (already exist)", str(skipped_count))
        table.add_row("Failed", str(failed_count))
        table.add_row("Total", str(len(audio_files)))
        
        console.print(table)
        
        if failed_count == 0 and success_count > 0:
            console.print("[bold green]All new files transcribed successfully![/bold green]")
        elif failed_count == 0 and skipped_count > 0 and success_count == 0:
            console.print("[bold blue]All files already transcribed (use --overwrite to retranscribe)[/bold blue]")
        elif failed_count == 0:
            console.print("[bold green]Transcription completed successfully![/bold green]")
        else:
            console.print(f"[bold yellow]Warning: {failed_count} files failed to transcribe[/bold yellow]")
    else:
        console.print(f"Results: {success_count} transcribed, {skipped_count} skipped, {failed_count} failed")
    
    return failed_count

def main():
    global QUIET_MODE, VERBOSE_MODE
    
    parser = argparse.ArgumentParser(
        description="High-quality audio transcription with Faster-Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s /path/to/audio/folder --recursive
  %(prog)s audio.wav --model large-v3 --language en
  %(prog)s folder/ --device cuda --no-timestamps --quiet
        """,
    )

    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument(
        "--model", 
        default="large-v2", 
        choices=list(AVAILABLE_MODELS.keys()),
        help="Whisper model size (default: large-v2 for best balance)"
    )
    parser.add_argument(
        "--language",
        help="Source language code (auto-detect if not specified). Examples: en, es, fr, de, it, pt, ru, ja, ko, zh"
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform (default: transcribe)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Processing device (auto-detect if not specified)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (higher = more accurate but slower, default: 5)"
    )
    parser.add_argument(
        "--recursive", "-r", 
        action="store_true", 
        help="Process directories recursively"
    )
    parser.add_argument(
        "--no-timestamps", 
        action="store_true", 
        help="Disable timestamps in output (plain text only)"
    )
    parser.add_argument(
        "--keep-temp", 
        action="store_true", 
        help="Keep temporary files for debugging"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Minimal output mode"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true", 
        help="List available models and exit"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript files (default: skip existing files)"
    )

    # Check for --list-models before requiring input argument
    if '--list-models' in sys.argv:
        # Parse just to get the flag, ignore missing input
        args = argparse.Namespace(list_models=True)
    else:
        args = parser.parse_args()

    # List models and exit
    if getattr(args, 'list_models', False):
        console.print(Panel.fit(
            "[bold cyan]Available Faster-Whisper Models[/bold cyan]",
            border_style="blue"
        ))
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Description", style="white")
        
        for model, desc in AVAILABLE_MODELS.items():
            table.add_row(model, desc)
        
        console.print(table)
        sys.exit(0)

    # Set output modes and logging
    _setup_logging_and_modes(args)

    # Check dependencies
    if not HAS_FASTER_WHISPER:
        console.print("[red]Error: faster-whisper not installed. Install with:[/red]")
        console.print("[yellow]pip install faster-whisper[/yellow]")
        sys.exit(1)

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
            f"[bold cyan]Audio Transcription with Faster-Whisper[/bold cyan]\n"
            f"Found {len(audio_files)} audio file(s) to transcribe\n"
            f"Model: {args.model} | Task: {args.task}",
            border_style="blue"
        ))

    # Process files with progress bars
    success_count, skipped_count = process_files_with_progress(
        audio_files,
        model_name=args.model,
        device=args.device,
        language=args.language,
        task=args.task,
        include_timestamps=not args.no_timestamps,
        beam_size=args.beam_size,
        keep_temp=args.keep_temp,
        overwrite=args.overwrite
    )

    # Summary
    failed_count = _display_results_summary(audio_files, success_count, skipped_count)

    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()