#!/usr/bin/env python3

import argparse
import json
import re
import sys
import threading
import time
import pathlib

from rich.console import Console
from rich.live import Live
from rich.text import Text

_console = Console(stderr=True)



def _load_config():
    config_path = pathlib.Path.home() / ".config" / "readthis" / "config.json"
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}")
    # Fall back to config.json in the current directory for local dev.
    with open(config_path) as f:
        return json.load(f)


_config = _load_config()

# Audio output sample rate expected by the Kokoro model.
SAMPLE_RATE = 24000

# How many seconds a single left/right arrow keypress seeks.
SEEK_SECONDS = 5

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

URL_PATTERN = re.compile(r"^https?://")

VOICES = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
]

_generation_done = threading.Event()
_quit_requested = threading.Event()


def _get_text(input_arg):
    """Resolve the text to speak from stdin, clipboard, a URL, or a literal string."""
    # Piped input takes priority: `echo "hello" | python speak.py`
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if not text:
            raise SystemExit("No input from stdin.")
        return text

    # No argument → read whatever is on the clipboard.
    if input_arg is None:
        import pyperclip

        text = pyperclip.paste()
        if not text:
            raise SystemExit("Clipboard is empty.")
        return text

    # URL → fetch and extract the article body with trafilatura.
    if URL_PATTERN.match(input_arg):
        import trafilatura

        downloaded = trafilatura.fetch_url(input_arg)
        if downloaded is None:
            raise SystemExit("Failed to fetch URL.")
        text = trafilatura.extract(downloaded)
        if not text:
            raise SystemExit("Could not extract text from URL.")
        return text

    # Otherwise treat the argument as the literal text to speak.
    return input_arg


def _format_time(sample_count):
    """Convert a sample offset to a MM:SS display string."""
    total_seconds = int(sample_count / SAMPLE_RATE)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _play_streaming(audio_buffer):
    """Play audio from a growing buffer while generation continues in the background.

    How the shared buffer works
    ---------------------------
    `audio_buffer` is a one-element list that holds a numpy float32 array.
    The generation thread grows the audio by replacing `audio_buffer[0]` with a
    new, longer array (via np.concatenate). In CPython, assigning to a list index
    is protected by the GIL and is therefore atomic — the audio callback can read
    `audio_buffer[0]` at any time and will always see a valid, complete array,
    never a half-written one. This means no lock is needed in the hot callback path.

    The sounddevice OutputStream callback runs in a dedicated real-time audio thread.
    It reads `playback_pos` samples ahead on every call to fill the hardware buffer.
    `playback_lock` protects `playback_pos` and `is_paused` between the callback
    thread and the main input-polling thread.
    """
    import queue
    import readchar
    import sounddevice as sd

    # On Unix, when stdin is a pipe readchar reads from it instead of the terminal.
    # Redirect stdin to /dev/tty so keypresses still work. On Windows, readchar uses
    # msvcrt which reads from the console directly — no redirect needed.
    # Falls back to simple blocking playback when no terminal exists (headless).
    import platform

    original_stdin = sys.stdin
    tty_stream = None
    if platform.system() != "Windows" and not sys.stdin.isatty():
        try:
            import io

            tty_stream = io.open("/dev/tty", "r")
            sys.stdin = tty_stream
        except OSError:
            _generation_done.wait()
            sd.play(audio_buffer[0], samplerate=SAMPLE_RATE)
            sd.wait()
            return

    playback_pos = [0]
    is_paused = [False]
    playback_lock = threading.Lock()
    seek_samples = SEEK_SECONDS * SAMPLE_RATE
    spinner_frame = [0]
    key_queue = queue.Queue()

    def audio_callback(outdata, frame_count, time_info, status):
        """Fill the sounddevice output buffer each cycle.

        Called from a real-time audio thread — must not block.
        Reads `frame_count` samples from `audio_buffer[0]` starting at
        `playback_pos[0]`. Outputs silence when paused, when buffering
        (playback has caught up to generation), or past the end of audio.
        """
        # Atomic read: always a valid numpy array, no lock needed (see docstring).
        current_buf = audio_buffer[0]
        available_samples = len(current_buf)

        with playback_lock:
            if is_paused[0]:
                outdata[:] = 0
                return

            current_sample = playback_pos[0]

            # Playback has reached or passed the available audio.
            if current_sample >= available_samples:
                outdata[:] = 0
                # If generation is also finished, auto-pause at the end.
                if _generation_done.is_set():
                    is_paused[0] = True
                # Otherwise output silence and wait for more audio to arrive.
                return

            end_sample = current_sample + frame_count

            if end_sample > available_samples:
                # Partial buffer: copy what exists, zero-pad the rest.
                samples_to_copy = available_samples - current_sample
                outdata[:samples_to_copy, 0] = current_buf[
                    current_sample:available_samples
                ]
                outdata[samples_to_copy:] = 0
                playback_pos[0] = available_samples
            else:
                # Full buffer: copy exactly frame_count samples.
                outdata[:, 0] = current_buf[current_sample:end_sample]
                playback_pos[0] = end_sample

    def make_status():
        with playback_lock:
            current_sample = playback_pos[0]
            currently_paused = is_paused[0]

        total_available_samples = len(audio_buffer[0])
        state_icon = "⏸" if currently_paused else "▶"

        if _generation_done.is_set():
            generating_suffix = "  ✓"
        else:
            frame = SPINNER_FRAMES[spinner_frame[0] % len(SPINNER_FRAMES)]
            spinner_frame[0] += 1
            generating_suffix = f"  {frame}"

        return Text(
            f"{state_icon} "
            f"{_format_time(current_sample)} / {_format_time(total_available_samples)}"
            f"  [space] pause  [←/→] ±{SEEK_SECONDS}s  [q] quit"
            f"{generating_suffix}"
        )

    def key_reader():
        try:
            while not _quit_requested.is_set():
                key_queue.put(readchar.readkey())
        except Exception:
            pass

    # Keyboard input is handled by readchar in a dedicated daemon thread, which puts
    # each keypress into a queue. The main thread drains that queue with a 0.25s timeout
    threading.Thread(target=key_reader, daemon=True).start()

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()

    with Live(
        make_status(), console=_console, auto_refresh=False, transient=True
    ) as live:
        try:
            while True:
                try:
                    key = key_queue.get(timeout=0.25)
                except queue.Empty:
                    live.update(make_status())
                    live.refresh()
                    continue

                if key == readchar.key.SPACE:
                    with playback_lock:
                        is_paused[0] = not is_paused[0]
                elif key == readchar.key.LEFT:
                    with playback_lock:
                        playback_pos[0] = max(0, playback_pos[0] - seek_samples)
                elif key == readchar.key.RIGHT:
                    with playback_lock:
                        samples_available = len(audio_buffer[0])
                        playback_pos[0] = min(
                            max(samples_available - 1, 0),
                            playback_pos[0] + seek_samples,
                        )
                elif key in ("q", "Q", readchar.key.CTRL_C):
                    _quit_requested.set()
                    break

                live.update(make_status())
                live.refresh()
        finally:
            stream.stop()
            stream.close()
            if tty_stream is not None:
                sys.stdin = original_stdin
                tty_stream.close()


def speak(text, voice="Vivian", speed=1.0, lang="Chinese"):
    """Generate speech for `text` and stream it to the audio device.

    Architecture overview
    ---------------------
    A Qwen3-TTS model runs in a background thread and yields audio chunks streamingly.
    Each chunk is appended to a shared buffer, then _play_streaming starts consuming that buffer as soon
    as the first chunk is available — so the user hears audio almost immediately
    rather than waiting for the full text to be synthesised first.
    """
    import numpy as np
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit")

    # audio_buffer[0] starts empty and grows as the generation thread appends chunks.
    # Using a list lets the generation thread replace the reference (list[0] = new_array)
    # which is an atomic operation under CPython's GIL — the playback callback can
    # always read audio_buffer[0] and get a valid complete array.
    audio_buffer = [np.zeros(0, dtype=np.float32)]

    def generate_audio():
        """Run the TTS pipeline and append each audio chunk to the shared buffer."""
        # Use stream=True to get audio chunks as they are generated
        for result in model.generate_custom_voice(
            text=text,
            speaker=voice,
            language=lang,
            instruct="Clear and natural reading voice.",
            stream=True,
            streaming_interval=0.32
        ):
            if _quit_requested.is_set():
                break
            audio_chunk = np.array(result.audio)

            # Extend the buffer by creating a new concatenated array and swapping
            # the reference. The old array is garbage-collected by Python once no
            # other thread holds a reference to it.
            audio_buffer[0] = np.concatenate([audio_buffer[0], audio_chunk])
        _generation_done.set()

    generation_thread = threading.Thread(target=generate_audio, daemon=True)
    generation_thread.start()

    with _console.status("Generating..."):
        while len(audio_buffer[0]) == 0:
            time.sleep(0.08)

    _play_streaming(audio_buffer)

    # Ensure the generation thread has finished before returning, even if the
    # user quit playback early (the daemon flag means it won't block process exit).
    generation_thread.join()


def _config_path():
    return pathlib.Path.home() / ".config" / "readthis" / "config.json"


def _save_config(updates):
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        with open(path) as f:
            current = json.load(f)
    current.update(updates)
    with open(path, "w") as f:
        json.dump(current, f, indent=2)


def _run():
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        parser = argparse.ArgumentParser(
            prog="readthis config", description="Read or write config.json settings"
        )
        parser.add_argument("--voice", choices=VOICES, help="Set default voice")
        parser.add_argument(
            "--speed", type=float, help="Set default speech speed multiplier"
        )
        args = parser.parse_args(sys.argv[2:])
        updates = {}
        if args.voice is not None:
            updates["voice"] = args.voice
        if args.speed is not None:
            updates["speed"] = args.speed
        _save_config(updates)
        return

    parser = argparse.ArgumentParser(description="Text-to-speech using Qwen3-TTS")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Text to speak, URL to an article, or omit to read from clipboard",
    )
    parser.add_argument(
        "--voice",
        default=_config.get("voice", "Vivian"),
        choices=VOICES,
        help="Voice name (default: Vivian, or set in config.json)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=_config.get("speed", 1.0),
        help="Speech speed multiplier (default: 1.0, or set in config.json)",
    )
    parser.add_argument(
        "--lang", default=_config.get("lang", "Chinese"), help="Language code (default: Chinese)"
    )
    args = parser.parse_args()

    text = _get_text(args.input)
    # Replace single newlines with spaces, but preserve paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    speak(text, voice=args.voice, speed=args.speed, lang=args.lang)


def main():
    try:
        _run()
    except KeyboardInterrupt:
        _quit_requested.set()
        _console.print()
        # Exit 130: the Unix convention of 128 + signal number, where SIGINT
        # (Ctrl-C) is signal 2. This reports the same status the shell would
        # have used had Python not intercepted the interrupt, so callers
        # checking $? see "interrupted by user" rather than a generic failure.
        raise SystemExit(130)


if __name__ == "__main__":
    main()
