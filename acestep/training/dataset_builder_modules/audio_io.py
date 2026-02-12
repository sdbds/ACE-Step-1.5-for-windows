import os
from typing import Tuple

from loguru import logger


def _read_text_file(path: str) -> Tuple[str, bool]:
    """Read a text file; return (content.strip(), True) if present and non-empty."""
    if not os.path.exists(path):
        return "", False
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content, True
        return "", False
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return "", False


def load_caption_file(audio_path: str) -> Tuple[str, bool]:
    """Load caption from <basename>.caption.txt (explicit convention)."""
    base_path = os.path.splitext(audio_path)[0]
    caption_path = base_path + ".caption.txt"
    content, ok = _read_text_file(caption_path)
    if ok:
        logger.debug(f"Loaded caption from {caption_path}")
    return content, ok


def load_lyrics_file(audio_path: str) -> Tuple[str, bool]:
    """Load lyrics from <basename>.lyrics.txt, then fallback to <basename>.txt for backward compat."""
    base_path = os.path.splitext(audio_path)[0]
    for suffix in (".lyrics.txt", ".txt"):
        path = base_path + suffix
        content, ok = _read_text_file(path)
        if ok:
            if suffix == ".lyrics.txt":
                logger.debug(f"Loaded lyrics from {path}")
            else:
                logger.debug(f"Loaded lyrics from {path} (legacy .txt)")
            return content, True
    return "", False


def get_audio_duration(audio_path: str) -> int:
    """Get the duration of an audio file in seconds."""
    # Primary: torchcodec (ships with torchaudio >=2.9, supports all ffmpeg formats)
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(audio_path)
        if audio is not None:
            return int(audio.info.length)
    except Exception as e:
        logger.debug(f"torchcodec failed for {audio_path}: {e}, trying soundfile")
    # Fallback: soundfile (fast for wav/flac/ogg)
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return int(info.duration)
    except Exception as e:
        logger.warning(f"Failed to get duration for {audio_path}: {e}")
    return 0
