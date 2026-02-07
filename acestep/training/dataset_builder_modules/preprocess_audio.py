import torchaudio

# Cache Resampler objects keyed by (orig_sr, target_sr).
# Construction of torchaudio.transforms.Resample is expensive (builds filter
# kernel), so reusing them across samples gives a significant speed-up.
_RESAMPLER_CACHE: dict = {}


def clear_resampler_cache():
    _RESAMPLER_CACHE.clear()


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    audio, sr = torchaudio.load(audio_path)

    if sr != target_sample_rate:
        cache_key = (sr, target_sample_rate)
        if cache_key not in _RESAMPLER_CACHE:
            _RESAMPLER_CACHE[cache_key] = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = _RESAMPLER_CACHE[cache_key](audio)

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
