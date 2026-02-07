import torch

# Pre-allocated placeholder tensors for text2music (no refer audio).
# Lazily initialised on first call and cached across invocations so we
# avoid creating tiny GPU tensors on every sample.
_REFER_AUDIO_CACHE: dict = {}


def clear_refer_audio_cache():
    _REFER_AUDIO_CACHE.clear()


def run_encoder(
    model,
    text_hidden_states,
    text_attention_mask,
    lyric_hidden_states,
    lyric_attention_mask,
    device,
    dtype,
):
    """Run model encoder to get hidden states and attention mask."""
    cache_key = (device, dtype)
    if cache_key not in _REFER_AUDIO_CACHE:
        _REFER_AUDIO_CACHE[cache_key] = (
            torch.zeros(1, 1, 64, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=torch.long),
        )
    refer_audio_hidden, refer_audio_order_mask = _REFER_AUDIO_CACHE[cache_key]

    refer_audio_hidden.zero_()
    refer_audio_order_mask.zero_()

    encoder_hidden_states, encoder_attention_mask = model.encoder(
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
        refer_audio_order_mask=refer_audio_order_mask,
    )

    return encoder_hidden_states, encoder_attention_mask
