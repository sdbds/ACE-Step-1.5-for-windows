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
    refer_audio_hidden_states_packed=None,
    refer_audio_order_mask=None,
):
    """Run model encoder to get hidden states and attention mask."""
    if refer_audio_hidden_states_packed is None and refer_audio_order_mask is None:
        cache_key = (device, dtype)
        if cache_key not in _REFER_AUDIO_CACHE:
            _REFER_AUDIO_CACHE[cache_key] = (
                torch.zeros(1, 1, 64, device=device, dtype=dtype),
                torch.zeros(1, device=device, dtype=torch.long),
            )
        refer_audio_hidden, refer_audio_order_mask = _REFER_AUDIO_CACHE[cache_key]
        refer_audio_hidden.zero_()
        refer_audio_order_mask.zero_()
    else:
        refer_audio_hidden = refer_audio_hidden_states_packed
        if refer_audio_hidden is None:
            refer_audio_hidden = torch.zeros(1, 1, 64, device=device, dtype=dtype)
        else:
            if refer_audio_hidden.dim() == 2:
                refer_audio_hidden = refer_audio_hidden.unsqueeze(0)
            refer_audio_hidden = refer_audio_hidden.to(device=device, dtype=dtype)
            if refer_audio_hidden.shape[0] < 1:
                refer_audio_hidden = refer_audio_hidden.expand(1, -1, -1)

        if refer_audio_order_mask is None:
            refer_audio_order_mask = torch.zeros(
                refer_audio_hidden.shape[0],
                device=device,
                dtype=torch.long,
            )
        else:
            refer_audio_order_mask = refer_audio_order_mask.to(device=device, dtype=torch.long)

    encoder_hidden_states, encoder_attention_mask = model.encoder(
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
        refer_audio_order_mask=refer_audio_order_mask,
    )

    return encoder_hidden_states, encoder_attention_mask
