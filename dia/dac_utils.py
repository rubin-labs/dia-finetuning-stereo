import torch
import dac


def encode_mono_channel(
    wav_1xS: torch.Tensor,
    dac_model: dac.DAC,
    sample_rate: int = 44100,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.long,
) -> torch.Tensor:
    """
    Encode a single mono channel into DAC codebook tokens.

    Args:
        wav_1xS: Audio tensor with shape (1, samples)
        dac_model: Loaded DAC model
        sample_rate: Sample rate expected by the DAC preprocess step
        device: Optional device override; defaults to the model's device
        dtype: Optional output dtype (e.g., torch.long or torch.int16). If None, leaves dtype unchanged.
    """
    target_device = device or next(dac_model.parameters()).device
    audio_tensor = dac_model.preprocess(
        wav_1xS.unsqueeze(0).to(target_device),  # -> (1, 1, S)
        sample_rate,
    )
    _, enc, *_ = dac_model.encode(audio_tensor, n_quantizers=None)  # (1, 9, T)
    enc = enc.squeeze(0).transpose(0, 1).contiguous()  # (T, 9)
    return enc if dtype is None else enc.to(dtype)


def encode_waveform_stereo(
    waveform: torch.Tensor,
    dac_model: dac.DAC,
    sample_rate: int = 44100,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.long,
    duplicate_mono: bool = True,
) -> torch.Tensor:
    """
    Encode mono or stereo waveform to DAC codebook tokens.

    - Stereo: returns cat([L_codes, R_codes], dim=1) with shape (T, 18)
    - Mono: duplicates to stereo when duplicate_mono=True to match training expectations
    """
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform of shape (channels, samples); got {waveform.shape}")

    num_channels = waveform.shape[0]
    if num_channels > 2:
        raise ValueError(f"Waveform has {num_channels} channels. Only mono or stereo supported.")

    if num_channels == 2:
        codes_L = encode_mono_channel(
            waveform[0:1, :],
            dac_model=dac_model,
            sample_rate=sample_rate,
            device=device,
            dtype=dtype,
        )
        codes_R = encode_mono_channel(
            waveform[1:2, :],
            dac_model=dac_model,
            sample_rate=sample_rate,
            device=device,
            dtype=dtype,
        )
        return torch.cat([codes_L, codes_R], dim=1)  # (T, 18)

    mono_codes = encode_mono_channel(
        waveform[0:1, :],
        dac_model=dac_model,
        sample_rate=sample_rate,
        device=device,
        dtype=dtype,
    )
    if duplicate_mono:
        return torch.cat([mono_codes, mono_codes.clone()], dim=1)  # (T, 18)
    return mono_codes  # (T, 9)
