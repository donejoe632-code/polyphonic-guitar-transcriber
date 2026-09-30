"""
Core Digital Signal Processing (DSP) routines for polyphonic guitar transcription.

Includes Short-Time Fourier Transform (STFT), Harmonic-Percussive Separation (HPS),
Constant-Q log-frequency filterbanks, spectral peak detection with parabolic interpolation,
and RMS energy envelopes.
"""

from typing import List, Tuple, Optional
import numpy as np
import scipy.signal


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure the audio array is single-channel (1D mono).
    If multi-channel, averages channels.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Standard convention: (samples, channels) e.g. (N, 2)
        if audio.shape[1] <= audio.shape[0]:
            return np.mean(audio, axis=1)
        else:
            return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unsupported audio dimension: {audio.ndim}")


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) magnitude spectrum.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio waveform.
    sample_rate : int
        Audio sampling rate in Hz.
    n_fft : int, default=4096
        FFT window length. Large window (e.g. 4096 at 44.1kHz or 22.05kHz)
        provides fine sub-semitone frequency resolution for low guitar strings (E2 = 82.4Hz).
    hop_length : int, default=1024
        Step size between successive analysis frames.
    window : str, default="hann"
        Analysis window function name.

    Returns
    -------
    frequencies : np.ndarray
        Array of sample frequencies (shape: [n_freq_bins]).
    time_bins : np.ndarray
        Array of segment segment center times in seconds (shape: [n_frames]).
    magnitude_spec : np.ndarray
        2D non-negative magnitude spectrum (shape: [n_freq_bins, n_frames]).
    """
    mono_audio = ensure_mono(audio)
    
    # Pad if audio is shorter than n_fft
    if len(mono_audio) < n_fft:
        pad_width = n_fft - len(mono_audio)
        mono_audio = np.pad(mono_audio, (0, pad_width), mode="constant")

    noverlap = n_fft - hop_length
    freqs, times, Zxx = scipy.signal.stft(
        mono_audio,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=True,
    )
    magnitude_spec = np.abs(Zxx)
    return freqs, times, magnitude_spec


def harmonic_percussive_separation(
    magnitude_spec: np.ndarray,
    kernel_size: int = 31,
    margin: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate harmonic (tonal chord sustained resonance) from percussive (pick transient attacks)
    components using 2D median filtering.

    Guitar pick attacks create broadband vertical streaks in spectrograms that corrupt
    chroma profiles. Removing percussive pick noise markedly improves chord classification precision.

    Parameters
    ----------
    magnitude_spec : np.ndarray
        Magnitude spectrogram of shape (n_freqs, n_frames).
    kernel_size : int, default=31
        Median filter kernel length (must be odd).
    margin : float, default=1.0
        Separation margin factor.

    Returns
    -------
    harmonic_spec : np.ndarray
        Harmonic component magnitude spectrogram.
    percussive_spec : np.ndarray
        Percussive component magnitude spectrogram.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Time-direction median filter extracts sustained horizontal lines (harmonics)
    # Frequency-direction median filter extracts vertical bursts (percussive attacks)
    H_med = scipy.signal.medfilt2d(magnitude_spec, kernel_size=[1, kernel_size])
    P_med = scipy.signal.medfilt2d(magnitude_spec, kernel_size=[kernel_size, 1])

    # Soft mask with Wiener-like power weighting
    eps = 1e-9
    H_p = (H_med * margin) ** 2
    P_p = (P_med * margin) ** 2
    total = H_p + P_p + eps

    mask_h = H_p / total
    mask_p = P_p / total

    harmonic_spec = magnitude_spec * mask_h
    percussive_spec = magnitude_spec * mask_p
    return harmonic_spec, percussive_spec


def spectral_peaks(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    min_freq: float = 70.0,
    max_freq: float = 5000.0,
    threshold_db: float = -60.0,
    max_peaks: int = 40,
) -> List[Tuple[float, float]]:
    """
    Detect local spectral peaks with parabolic interpolation for sub-bin frequency accuracy.

    Parameters
    ----------
    spectrum : np.ndarray
        1D magnitude spectrum for a single time frame.
    freqs : np.ndarray
        Frequency bin values in Hz.
    min_freq : float, default=70.0
        Minimum frequency to consider (just below low E string 82.41 Hz).
    max_freq : float, default=5000.0
        Maximum frequency to consider.
    threshold_db : float, default=-60.0
        Relative dB threshold below maximum peak to discard noise.
    max_peaks : int, default=40
        Maximum number of peaks to return.

    Returns
    -------
    peaks : List[Tuple[float, float]]
        List of (frequency_hz, refined_magnitude) tuples sorted by magnitude descending.
    """
    valid_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    if len(valid_idx) < 3:
        return []

    sub_spec = spectrum[valid_idx]
    sub_freqs = freqs[valid_idx]

    max_val = np.max(sub_spec) if len(sub_spec) > 0 else 0.0
    if max_val <= 1e-9:
        return []

    min_val = max_val * (10.0 ** (threshold_db / 20.0))
    peak_indices, _ = scipy.signal.find_peaks(sub_spec, height=min_val, distance=2)

    peaks: List[Tuple[float, float]] = []
    df = sub_freqs[1] - sub_freqs[0] if len(sub_freqs) > 1 else 1.0

    for idx in peak_indices:
        if 0 < idx < len(sub_spec) - 1:
            # Parabolic interpolation on log magnitude
            alpha = np.log(max(sub_spec[idx - 1], 1e-9))
            beta = np.log(max(sub_spec[idx], 1e-9))
            gamma = np.log(max(sub_spec[idx + 1], 1e-9))

            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                delta = np.clip(delta, -1.0, 1.0)
            else:
                delta = 0.0

            refined_freq = sub_freqs[idx] + delta * df
            refined_mag = np.exp(beta - 0.25 * (alpha - gamma) * delta)
            peaks.append((float(refined_freq), float(refined_mag)))
        else:
            peaks.append((float(sub_freqs[idx]), float(sub_spec[idx])))

    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks[:max_peaks]


def create_cqt_filterbank(
    freqs: np.ndarray,
    fmin: float = 82.41,
    fmax: float = 1318.51,
    bins_per_semitone: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a Constant-Q / semitone triangular filterbank matrix mapping linear STFT
    frequencies to guitar note pitches.

    Parameters
    ----------
    freqs : np.ndarray
        Array of linear STFT frequency bins in Hz.
    fmin : float, default=82.41
        Minimum frequency (E2, standard guitar 6th string open).
    fmax : float, default=1318.51
        Maximum frequency (E6, 24th fret on high E string).
    bins_per_semitone : int, default=1
        Sub-semitone resolution factor (1 for standard semitones).

    Returns
    -------
    filterbank : np.ndarray
        2D matrix of shape (n_pitch_bins, n_freqs).
    center_freqs : np.ndarray
        Center frequencies for each pitch bin in Hz.
    """
    # Number of semitones from fmin to fmax
    total_semitones = int(np.round(12.0 * np.log2(fmax / fmin)))
    n_bins = total_semitones * bins_per_semitone + 1

    semitone_steps = np.arange(n_bins) / bins_per_semitone
    center_freqs = fmin * (2.0 ** (semitone_steps / 12.0))

    filterbank = np.zeros((n_bins, len(freqs)), dtype=np.float32)

    # Ratio between adjacent bin centers
    r = 2.0 ** (1.0 / (12.0 * bins_per_semitone))

    for i, fc in enumerate(center_freqs):
        f_lower = fc / r
        f_upper = fc * r

        # Left slope
        left_mask = (freqs >= f_lower) & (freqs <= fc)
        if np.any(left_mask):
            filterbank[i, left_mask] = (freqs[left_mask] - f_lower) / (fc - f_lower + 1e-9)

        # Right slope
        right_mask = (freqs > fc) & (freqs <= f_upper)
        if np.any(right_mask):
            filterbank[i, right_mask] = (f_upper - freqs[right_mask]) / (f_upper - fc + 1e-9)

        # Normalize filter area
        total_energy = np.sum(filterbank[i])
        if total_energy > 0:
            filterbank[i] /= total_energy

    return filterbank, center_freqs


def energy_envelope(
    audio: np.ndarray,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> np.ndarray:
    """
    Compute the root-mean-square (RMS) energy envelope over sliding frames.

    Parameters
    ----------
    audio : np.ndarray
        1D mono audio signal.
    hop_length : int, default=512
        Hop length between frames.
    frame_length : int, default=2048
        Length of windowed frame.

    Returns
    -------
    rms_envelope : np.ndarray
        RMS energy for each frame.
    """
    mono = ensure_mono(audio)
    if len(mono) < frame_length:
        pad_width = frame_length - len(mono)
        mono = np.pad(mono, (0, pad_width), mode="constant")

    n_frames = (len(mono) - frame_length) // hop_length + 1
    rms = np.empty(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = mono[start : start + frame_length]
        rms[i] = np.sqrt(np.mean(frame**2) + 1e-12)

    return rms


def spectral_flux_onsets(
    magnitude_spec: np.ndarray,
    sample_rate: int,
    hop_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute onset detection function via positive half-wave rectified spectral flux.

    Parameters
    ----------
    magnitude_spec : np.ndarray
        Magnitude spectrogram of shape (n_freqs, n_frames).
    sample_rate : int
        Sample rate in Hz.
    hop_length : int
        Hop length between STFT frames.

    Returns
    -------
    onset_env : np.ndarray
        Normalized onset strength envelope.
    onset_times : np.ndarray
        Time timestamps in seconds corresponding to each onset frame.
    """
    # Difference along time axis
    diff = np.diff(magnitude_spec, axis=1)
    # Half-wave rectification (keep only positive energy increases)
    flux = np.sum(np.maximum(0.0, diff), axis=0)

    # Pad first frame so length matches magnitude_spec.shape[1]
    flux = np.concatenate([[0.0], flux])

    # Normalize to [0, 1]
    max_flux = np.max(flux)
    if max_flux > 1e-9:
        flux = flux / max_flux

    times = np.arange(len(flux)) * (hop_length / float(sample_rate))
    return flux, times
