"""
12-Semitone Chroma Pitch Class Profile (PCP) extractor optimized for guitar signals.

Features:
- Guitar-specific frequency windowing (E2 ~82.4 Hz to E6 ~1318.5 Hz)
- Logarithmic dynamic compression for distorted / saturated guitar tones
- Harmonic overtone attenuation (suppresses spurious 3rd/5th harmonic leakage)
- Frame-wise L1/L2 normalization and temporal smoothing
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import scipy.signal

from .dsp import compute_stft, harmonic_percussive_separation, ensure_mono

PITCH_CLASSES: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

NOTE_TO_PITCH_CLASS = {note: i for i, note in enumerate(PITCH_CLASSES)}
# Enharmonic aliases
ENHARMONIC_ALIASES = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}


@dataclass
class ChromaConfig:
    """Configuration parameters for guitar chroma feature extraction."""
    sample_rate: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024
    fmin: float = 80.0       # Just below low E string (~82.41 Hz)
    fmax: float = 2200.0     # Covers guitar fundamental range + low harmonics
    tuning_a4: float = 440.0 # Concert pitch reference
    log_compression_gamma: float = 25.0
    harmonic_suppression: float = 0.18
    temporal_smoothing_frames: int = 5
    apply_hps: bool = True   # Separate pick clicks from sustained tones
    mode: str = "auto"       # "auto", "clean", or "distorted"


class ChromaExtractor:
    """
    Extracts 12-element Pitch Class Profile (Chroma) vectors from audio signals.
    Optimized for clean and distorted electric and acoustic guitars.
    """

    def __init__(self, config: Optional[ChromaConfig] = None):
        self.config = config or ChromaConfig()
        self.pitch_classes = PITCH_CLASSES

    def hz_to_semitone(self, freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert frequency in Hz to fractional MIDI semitone value relative to A4 (MIDI 69)."""
        freq_safe = np.maximum(freq, 1e-6)
        return 69.0 + 12.0 * np.log2(freq_safe / self.config.tuning_a4)

    def semitone_to_hz(self, semitone: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert MIDI semitone value to frequency in Hz."""
        return self.config.tuning_a4 * (2.0 ** ((semitone - 69.0) / 12.0))

    def extract_from_audio(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chroma matrix from audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Input audio waveform (mono or stereo).
        sample_rate : int, optional
            Sample rate of audio. If None, uses config.sample_rate.

        Returns
        -------
        chroma : np.ndarray
            Normalized chroma vectors of shape (12, n_frames).
        times : np.ndarray
            Timestamp in seconds for each frame.
        """
        sr = sample_rate or self.config.sample_rate
        freqs, times, mag_spec = compute_stft(
            audio,
            sample_rate=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        if self.config.apply_hps:
            # Suppress percussive pick transient attacks to clean up tonal spectra
            harmonic_spec, _ = harmonic_percussive_separation(mag_spec, kernel_size=25)
            spec_to_use = harmonic_spec
        else:
            spec_to_use = mag_spec

        chroma = self.extract_from_spectrogram(spec_to_use, freqs)
        return chroma, times

    def extract_from_spectrogram(
        self, magnitude_spec: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray:
        """
        Map STFT magnitude spectrogram to 12-semitone chroma profile.

        Parameters
        ----------
        magnitude_spec : np.ndarray
            Magnitude spectrum of shape (n_freqs, n_frames).
        freqs : np.ndarray
            Array of linear frequency bin values in Hz.

        Returns
        -------
        chroma : np.ndarray
            Normalized chroma representation of shape (12, n_frames).
        """
        n_freqs, n_frames = magnitude_spec.shape
        raw_chroma = np.zeros((12, n_frames), dtype=np.float32)

        # Apply logarithmic dynamic range compression
        # This equalizes heavily distorted chords with quiet acoustic strums
        gamma = self.config.log_compression_gamma
        if self.config.mode == "distorted":
            gamma *= 1.5
        elif self.config.mode == "clean":
            gamma *= 0.8

        comp_spec = np.log1p(gamma * magnitude_spec)

        # Filter valid frequency range for guitar fundamentals and direct lower harmonics
        valid_indices = np.where(
            (freqs >= self.config.fmin) & (freqs <= self.config.fmax)
        )[0]

        if len(valid_indices) == 0:
            return raw_chroma

        valid_freqs = freqs[valid_indices]
        semitones = self.hz_to_semitone(valid_freqs)
        
        # Octave-based weighting: prioritize guitar fundamental register (80 - 800 Hz)
        # Guitars produce fundamentals mainly between MIDI 40 (E2) and MIDI 84 (C7)
        # Gaussian envelope centered around MIDI 57 (A3 ~220 Hz)
        octave_weights = np.exp(-0.5 * ((semitones - 57.0) / 20.0) ** 2)

        for i, idx in enumerate(valid_indices):
            s = semitones[i]
            nearest_pitch = int(np.round(s))
            pitch_class = nearest_pitch % 12
            
            # Sub-semitone proximity weight (Gaussian within semitone bin)
            dist = abs(s - nearest_pitch)
            proximity = np.exp(-0.5 * (dist / 0.3) ** 2)
            
            w = octave_weights[i] * proximity
            raw_chroma[pitch_class, :] += w * comp_spec[idx, :]

        # Harmonic overtone attenuation:
        # In guitar chords, 3rd harmonics (perfect 5th + 1 octave) bleed into the 5th degree.
        # Attenuate the bleeding 3rd harmonic to avoid false dominant/fifth confusion.
        if self.config.harmonic_suppression > 0:
            suppressed = raw_chroma.copy()
            for pc in range(12):
                fifth = (pc + 7) % 12
                # Subtract small fraction of the fundamental from the 5th
                bleed = self.config.harmonic_suppression * raw_chroma[pc, :]
                suppressed[fifth, :] = np.maximum(0.0, suppressed[fifth, :] - bleed)
            raw_chroma = suppressed

        # Temporal smoothing along time frames (median filter / moving average)
        k_smooth = self.config.temporal_smoothing_frames
        if k_smooth > 1 and n_frames >= k_smooth:
            if k_smooth % 2 == 0:
                k_smooth += 1
            raw_chroma = scipy.signal.medfilt2d(raw_chroma, kernel_size=[1, k_smooth])

        # L2-normalization for each frame
        norms = np.linalg.norm(raw_chroma, axis=0, keepdims=True)
        eps = 1e-8
        normalized_chroma = raw_chroma / (norms + eps)

        return normalized_chroma

    def get_pitch_class_name(self, index: int) -> str:
        """Return the pitch class name for index 0..11."""
        return self.pitch_classes[index % 12]

    def get_pitch_class_index(self, name: str) -> int:
        """Return index 0..11 for a pitch class name (e.g. 'C#', 'Bb')."""
        standard_name = ENHARMONIC_ALIASES.get(name, name)
        if standard_name in NOTE_TO_PITCH_CLASS:
            return NOTE_TO_PITCH_CLASS[standard_name]
        raise ValueError(f"Unknown pitch class name: {name}")
