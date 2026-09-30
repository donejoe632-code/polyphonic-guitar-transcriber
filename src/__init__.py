"""
Polyphonic Guitar Transcriber
=============================
A high-performance, lightweight Python library and CLI tool for polyphonic guitar
transcription, pitch detection, chroma feature extraction, and chord recognition
for clean and distorted guitar signals.
"""

from .dsp import (
    compute_stft,
    harmonic_percussive_separation,
    spectral_peaks,
    energy_envelope,
    spectral_flux_onsets,
)
from .chroma import ChromaExtractor, ChromaConfig
from .chord_detector import ChordDetector, ChordCandidate, ChordTemplate
from .transcriber import GuitarTranscriber, TranscriberConfig, TimelineEvent, TranscriptionResult

__version__ = "1.0.0"
__all__ = [
    "compute_stft",
    "harmonic_percussive_separation",
    "spectral_peaks",
    "energy_envelope",
    "spectral_flux_onsets",
    "ChromaExtractor",
    "ChromaConfig",
    "ChordDetector",
    "ChordCandidate",
    "ChordTemplate",
    "GuitarTranscriber",
    "TranscriberConfig",
    "TimelineEvent",
    "TranscriptionResult",
]
