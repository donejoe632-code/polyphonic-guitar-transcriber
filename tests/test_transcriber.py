"""
Unit and integration tests for Polyphonic Guitar Transcriber.

Tests cover:
- Frequency-to-pitch math and invertibility
- STFT and Harmonic-Percussive Separation (HPS)
- Spectral peak parabolic interpolation
- 12-semitone chroma extraction and L2 normalization
- Chord template matching across Major, Minor, 7th, and Power chords
- Distorted guitar overdrive resilience
- End-to-end timeline segmentation and event formatting
"""

import pytest
import numpy as np

from src.dsp import (
    ensure_mono,
    compute_stft,
    harmonic_percussive_separation,
    spectral_peaks,
    create_cqt_filterbank,
    energy_envelope,
    spectral_flux_onsets,
)
from src.chroma import (
    ChromaExtractor,
    ChromaConfig,
    PITCH_CLASSES,
    NOTE_TO_PITCH_CLASS,
)
from src.chord_detector import ChordDetector, ChordCandidate
from src.transcriber import (
    GuitarTranscriber,
    TranscriberConfig,
    TimelineEvent,
    TranscriptionResult,
)


# ============================================================================
# DSP & Math Tests
# ============================================================================

def test_hz_to_semitone_conversion():
    extractor = ChromaExtractor(ChromaConfig(tuning_a4=440.0))

    # A4 = 440 Hz -> MIDI 69
    assert np.isclose(extractor.hz_to_semitone(440.0), 69.0, atol=1e-4)

    # A3 = 220 Hz -> MIDI 57
    assert np.isclose(extractor.hz_to_semitone(220.0), 57.0, atol=1e-4)

    # A5 = 880 Hz -> MIDI 81
    assert np.isclose(extractor.hz_to_semitone(880.0), 81.0, atol=1e-4)

    # C4 = ~261.6256 Hz -> MIDI 60
    assert np.isclose(extractor.hz_to_semitone(261.6256), 60.0, atol=1e-2)

    # E2 (low guitar string) = ~82.4069 Hz -> MIDI 40
    assert np.isclose(extractor.hz_to_semitone(82.4069), 40.0, atol=1e-2)


def test_semitone_hz_invertibility():
    extractor = ChromaExtractor(ChromaConfig(tuning_a4=440.0))
    guitar_freqs = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 659.25, 1318.51]

    for freq in guitar_freqs:
        semitone = extractor.hz_to_semitone(freq)
        reconstructed_freq = extractor.semitone_to_hz(semitone)
        assert np.isclose(freq, reconstructed_freq, rtol=1e-5)


def test_ensure_mono():
    # 1D array remains 1D
    mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert np.allclose(ensure_mono(mono), mono)

    # 2D stereo (N, 2)
    stereo = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    mono_res = ensure_mono(stereo)
    assert mono_res.shape == (2,)
    assert np.allclose(mono_res, [0.3, 0.7])


def test_stft_computation():
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz test tone
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    n_fft = 2048
    hop_length = 512
    freqs, times, mag = compute_stft(
        signal, sample_rate=sr, n_fft=n_fft, hop_length=hop_length
    )

    assert mag.ndim == 2
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], sr / 2.0, atol=1e-1)
    assert np.all(mag >= 0.0)

    # Frequency bin closest to 440 Hz should contain dominant magnitude
    closest_bin = np.argmin(np.abs(freqs - 440.0))
    center_frame = mag.shape[1] // 2
    assert np.argmax(mag[:, center_frame]) == closest_bin


def test_harmonic_percussive_separation():
    # Synthetic spectrogram with a sustained tone (horizontal line)
    # and a sharp percussive click (vertical line)
    spec = np.zeros((64, 64), dtype=np.float32)
    spec[20, :] = 5.0  # Horizontal harmonic line
    spec[:, 30] += 5.0  # Vertical percussive spike

    h_spec, p_spec = harmonic_percussive_separation(spec, kernel_size=15)

    # Harmonic component should dominate along row 20 away from click
    assert h_spec[20, 10] > p_spec[20, 10]
    # Percussive component should dominate along column 30 away from tone
    assert p_spec[10, 30] > h_spec[10, 30]


def test_spectral_peaks_sub_bin_interpolation():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    target_freq = 440.35  # Non-integer bin frequency
    sig = np.sin(2 * np.pi * target_freq * t)

    freqs, _, mag = compute_stft(sig, sample_rate=sr, n_fft=4096, hop_length=1024)
    frame = mag[:, mag.shape[1] // 2]

    peaks = spectral_peaks(frame, freqs, min_freq=100.0, max_freq=1000.0)
    assert len(peaks) > 0
    top_peak_freq, top_peak_mag = peaks[0]

    # Parabolic interpolation should detect peak frequency with high sub-bin accuracy
    assert abs(top_peak_freq - target_freq) < 1.0


def test_create_cqt_filterbank():
    freqs = np.linspace(0, 11025, 2049)
    fb, center_freqs = create_cqt_filterbank(freqs, fmin=82.41, fmax=1318.51)

    assert fb.shape[0] == len(center_freqs)
    assert fb.shape[1] == len(freqs)
    assert np.all(fb >= 0.0)
    # Ensure each filter sums to ~1.0
    for row in fb:
        assert np.isclose(np.sum(row), 1.0, atol=1e-3)


# ============================================================================
# Chroma Feature Extraction Tests
# ============================================================================

def test_pitch_class_lookup():
    extractor = ChromaExtractor()
    assert extractor.get_pitch_class_name(0) == "C"
    assert extractor.get_pitch_class_name(9) == "A"
    assert extractor.get_pitch_class_name(4) == "E"

    assert extractor.get_pitch_class_index("C") == 0
    assert extractor.get_pitch_class_index("A") == 9
    assert extractor.get_pitch_class_index("Bb") == 10
    assert extractor.get_pitch_class_index("F#") == 6


def test_chroma_pure_tones():
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    test_notes = [
        (440.0, "A", 9),
        (261.63, "C", 0),
        (329.63, "E", 4),
        (196.00, "G", 7),
    ]

    extractor = ChromaExtractor(ChromaConfig(sample_rate=sr, apply_hps=False))

    for freq, note_name, expected_pc in test_notes:
        sig = 0.7 * np.sin(2 * np.pi * freq * t)
        chroma, _ = extractor.extract_from_audio(sig, sample_rate=sr)

        # Average chroma across middle frames
        mid_chroma = np.mean(chroma[:, 2:-2], axis=1)
        detected_pc = int(np.argmax(mid_chroma))

        assert detected_pc == expected_pc, (
            f"Expected {note_name} (pc={expected_pc}) for {freq}Hz, "
            f"got {PITCH_CLASSES[detected_pc]} (pc={detected_pc})"
        )


def test_chroma_vector_normalization():
    sr = 22050
    sig = np.random.normal(0, 0.1, sr)
    extractor = ChromaExtractor(ChromaConfig(sample_rate=sr))
    chroma, _ = extractor.extract_from_audio(sig, sample_rate=sr)

    # Each non-silent frame vector should have L2-norm == 1.0
    for frame_idx in range(chroma.shape[1]):
        norm = np.linalg.norm(chroma[:, frame_idx])
        assert np.isclose(norm, 1.0, atol=1e-4)


# ============================================================================
# Chord Detection & Template Matching Tests
# ============================================================================

def test_triad_template_matching():
    detector = ChordDetector()

    # C Major: C (0), E (4), G (7)
    c_maj = np.zeros(12, dtype=np.float32)
    c_maj[[0, 4, 7]] = [1.0, 0.9, 0.8]
    cand = detector.detect_frame(c_maj)
    assert cand.name == "C"
    assert cand.quality == "major"
    assert cand.confidence > 0.75

    # A Minor: A (9), C (0), E (4)
    a_min = np.zeros(12, dtype=np.float32)
    a_min[[9, 0, 4]] = [1.0, 0.9, 0.8]
    cand = detector.detect_frame(a_min)
    assert cand.name == "Am"
    assert cand.quality == "minor"
    assert cand.confidence > 0.75

    # G Major: G (7), B (11), D (2)
    g_maj = np.zeros(12, dtype=np.float32)
    g_maj[[7, 11, 2]] = [1.0, 0.9, 0.8]
    cand = detector.detect_frame(g_maj)
    assert cand.name == "G"
    assert cand.quality == "major"
    assert cand.confidence > 0.70

    # F Major: F (5), A (9), C (0)
    f_maj = np.zeros(12, dtype=np.float32)
    f_maj[[5, 9, 0]] = [1.0, 0.9, 0.8]
    cand = detector.detect_frame(f_maj)
    assert cand.name == "F"
    assert cand.quality == "major"
    assert cand.confidence > 0.75


def test_power_chord_and_7th_matching():
    detector = ChordDetector()

    # E Power chord (E5): E (4), B (11)
    e5 = np.zeros(12, dtype=np.float32)
    e5[[4, 11]] = [1.0, 0.85]
    cand = detector.detect_frame(e5)
    assert cand.name == "E5"
    assert cand.quality == "5"

    # D Dominant 7th (D7): D (2), F# (6), A (9), C (0)
    d7 = np.zeros(12, dtype=np.float32)
    d7[[2, 6, 9, 0]] = [1.0, 0.85, 0.75, 0.85]
    cand = detector.detect_frame(d7)
    assert cand.name == "D7"
    assert cand.quality == "7"

    # F Major 7th (Fmaj7): F (5), A (9), C (0), E (4)
    f_maj7 = np.zeros(12, dtype=np.float32)
    f_maj7[[5, 9, 0, 4]] = [1.0, 0.85, 0.75, 0.85]
    cand = detector.detect_frame(f_maj7)
    assert cand.name == "Fmaj7"
    assert cand.quality == "maj7"


def test_silence_detection():
    detector = ChordDetector(min_energy_threshold=0.05)
    zero_chroma = np.zeros(12, dtype=np.float32)
    cand = detector.detect_frame(zero_chroma)
    assert cand.name == "N"
    assert cand.quality == "none"
    assert cand.confidence == 1.0


def test_distorted_chord_robustness():
    detector = ChordDetector()
    # A minor with simulated distortion (3rd and 5th harmonic spillover and noise)
    noisy_am = np.zeros(12, dtype=np.float32)
    noisy_am[[9, 0, 4]] = [1.0, 0.85, 0.75]
    # Add distortion harmonic noise across unrelated pitch classes
    noise = np.random.uniform(0.05, 0.15, 12)
    noisy_am += noise
    cand = detector.detect_frame(noisy_am)

    assert cand.name == "Am"
    assert cand.quality == "minor"
    assert cand.confidence > 0.60


# ============================================================================
# End-to-End Transcription Integration Tests
# ============================================================================

def test_end_to_end_transcription_pipeline():
    sr = 22050
    duration_per_chord = 1.0
    t = np.linspace(0, duration_per_chord, int(sr * duration_per_chord), endpoint=False)

    # Synthesize two distinct chords: C major then A minor
    # C major: C3 (130.81), E3 (164.81), G3 (196.00)
    c_audio = (
        0.4 * np.sin(2 * np.pi * 130.81 * t)
        + 0.4 * np.sin(2 * np.pi * 164.81 * t)
        + 0.4 * np.sin(2 * np.pi * 196.00 * t)
    )
    # A minor: A2 (110.00), C3 (130.81), E3 (164.81)
    a_audio = (
        0.4 * np.sin(2 * np.pi * 110.00 * t)
        + 0.4 * np.sin(2 * np.pi * 130.81 * t)
        + 0.4 * np.sin(2 * np.pi * 164.81 * t)
    )

    progression = np.concatenate([c_audio, a_audio])

    config = TranscriberConfig(
        sample_rate=sr,
        min_event_duration=0.3,
        mode="clean",
    )
    transcriber = GuitarTranscriber(config)
    res = transcriber.transcribe(progression, sample_rate=sr)

    assert isinstance(res, TranscriptionResult)
    assert len(res.events) >= 2

    # Check detected chords
    chords = [ev.chord for ev in res.events]
    assert "C" in chords
    assert "Am" in chords

    # Check timeline formatting methods
    timeline_str = res.format_timeline()
    assert "POLYPHONIC GUITAR TRANSCRIPTION TIMELINE" in timeline_str
    assert "C" in timeline_str
    assert "Am" in timeline_str

    # Check JSON export dict
    res_dict = res.to_dict()
    assert res_dict["event_count"] == len(res.events)
    assert "events" in res_dict

    # Check CSV export
    csv_str = res.to_csv()
    assert "start_time,end_time,duration,chord" in csv_str
    assert "C" in csv_str
