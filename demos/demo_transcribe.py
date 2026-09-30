"""
Demonstration script for Polyphonic Guitar Transcriber.

Synthesizes a realistic 4-chord guitar progression (Am -> C -> G -> F)
using physical acoustic string modeling (harmonic decay + pick attack + strum timing),
runs the end-to-end transcription engine, prints the formatted timeline,
and exports the synthesized audio to 'demo_riff.wav'.
"""

import os
import sys
import numpy as np
import soundfile as sf

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transcriber import GuitarTranscriber, TranscriberConfig


def synthesize_guitar_string(
    freq: float,
    duration: float,
    sample_rate: int = 22050,
    amplitude: float = 0.7,
    decay_sec: float = 1.6,
) -> np.ndarray:
    """
    Synthesize an acoustic guitar string vibration using harmonic modeling
    with frequency-dependent damping and pick impulse.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Harmonics up to Nyquist
    max_harmonic = min(18, int((sample_rate * 0.45) / freq))
    for h in range(1, max_harmonic + 1):
        h_freq = freq * h
        # Harmonic amplitude roll-off ~ 1 / h^1.2
        h_amp = amplitude / (h**1.15)
        # Higher harmonics decay faster than lower fundamentals
        h_decay = decay_sec / (h**0.6)
        envelope = np.exp(-t / h_decay)

        phase = np.random.uniform(0, 2 * np.pi)
        signal += h_amp * np.sin(2 * np.pi * h_freq * t + phase) * envelope

    # Add a short pick attack transient (first 15ms)
    attack_samples = int(0.015 * sample_rate)
    noise = np.random.normal(0, 0.15 * amplitude, attack_samples)
    noise *= np.linspace(1.0, 0.0, attack_samples)
    signal[:attack_samples] += noise

    # Pluck onset fade-in (1 ms) to prevent click
    fade_in = min(int(0.002 * sample_rate), len(signal))
    signal[:fade_in] *= np.linspace(0.0, 1.0, fade_in)

    return signal


def synthesize_strummed_chord(
    frequencies: list,
    duration: float,
    sample_rate: int = 22050,
    strum_delay_sec: float = 0.022,
) -> np.ndarray:
    """
    Synthesize a strummed guitar chord with realistic string-to-string strum delay.
    """
    total_samples = int(sample_rate * duration)
    chord_signal = np.zeros(total_samples, dtype=np.float32)

    for i, freq in enumerate(frequencies):
        delay_samples = int(i * strum_delay_sec * sample_rate)
        rem_samples = total_samples - delay_samples
        if rem_samples <= 0:
            break
        string_dur = rem_samples / sample_rate
        string_audio = synthesize_guitar_string(
            freq, string_dur, sample_rate=sample_rate, amplitude=0.45
        )
        chord_signal[delay_samples : delay_samples + len(string_audio)] += string_audio

    return chord_signal


def generate_demo_progression(sample_rate: int = 22050) -> tuple:
    """
    Generate an authentic 4-chord guitar progression:
    Am -> C -> G -> F (1.5 seconds each, total 6.0 seconds)
    """
    # Standard guitar chord string frequencies in Hz:
    # Am: A2 (110), E3 (164.81), A3 (220), C4 (261.63), E4 (329.63)
    am_freqs = [110.00, 164.81, 220.00, 261.63, 329.63]

    # C: C3 (130.81), E3 (164.81), G3 (196.00), C4 (261.63), E4 (329.63)
    c_freqs = [130.81, 164.81, 196.00, 261.63, 329.63]

    # G: G2 (98.00), B2 (123.47), D3 (146.83), G3 (196.00), B3 (246.94), G4 (392.00)
    g_freqs = [98.00, 123.47, 146.83, 196.00, 246.94, 392.00]

    # F: F2 (87.31), C3 (130.81), F3 (174.61), A3 (220.00), C4 (261.63), F4 (349.23)
    f_freqs = [87.31, 130.81, 174.61, 220.00, 261.63, 349.23]

    chord_duration = 1.5  # seconds per chord
    chords_meta = [
        ("Am", am_freqs),
        ("C", c_freqs),
        ("G", g_freqs),
        ("F", f_freqs),
    ]

    progression_audio = []
    ground_truth = []

    current_time = 0.0
    for name, freqs in chords_meta:
        audio = synthesize_strummed_chord(
            freqs, chord_duration, sample_rate=sample_rate
        )
        progression_audio.append(audio)
        ground_truth.append((current_time, current_time + chord_duration, name))
        current_time += chord_duration

    full_audio = np.concatenate(progression_audio)
    # Normalize peak audio to -1 dB (0.89)
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = (full_audio / peak) * 0.89

    return full_audio, sample_rate, ground_truth


def main():
    print("=" * 80)
    print("  POLYPHONIC GUITAR TRANSCRIBER DEMO")
    print("  Synthesizing acoustic guitar progression: Am -> C -> G -> F")
    print("=" * 80)

    sample_rate = 22050
    audio, sr, ground_truth = generate_demo_progression(sample_rate)
    duration = len(audio) / sr

    output_wav = os.path.join(os.path.dirname(__file__), "demo_riff.wav")
    sf.write(output_wav, audio, sr)
    print(f"\n[+] Audio synthesized: {duration:.2f}s, saved to '{output_wav}'")

    print("\n[+] Ground Truth Progression:")
    for start, end, chord in ground_truth:
        print(f"    [{start:04.2f}s - {end:04.2f}s] : {chord}")

    print("\n[+] Initializing GuitarTranscriber...")
    config = TranscriberConfig(
        sample_rate=sr,
        n_fft=4096,
        hop_length=1024,
        min_event_duration=0.3,
        mode="clean",
    )
    transcriber = GuitarTranscriber(config)

    print("[+] Running polyphonic transcription...")
    result = transcriber.transcribe(audio, sample_rate=sr)

    print("\n" + result.format_timeline())

    # Verification summary
    print("\n[+] Verification Check:")
    detected_chords = [ev.chord for ev in result.events]
    expected_chords = [chord for _, _, chord in ground_truth]

    print(f"    Expected: {expected_chords}")
    print(f"    Detected: {detected_chords}")

    matches = sum(
        1
        for exp, det in zip(expected_chords, detected_chords)
        if exp == det
    )
    accuracy = (matches / len(expected_chords)) * 100.0 if expected_chords else 0.0
    print(f"    Progression Accuracy: {accuracy:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
