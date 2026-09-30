"""
End-to-End Polyphonic Guitar Transcriber pipeline.

Processes audio signals or audio files, extracts tonal features, performs temporal
segmentation, and generates a structured timeline of chord progression events.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
import os
import sys
import argparse
import numpy as np
import soundfile as sf
import scipy.signal

from .dsp import ensure_mono, energy_envelope
from .chroma import ChromaExtractor, ChromaConfig
from .chord_detector import ChordDetector, ChordCandidate


@dataclass
class TranscriberConfig:
    """Configuration options for the GuitarTranscriber pipeline."""
    sample_rate: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024
    min_event_duration: float = 0.25   # Minimum chord duration in seconds
    min_confidence: float = 0.25       # Minimum confidence to accept a chord
    energy_silence_threshold: float = 0.008  # RMS energy threshold for silence
    smoothing_window_frames: int = 7   # Temporal mode filter window
    mode: str = "auto"                 # "auto", "clean", or "distorted"
    qualities: Optional[List[str]] = None


@dataclass
class TimelineEvent:
    """Represents a sustained chord event across a time span."""
    start_time: float
    end_time: float
    chord: str
    root: str
    quality: str
    confidence: float
    notes: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def format_timestamp(self) -> str:
        s_m, s_s = divmod(self.start_time, 60.0)
        e_m, e_s = divmod(self.end_time, 60.0)
        return f"[{int(s_m):02d}:{s_s:05.2f} -> {int(e_m):02d}:{e_s:05.2f}]"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


@dataclass
class TranscriptionResult:
    """Complete results of a transcription analysis."""
    events: List[TimelineEvent]
    duration_seconds: float
    sample_rate: int
    chroma_matrix: np.ndarray
    frame_times: np.ndarray
    frame_candidates: List[ChordCandidate]

    def format_timeline(self) -> str:
        """Format detected events into a human-readable ASCII timeline table."""
        header = (
            "=" * 82 + "\n"
            f"  POLYPHONIC GUITAR TRANSCRIPTION TIMELINE\n"
            f"  Duration: {self.duration_seconds:.2f}s | Events Detected: {len(self.events)}\n"
            + "-" * 82 + "\n"
            f"  {'Time Interval':<24} {'Chord':<8} {'Quality':<10} {'Conf':<8} {'Notes':<20}\n"
            + "-" * 82
        )
        lines = [header]

        for ev in self.events:
            notes_str = " - ".join(ev.notes) if ev.notes else "(silent)"
            time_str = ev.format_timestamp()
            line = (
                f"  {time_str:<24} {ev.chord:<8} {ev.quality:<10} "
                f"{ev.confidence:>5.1%}   {notes_str:<20}"
            )
            lines.append(line)

        lines.append("=" * 82)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "event_count": len(self.events),
            "events": [ev.to_dict() for ev in self.events],
        }

    def to_csv(self) -> str:
        """Export timeline events as standard CSV."""
        rows = ["start_time,end_time,duration,chord,root,quality,confidence,notes"]
        for ev in self.events:
            notes_str = ";".join(ev.notes)
            rows.append(
                f"{ev.start_time:.3f},{ev.end_time:.3f},{ev.duration:.3f},"
                f"{ev.chord},{ev.root},{ev.quality},{ev.confidence:.3f},{notes_str}"
            )
        return "\n".join(rows)


class GuitarTranscriber:
    """
    High-level transcription engine for polyphonic guitar performances.
    """

    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or TranscriberConfig()
        
        chroma_cfg = ChromaConfig(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            mode=self.config.mode,
        )
        self.chroma_extractor = ChromaExtractor(chroma_cfg)
        self.chord_detector = ChordDetector(
            qualities=self.config.qualities,
            min_energy_threshold=self.config.energy_silence_threshold,
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio array into a chord progression timeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio waveform (mono or stereo).
        sample_rate : int, optional
            Sampling rate of the audio. If None, uses config.sample_rate.

        Returns
        -------
        TranscriptionResult
            Transcription result with timeline events and feature matrices.
        """
        sr = sample_rate or self.config.sample_rate
        mono_audio = ensure_mono(audio)
        total_duration = len(mono_audio) / float(sr)

        # 1. Compute frame-wise RMS energy
        rms_env = energy_envelope(
            mono_audio,
            hop_length=self.config.hop_length,
            frame_length=self.config.n_fft,
        )

        # 2. Extract 12-semitone chroma profile
        chroma_matrix, frame_times = self.chroma_extractor.extract_from_audio(
            mono_audio, sample_rate=sr
        )
        n_frames = chroma_matrix.shape[1]

        # Align rms envelope length with chroma frame count
        if len(rms_env) < n_frames:
            rms_env = np.pad(rms_env, (0, n_frames - len(rms_env)), mode="edge")
        else:
            rms_env = rms_env[:n_frames]

        # 3. Classify individual frames
        frame_candidates = self.chord_detector.detect_sequence(
            chroma_matrix, energies=rms_env
        )

        # 4. Temporal continuity smoothing (rolling mode filter)
        smoothed_names = self._smooth_chord_sequence(
            [c.name for c in frame_candidates],
            window_size=self.config.smoothing_window_frames,
        )

        # Re-associate candidates with smoothed names
        candidate_lookup = {c.name: c for c in frame_candidates}
        # Include silence candidate
        candidate_lookup["N"] = ChordCandidate(
            name="N", root="N", quality="none", confidence=1.0, notes=[]
        )

        smoothed_candidates: List[ChordCandidate] = []
        for i, name in enumerate(smoothed_names):
            orig = frame_candidates[i]
            if orig.name == name:
                smoothed_candidates.append(orig)
            else:
                tmpl_info = candidate_lookup.get(name)
                if tmpl_info:
                    smoothed_candidates.append(
                        ChordCandidate(
                            name=name,
                            root=tmpl_info.root,
                            quality=tmpl_info.quality,
                            confidence=orig.confidence * 0.85,
                            notes=tmpl_info.notes,
                            raw_similarity=orig.raw_similarity,
                        )
                    )
                else:
                    smoothed_candidates.append(orig)

        # 5. Segment into contiguous timeline events
        events = self._segment_events(
            smoothed_candidates, frame_times, total_duration
        )

        # 6. Filter out sub-duration jitter and low-confidence noise
        cleaned_events = self._filter_events(events)

        return TranscriptionResult(
            events=cleaned_events,
            duration_seconds=total_duration,
            sample_rate=sr,
            chroma_matrix=chroma_matrix,
            frame_times=frame_times,
            frame_candidates=smoothed_candidates,
        )

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Load an audio file from disk and run transcription.

        Parameters
        ----------
        file_path : str
            Path to .wav, .flac, or .ogg audio file.

        Returns
        -------
        TranscriptionResult
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio, sr = sf.read(file_path, dtype="float32")
        return self.transcribe(audio, sample_rate=sr)

    def _smooth_chord_sequence(
        self, chord_names: List[str], window_size: int = 7
    ) -> List[str]:
        """Apply a majority voting (mode filter) across a sliding window."""
        if len(chord_names) <= window_size:
            return chord_names

        half_w = window_size // 2
        smoothed: List[str] = []
        for i in range(len(chord_names)):
            start = max(0, i - half_w)
            end = min(len(chord_names), i + half_w + 1)
            window = chord_names[start:end]

            # Find majority chord in window
            counts: Dict[str, int] = {}
            for item in window:
                counts[item] = counts.get(item, 0) + 1

            majority = max(counts.items(), key=lambda kv: kv[1])[0]
            smoothed.append(majority)

        return smoothed

    def _segment_events(
        self,
        candidates: List[ChordCandidate],
        frame_times: np.ndarray,
        total_duration: float,
    ) -> List[TimelineEvent]:
        """Group contiguous frames with the same chord prediction into TimelineEvents."""
        if not candidates:
            return []

        events: List[TimelineEvent] = []
        n_frames = len(candidates)
        frame_dt = (
            (frame_times[1] - frame_times[0])
            if len(frame_times) > 1
            else (self.config.hop_length / float(self.config.sample_rate))
        )

        curr_chord = candidates[0].name
        curr_cand = candidates[0]
        start_idx = 0
        conf_sum = candidates[0].confidence
        conf_count = 1

        for i in range(1, n_frames):
            cand = candidates[i]
            if cand.name == curr_chord:
                conf_sum += cand.confidence
                conf_count += 1
            else:
                start_t = float(frame_times[start_idx])
                if start_idx == 0 and start_t < 0.15:
                    start_t = 0.0

                end_t = float(frame_times[i])
                avg_conf = conf_sum / conf_count

                if curr_chord != "N":
                    events.append(
                        TimelineEvent(
                            start_time=round(start_t, 3),
                            end_time=round(end_t, 3),
                            chord=curr_cand.name,
                            root=curr_cand.root,
                            quality=curr_cand.quality,
                            confidence=round(float(avg_conf), 3),
                            notes=list(curr_cand.notes),
                        )
                    )

                # Reset for next chord
                curr_chord = cand.name
                curr_cand = cand
                start_idx = i
                conf_sum = cand.confidence
                conf_count = 1

        # Last segment
        start_t = float(frame_times[start_idx])
        if start_idx == 0 and start_t < 0.15:
            start_t = 0.0

        end_t = float(min(total_duration, float(frame_times[-1]) + frame_dt))
        if total_duration - end_t < 0.15:
            end_t = float(total_duration)

        avg_conf = conf_sum / conf_count
        if curr_chord != "N":
            events.append(
                TimelineEvent(
                    start_time=round(start_t, 3),
                    end_time=round(end_t, 3),
                    chord=curr_cand.name,
                    root=curr_cand.root,
                    quality=curr_cand.quality,
                    confidence=round(float(avg_conf), 3),
                    notes=list(curr_cand.notes),
                )
            )

        return events

    def _filter_events(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Prune spurious transient chord blips below min_event_duration or min_confidence."""
        if not events:
            return []

        filtered: List[TimelineEvent] = []
        for ev in events:
            if (
                ev.duration >= self.config.min_event_duration
                and ev.confidence >= self.config.min_confidence
            ):
                filtered.append(ev)

        # Merge adjacent identical chords that might have been separated by pruned blips
        if not filtered:
            return []

        merged: List[TimelineEvent] = [filtered[0]]
        for ev in filtered[1:]:
            prev = merged[-1]
            if ev.chord == prev.chord and abs(ev.start_time - prev.end_time) < 0.2:
                # Merge into prev
                total_dur = (prev.duration + ev.duration) or 1.0
                weighted_conf = (
                    prev.confidence * prev.duration + ev.confidence * ev.duration
                ) / total_dur
                merged[-1] = TimelineEvent(
                    start_time=prev.start_time,
                    end_time=ev.end_time,
                    chord=prev.chord,
                    root=prev.root,
                    quality=prev.quality,
                    confidence=round(weighted_conf, 3),
                    notes=prev.notes,
                )
            else:
                merged.append(ev)

        return merged


def main():
    """Command Line Interface entrypoint for guitar transcription."""
    parser = argparse.ArgumentParser(
        description="Polyphonic Guitar Transcriber: Audio to Chord Timeline CLI"
    )
    parser.add_argument("audio_file", help="Path to input audio file (.wav, .flac, .ogg)")
    parser.add_argument(
        "--mode",
        choices=["auto", "clean", "distorted"],
        default="auto",
        help="Guitar signal mode (distorted boosts harmonic compression)",
    )
    parser.add_argument(
        "--output", "-o", help="Optional path to export timeline as JSON or CSV"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="Minimum chord duration in seconds (default: 0.25)",
    )

    args = parser.parse_args()

    cfg = TranscriberConfig(mode=args.mode, min_event_duration=args.min_duration)
    transcriber = GuitarTranscriber(cfg)

    print(f"Loading and transcribing: {args.audio_file} (mode: {args.mode})...")
    result = transcriber.transcribe_file(args.audio_file)
    print(result.format_timeline())

    if args.output:
        out_ext = os.path.splitext(args.output)[1].lower()
        if out_ext == ".json":
            import json
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Saved JSON transcription to: {args.output}")
        elif out_ext == ".csv":
            with open(args.output, "w") as f:
                f.write(result.to_csv())
            print(f"Saved CSV transcription to: {args.output}")
        else:
            with open(args.output, "w") as f:
                f.write(result.format_timeline())
            print(f"Saved text transcription to: {args.output}")


if __name__ == "__main__":
    main()
