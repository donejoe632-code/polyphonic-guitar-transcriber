"""
Chord detection module for guitar transcription.

Provides template-based correlation matching, negative-evidence penalty weighting,
and probabilistic softmax classification with confidence scores for triads, 7ths,
and power chords.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .chroma import PITCH_CLASSES, NOTE_TO_PITCH_CLASS, ENHARMONIC_ALIASES

CHORD_QUALITIES = {
    "major": {
        "intervals": [0, 4, 7],
        "weights": [1.0, 0.9, 0.8],
        "symbol": "",
        "penalties": {3: 0.3},  # Minor 3rd penalty
    },
    "minor": {
        "intervals": [0, 3, 7],
        "weights": [1.0, 0.9, 0.8],
        "symbol": "m",
        "penalties": {4: 0.3},  # Major 3rd penalty
    },
    "5": {
        "intervals": [0, 7],
        "weights": [1.1, 0.9],
        "symbol": "5",
        "penalties": {3: 0.4, 4: 0.4},  # Penalize presence of any 3rd
    },
    "7": {
        "intervals": [0, 4, 7, 10],
        "weights": [1.0, 0.85, 0.75, 0.85],
        "symbol": "7",
        "penalties": {3: 0.3, 11: 0.3},
    },
    "maj7": {
        "intervals": [0, 4, 7, 11],
        "weights": [1.0, 0.85, 0.75, 0.85],
        "symbol": "maj7",
        "penalties": {3: 0.3, 10: 0.3},
    },
    "m7": {
        "intervals": [0, 3, 7, 10],
        "weights": [1.0, 0.85, 0.75, 0.85],
        "symbol": "m7",
        "penalties": {4: 0.3, 11: 0.3},
    },
    "sus4": {
        "intervals": [0, 5, 7],
        "weights": [1.0, 0.9, 0.8],
        "symbol": "sus4",
        "penalties": {4: 0.3},
    },
    "sus2": {
        "intervals": [0, 2, 7],
        "weights": [1.0, 0.9, 0.8],
        "symbol": "sus2",
        "penalties": {3: 0.3, 4: 0.3},
    },
    "dim": {
        "intervals": [0, 3, 6],
        "weights": [1.0, 0.85, 0.8],
        "symbol": "dim",
        "penalties": {7: 0.3},
    },
}


@dataclass
class ChordTemplate:
    """Mathematical template representing a chord profile in 12-dimensional pitch space."""
    name: str
    root: str
    quality: str
    intervals: List[int]
    notes: List[str]
    vector: np.ndarray  # Normalized 12-element template vector


@dataclass
class ChordCandidate:
    """Result of chord recognition for a single frame or time window."""
    name: str
    root: str
    quality: str
    confidence: float
    notes: List[str] = field(default_factory=list)
    raw_similarity: float = 0.0

    def __repr__(self) -> str:
        notes_str = "-".join(self.notes) if self.notes else ""
        return (
            f"<ChordCandidate {self.name} "
            f"conf={self.confidence:.1%} [{notes_str}]>"
        )


class ChordDetector:
    """
    Template matcher and probabilistic classifier for polyphonic guitar chords.
    """

    def __init__(
        self,
        qualities: Optional[List[str]] = None,
        negative_weight: float = -0.22,
        temperature: float = 0.04,
        min_energy_threshold: float = 0.05,
    ):
        """
        Parameters
        ----------
        qualities : Optional[List[str]]
            List of chord qualities to include. Defaults to all standard guitar chord types.
        negative_weight : float, default=-0.22
            Penalty applied to non-chord pitch classes to prevent false positive matches.
        temperature : float, default=0.04
            Softmax temperature scaling for probabilistic confidence estimation.
        min_energy_threshold : float, default=0.05
            Minimum chroma magnitude to trigger chord detection instead of silence/No Chord ('N').
        """
        self.qualities = qualities or list(CHORD_QUALITIES.keys())
        self.negative_weight = negative_weight
        self.temperature = temperature
        self.min_energy_threshold = min_energy_threshold

        self.templates: List[ChordTemplate] = self._build_templates()
        # Matrix shape: (n_templates, 12)
        self.template_matrix = np.array([t.vector for t in self.templates], dtype=np.float32)

    def _build_templates(self) -> List[ChordTemplate]:
        """Construct the bank of normalized chord templates across all roots and qualities."""
        templates: List[ChordTemplate] = []

        for root_idx, root_name in enumerate(PITCH_CLASSES):
            for quality_name in self.qualities:
                q_info = CHORD_QUALITIES[quality_name]
                intervals = q_info["intervals"]
                weights = q_info["weights"]
                symbol = q_info["symbol"]
                penalties = q_info.get("penalties", {})

                chord_name = f"{root_name}{symbol}"
                chord_notes = [
                    PITCH_CLASSES[(root_idx + interval) % 12] for interval in intervals
                ]

                # Initialize template with negative background penalty
                vec = np.full(12, self.negative_weight, dtype=np.float32)

                # Set positive weights for chord tones
                for interval, weight in zip(intervals, weights):
                    vec[(root_idx + interval) % 12] = weight

                # Apply specific negative penalties
                for interval, pen in penalties.items():
                    vec[(root_idx + interval) % 12] -= pen

                # Normalize to unit length
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                templates.append(
                    ChordTemplate(
                        name=chord_name,
                        root=root_name,
                        quality=quality_name,
                        intervals=intervals,
                        notes=chord_notes,
                        vector=vec,
                    )
                )

        return templates

    def detect_frame(self, chroma_vector: np.ndarray) -> ChordCandidate:
        """
        Classify a single 12-element chroma vector.

        Parameters
        ----------
        chroma_vector : np.ndarray
            1D array of length 12 representing the pitch class profile.

        Returns
        -------
        ChordCandidate
            The most probable chord candidate with confidence and notes.
        """
        norm = np.linalg.norm(chroma_vector)
        if norm < self.min_energy_threshold:
            return ChordCandidate(
                name="N",
                root="N",
                quality="none",
                confidence=1.0,
                notes=[],
                raw_similarity=0.0,
            )

        # Normalize vector
        c_norm = chroma_vector / (norm + 1e-9)

        # Vectorized dot product against all templates: (n_templates,)
        similarities = np.dot(self.template_matrix, c_norm)

        # Softmax probability distribution
        scaled = similarities / self.temperature
        # Subtract max for numerical stability
        exp_scaled = np.exp(scaled - np.max(scaled))
        probabilities = exp_scaled / np.sum(exp_scaled)

        best_idx = int(np.argmax(probabilities))
        best_template = self.templates[best_idx]
        raw_sim = float(similarities[best_idx])
        prob = float(probabilities[best_idx])
        # Calibrated confidence blending similarity and categorical probability
        confidence = float(np.clip(0.65 * max(0.0, raw_sim) + 0.35 * prob, 0.0, 1.0))

        return ChordCandidate(
            name=best_template.name,
            root=best_template.root,
            quality=best_template.quality,
            confidence=confidence,
            notes=best_template.notes,
            raw_similarity=raw_sim,
        )

    def detect_sequence(
        self, chroma_matrix: np.ndarray, energies: Optional[np.ndarray] = None
    ) -> List[ChordCandidate]:
        """
        Classify a sequence of chroma frames.

        Parameters
        ----------
        chroma_matrix : np.ndarray
            Matrix of shape (12, n_frames).
        energies : Optional[np.ndarray]
            Array of shape (n_frames,) with RMS energy or chroma norm.

        Returns
        -------
        List[ChordCandidate]
            Detected chord candidate for each frame.
        """
        n_frames = chroma_matrix.shape[1]
        results: List[ChordCandidate] = []

        # Vectorized matrix multiplication: (n_templates, n_frames)
        # First ensure each column is unit normalized
        norms = np.linalg.norm(chroma_matrix, axis=0, keepdims=True) + 1e-9
        normed_chroma = chroma_matrix / norms

        all_sims = np.dot(self.template_matrix, normed_chroma)  # (n_templates, n_frames)

        # Apply softmax per column
        scaled = all_sims / self.temperature
        max_scaled = np.max(scaled, axis=0, keepdims=True)
        exp_scaled = np.exp(scaled - max_scaled)
        probs = exp_scaled / np.sum(exp_scaled, axis=0, keepdims=True)

        best_indices = np.argmax(probs, axis=0)

        for f in range(n_frames):
            frame_norm = norms[0, f]
            is_silent = False
            if energies is not None and f < len(energies):
                if energies[f] < self.min_energy_threshold:
                    is_silent = True
            elif frame_norm < self.min_energy_threshold:
                is_silent = True

            if is_silent:
                results.append(
                    ChordCandidate(
                        name="N",
                        root="N",
                        quality="none",
                        confidence=1.0,
                        notes=[],
                        raw_similarity=0.0,
                    )
                )
            else:
                idx = best_indices[f]
                tmpl = self.templates[idx]
                raw_sim = float(all_sims[idx, f])
                prob = float(probs[idx, f])
                confidence = float(np.clip(0.65 * max(0.0, raw_sim) + 0.35 * prob, 0.0, 1.0))
                results.append(
                    ChordCandidate(
                        name=tmpl.name,
                        root=tmpl.root,
                        quality=tmpl.quality,
                        confidence=confidence,
                        notes=tmpl.notes,
                        raw_similarity=raw_sim,
                    )
                )

        return results

    def get_top_k(
        self, chroma_vector: np.ndarray, k: int = 3
    ) -> List[Tuple[ChordCandidate, float]]:
        """Return the top-k chord candidates ranked by probability."""
        norm = np.linalg.norm(chroma_vector)
        if norm < self.min_energy_threshold:
            return [(
                ChordCandidate(
                    name="N",
                    root="N",
                    quality="none",
                    confidence=1.0,
                    notes=[],
                    raw_similarity=0.0,
                ),
                1.0,
            )]

        c_norm = chroma_vector / (norm + 1e-9)
        similarities = np.dot(self.template_matrix, c_norm)

        scaled = similarities / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        probabilities = exp_scaled / np.sum(exp_scaled)

        top_indices = np.argsort(probabilities)[::-1][:k]

        output = []
        for idx in top_indices:
            tmpl = self.templates[idx]
            cand = ChordCandidate(
                name=tmpl.name,
                root=tmpl.root,
                quality=tmpl.quality,
                confidence=float(probabilities[idx]),
                notes=tmpl.notes,
                raw_similarity=float(similarities[idx]),
            )
            output.append((cand, float(probabilities[idx])))

        return output
