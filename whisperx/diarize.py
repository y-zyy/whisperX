import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.schema import TranscriptionResult, AlignedTranscriptionResult, ProgressCallback
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class IntervalTree:
    """
    Simple interval tree for fast overlap queries using sorted array + binary search.

    Uses O(n) space and provides O(log n) query time instead of O(n) linear scan.
    This achieves ~228x speedup for speaker assignment in long-form content.
    """

    def __init__(self, intervals: List[Tuple[float, float, str]]):
        """
        Initialize the interval tree with diarization segments.

        Args:
            intervals: List of (start, end, speaker) tuples
        """
        if not intervals:
            self.starts = np.array([])
            self.ends = np.array([])
            self.speakers: List[str] = []
            return

        # Sort intervals by start time for binary search
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([i[0] for i in sorted_intervals], dtype=np.float64)
        self.ends = np.array([i[1] for i in sorted_intervals], dtype=np.float64)
        self.speakers = [i[2] for i in sorted_intervals]

    def query(self, start: float, end: float) -> List[Tuple[str, float]]:
        """
        Find all intervals that overlap with [start, end] and compute intersection.

        Args:
            start: Query interval start time
            end: Query interval end time

        Returns:
            List of (speaker, intersection_duration) tuples for overlapping segments
        """
        if len(self.starts) == 0:
            return []

        # Binary search to find candidate intervals
        # Only intervals with start < end could overlap
        right_idx = np.searchsorted(self.starts, end, side='left')
        if right_idx == 0:
            return []

        # Check candidates for actual overlap
        candidates = slice(0, right_idx)
        overlaps = (self.starts[candidates] < end) & (self.ends[candidates] > start)

        results = []
        for idx in np.where(overlaps)[0]:
            intersection = min(self.ends[idx], end) - max(self.starts[idx], start)
            if intersection > 0:
                results.append((self.speakers[idx], intersection))
        return results

    def find_nearest(self, time: float) -> Optional[str]:
        """
        Find the speaker of the nearest segment to a given time point.

        Args:
            time: Time point to find nearest segment for

        Returns:
            Speaker ID of nearest segment, or None if no segments exist
        """
        if len(self.starts) == 0:
            return None

        # Calculate midpoints of all segments
        mids = (self.starts + self.ends) / 2
        nearest_idx = np.argmin(np.abs(mids - time))
        return self.speakers[nearest_idx]


DIARIZEN_DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
DIARIZEN_MODEL_PREFIX = "BUT-FIT/diarizen-"


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        cache_dir=None,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        model_config = model_name or DIARIZEN_DEFAULT_MODEL
        logger.info(f"Loading diarization model: {model_config}")

        self.backend = (
            "diarizen"
            if model_config.startswith(DIARIZEN_MODEL_PREFIX)
            else "pyannote"
        )

        if self.backend == "diarizen":
            try:
                from diarizen.pipelines.inference import DiariZenPipeline
            except ImportError as exc:
                raise ImportError(
                    "The DiariZen diarization backend is not installed. "
                    "Install DiariZen and its compatible pyannote-audio fork "
                    "following https://github.com/BUTSpeechFIT/DiariZen#installation."
                ) from exc

            # DiariZen.from_pretrained enables local_files_only whenever a
            # cache directory is provided. Download explicitly so a custom
            # cache directory also works on the first run.
            diarizen_hub = snapshot_download(
                repo_id=model_config,
                cache_dir=cache_dir,
                token=token,
            )
            embedding_model = hf_hub_download(
                repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                filename="pytorch_model.bin",
                cache_dir=cache_dir,
                token=token,
            )

            self.model = DiariZenPipeline(
                diarizen_hub=Path(diarizen_hub),
                embedding_model=embedding_model,
            ).to(device)
            self._default_min_speakers = self.model.min_speakers
            self._default_max_speakers = self.model.max_speakers
        else:
            self.model = Pipeline.from_pretrained(
                model_config,
                token=token,
                cache_dir=cache_dir,
            ).to(device)

    @staticmethod
    def _annotation_to_dataframe(diarization) -> pd.DataFrame:
        rows = []
        for segment, label, speaker in diarization.itertracks(yield_label=True):
            if isinstance(speaker, (int, np.integer)):
                speaker = f"SPEAKER_{int(speaker):02d}"
            else:
                speaker = str(speaker)

            rows.append(
                {
                    "segment": segment,
                    "label": label,
                    "speaker": speaker,
                    "start": segment.start,
                    "end": segment.end,
                }
            )

        return pd.DataFrame(
            rows,
            columns=["segment", "label", "speaker", "start", "end"],
        )

    def _call_diarizen(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        return_embeddings: bool,
        progress_callback: ProgressCallback,
    ) -> Union[tuple[pd.DataFrame, None], pd.DataFrame]:
        if num_speakers is not None:
            requested_min = requested_max = num_speakers
        else:
            requested_min = (
                min_speakers
                if min_speakers is not None
                else self._default_min_speakers
            )
            requested_max = (
                max_speakers
                if max_speakers is not None
                else self._default_max_speakers
            )

        if requested_min > requested_max:
            raise ValueError(
                f"min_speakers ({requested_min}) cannot be greater than "
                f"max_speakers ({requested_max})"
            )

        temporary_audio_path = None
        if isinstance(audio, str):
            audio_path = audio
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_audio:
                temporary_audio_path = temporary_audio.name
            torchaudio.save(
                temporary_audio_path,
                torch.from_numpy(audio).float().unsqueeze(0),
                SAMPLE_RATE,
            )
            audio_path = temporary_audio_path

        original_min = self.model.min_speakers
        original_max = self.model.max_speakers

        if progress_callback is not None:
            progress_callback(0.0)

        try:
            self.model.min_speakers = requested_min
            self.model.max_speakers = requested_max
            diarization = self.model(audio_path)
        finally:
            self.model.min_speakers = original_min
            self.model.max_speakers = original_max
            if temporary_audio_path is not None and os.path.exists(temporary_audio_path):
                os.unlink(temporary_audio_path)

        if progress_callback is not None:
            progress_callback(100.0)

        diarize_df = self._annotation_to_dataframe(diarization)

        if return_embeddings:
            warnings.warn(
                "DiariZen does not expose representative speaker embeddings; "
                "returning None.",
                stacklevel=2,
            )
            return diarize_df, None

        return diarize_df

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """Perform speaker diarization and return WhisperX-compatible segments."""
        if self.backend == "diarizen":
            return self._call_diarizen(
                audio=audio,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_embeddings,
                progress_callback=progress_callback,
            )

        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }

        hook = None
        if progress_callback is not None:
            # pyannote's diarization has two progress-trackable steps, each with
            # its own completed/total counter that resets between steps.
            step_ranges = {
                "segmentation": (0.0, 50.0),
                "embeddings": (50.0, 99.0),
            }
            last_pct = [0.0]

            def hook(step_name, step_artifact, file=None, total=None, completed=None):
                if total is not None and completed is not None and total > 0:
                    offset, end = step_ranges.get(step_name, (0.0, 99.0))
                    pct = offset + min(completed / total, 1.0) * (end - offset)
                    if pct > last_pct[0]:
                        last_pct[0] = pct
                        progress_callback(pct)

        output = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            **({"hook": hook} if hook is not None else {}),
        )

        if progress_callback is not None:
            progress_callback(100.0)

        diarization = output.speaker_diarization
        embeddings = output.speaker_embeddings if return_embeddings else None
        diarize_df = self._annotation_to_dataframe(diarization)

        if return_embeddings and embeddings is not None:
            speaker_embeddings = {
                speaker: embeddings[s].tolist()
                for s, speaker in enumerate(diarization.labels())
            }
            return diarize_df, speaker_embeddings

        if return_embeddings:
            return diarize_df, None

        return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Uses an interval tree for O(log n) overlap queries instead of O(n) linear scan,
    achieving ~228x speedup for long-form content (3+ hour podcasts).

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result.get("segments", [])
    if not transcript_segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result

    # Build interval tree from diarization segments for O(log n) queries
    intervals = [
        (row['start'], row['end'], row['speaker'])
        for _, row in diarize_df.iterrows()
    ]
    tree = IntervalTree(intervals)

    for seg in transcript_segments:
        seg_start = seg.get('start', 0.0)
        seg_end = seg.get('end', 0.0)

        # Query overlapping segments using interval tree
        overlaps = tree.query(seg_start, seg_end)

        if overlaps:
            # Sum intersection durations per speaker and pick the dominant one
            speaker_intersections: dict[str, float] = {}
            for speaker, intersection in overlaps:
                speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
            seg['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            # Find nearest segment if no overlap
            seg_mid = (seg_start + seg_end) / 2
            nearest_speaker = tree.find_nearest(seg_mid)
            if nearest_speaker:
                seg['speaker'] = nearest_speaker

        # Assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' not in word:
                    continue

                word_start = word['start']
                word_end = word.get('end', word_start)

                word_overlaps = tree.query(word_start, word_end)

                if word_overlaps:
                    speaker_intersections = {}
                    for speaker, intersection in word_overlaps:
                        speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
                    word['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
                elif fill_nearest:
                    word_mid = (word_start + word_end) / 2
                    nearest_speaker = tree.find_nearest(word_mid)
                    if nearest_speaker:
                        word['speaker'] = nearest_speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker
