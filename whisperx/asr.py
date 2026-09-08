import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import replace

import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_ctranslate2_storage
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.schema import SingleSegment, TranscriptionResult, ProgressCallback
from whisperx.vads import Vad, Silero, Pyannote
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


def merge_language_segments(
    segments: List[dict],
    chunk_size: float,
    max_gap: float = 0.4,
) -> List[dict]:
    """Merge adjacent VAD/LID pieces only when they have the same language."""
    if not segments:
        return []

    merged: List[dict] = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "language": segments[0]["language"],
        "language_probability": segments[0].get("language_probability", 0.0),
        "segments": list(segments[0].get("segments", [(segments[0]["start"], segments[0]["end"])])),
    }
    probability_weight = current["end"] - current["start"]

    for segment in segments[1:]:
        gap = segment["start"] - current["end"]
        combined_duration = segment["end"] - current["start"]
        same_language = segment["language"] == current["language"]
        if same_language and gap <= max_gap and combined_duration <= chunk_size:
            duration = segment["end"] - segment["start"]
            total_weight = probability_weight + duration
            if total_weight > 0:
                current["language_probability"] = (
                    current["language_probability"] * probability_weight
                    + segment.get("language_probability", 0.0) * duration
                ) / total_weight
            probability_weight = total_weight
            current["end"] = segment["end"]
            current["segments"].extend(
                segment.get("segments", [(segment["start"], segment["end"])])
            )
        else:
            merged.append(current)
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "language": segment["language"],
                "language_probability": segment.get("language_probability", 0.0),
                "segments": list(segment.get("segments", [(segment["start"], segment["end"])])),
            }
            probability_weight = current["end"] - current["start"]

    merged.append(current)
    return merged


def _smooth_language_predictions(
    predictions: List[Tuple[str, float]],
    probability_threshold: float,
) -> List[Tuple[str, float]]:
    """Remove a single low-confidence language island between matching neighbours."""
    if len(predictions) < 3:
        return predictions

    smoothed = list(predictions)
    for index in range(1, len(predictions) - 1):
        previous, current, following = predictions[index - 1:index + 2]
        if (
            previous[0] == following[0]
            and current[0] != previous[0]
            and (
                current[1] < probability_threshold
                or current[1] < min(previous[1], following[1])
            )
        ):
            smoothed[index] = (previous[0], max(previous[1], following[1]))
    return smoothed


def suppress_short_language_runs(
    segments: List[dict],
    min_duration: float,
    max_gap: float = 0.4,
) -> List[dict]:
    """Absorb short language runs into surrounding context.

    A short isolated VAD region is preserved when no neighbour is close enough;
    this avoids discarding a genuine short utterance after a long silence.
    """
    if min_duration <= 0 or len(segments) < 2:
        return segments

    runs = merge_language_segments(
        segments,
        chunk_size=float("inf"),
        max_gap=max_gap,
    )

    while len(runs) > 1:
        changed = False
        for index, run in enumerate(runs):
            if run["end"] - run["start"] >= min_duration:
                continue

            neighbours = []
            if index > 0 and run["start"] - runs[index - 1]["end"] <= max_gap:
                neighbours.append(runs[index - 1])
            if (
                index + 1 < len(runs)
                and runs[index + 1]["start"] - run["end"] <= max_gap
            ):
                neighbours.append(runs[index + 1])
            if not neighbours:
                continue

            languages = {neighbour["language"] for neighbour in neighbours}
            if len(languages) == 1:
                replacement_language = neighbours[0]["language"]
            else:
                strongest_neighbour = max(
                    neighbours,
                    key=lambda neighbour: (
                        neighbour["end"] - neighbour["start"]
                    ) * neighbour.get("language_probability", 0.0),
                )
                replacement_language = strongest_neighbour["language"]

            run["language"] = replacement_language
            runs = merge_language_segments(
                runs,
                chunk_size=float("inf"),
                max_gap=max_gap,
            )
            changed = True
            break

        if not changed:
            break

    return runs


class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output=None,
    ):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
            hotwords=options.hotwords
        )

        encoder_output = self.encode(features)
        
        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                repetition_penalty=options.repetition_penalty,
                return_scores=True,
            )

        tokens_batch = [x.sequences_ids[0] for x in result]

        avg_logprobs = []
        for res in result:
            seq_len = len(res.sequences_ids[0])
            cum_logprob = res.scores[0] * (seq_len ** options.length_penalty)
            avg_logprobs.append(cum_logprob / (seq_len + 1))

        def decode_batch(tokens: List[List[int]]) -> List[str]:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return {'text': text, 'avg_logprob': avg_logprobs}

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: WhisperModel,
        vad,
        vad_params: dict,
        options: TranscriptionOptions,
        tokenizer: Optional[Tokenizer] = None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        multilingual_lid: bool = False,
        lid_options: Optional[dict] = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self.multilingual_lid = multilingual_lid
        self.lid_options = {
            "languages": ("ko", "en"),
            "window_size": 3.0,
            "probability_threshold": 0.5,
            "min_language_duration": 3.0,
            "max_merge_gap": 0.4,
        }
        if lid_options is not None:
            self.lid_options.update(lid_options)
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        # Pre-process audio and merge chunks as defined by the respective VAD child class 
        # In case vad_model is manually assigned (see 'load_model') follow the functionality of pyannote toolkit
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks =  self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        raw_vad_segments = list(
            self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        )
        task = task or (self.tokenizer.task if self.tokenizer is not None else "transcribe")
        original_tokenizer = self.tokenizer

        if self.multilingual_lid:
            if not self.model.model.is_multilingual:
                raise ValueError("multilingual_lid requires a multilingual Whisper model")
            vad_segments = self.split_vad_segments_by_language(
                audio,
                raw_vad_segments,
                chunk_size=chunk_size,
            )
            if not vad_segments:
                return {"segments": [], "language": language or self.lid_options["languages"][0]}
            language = max(
                self.lid_options["languages"],
                key=lambda code: sum(
                    segment["end"] - segment["start"]
                    for segment in vad_segments
                    if segment["language"] == code
                ),
            )
            self.tokenizer = self._make_tokenizer(language, task)
        else:
            vad_segments = merge_chunks(
                raw_vad_segments,
                chunk_size,
                onset=self._vad_params["vad_onset"],
                offset=self._vad_params["vad_offset"],
            )
            if self.tokenizer is None:
                language = language or self.detect_language(audio)
                self.tokenizer = self._make_tokenizer(language, task)
            else:
                language = language or self.tokenizer.language_code
                if task != self.tokenizer.task or language != self.tokenizer.language_code:
                    self.tokenizer = self._make_tokenizer(language, task)

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            logger.info("Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        decoded: List[Optional[dict]] = [None] * total_segments

        language_groups: Dict[str, List[int]] = {}
        for index, segment in enumerate(vad_segments):
            segment_language = segment.get("language", language)
            language_groups.setdefault(segment_language, []).append(index)

        completed = 0
        for segment_language, indexes in language_groups.items():
            self.tokenizer = self._make_tokenizer(segment_language, task)
            grouped_segments = [vad_segments[index] for index in indexes]
            iterator = self.__call__(
                data(audio, grouped_segments),
                batch_size=batch_size,
                num_workers=num_workers,
            )
            for local_index, out in enumerate(iterator):
                decoded[indexes[local_index]] = out
                completed += 1
                if print_progress:
                    base_progress = (completed / total_segments) * 100
                    percent_complete = base_progress / 2 if combined_progress else base_progress
                    print(f"Progress: {percent_complete:.2f}%...")
                if progress_callback is not None:
                    progress_callback((completed / total_segments) * 100)

        for idx, out in enumerate(decoded):
            if out is None:
                raise RuntimeError(f"Missing decoder output for segment {idx}")
            text = out["text"]
            avg_logprob = out["avg_logprob"]
            if batch_size in [0, 1, None]:
                text = text[0]
                avg_logprob = avg_logprob[0]
            segment_language = vad_segments[idx].get("language", language)
            if verbose:
                print(
                    f"Transcript ({segment_language}): "
                    f"[{round(vad_segments[idx]['start'], 3)} --> "
                    f"{round(vad_segments[idx]['end'], 3)}] {text}"
                )
            output_segment: SingleSegment = {
                "text": text,
                "start": round(vad_segments[idx]["start"], 3),
                "end": round(vad_segments[idx]["end"], 3),
                "avg_logprob": avg_logprob,
            }
            if self.multilingual_lid:
                output_segment["language"] = segment_language
                output_segment["language_probability"] = vad_segments[idx].get(
                    "language_probability", 0.0
                )
            segments.append(output_segment)

        # Restore the tokenizer selected at model construction time.
        self.tokenizer = original_tokenizer

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = replace(self.options, suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}

    def _make_tokenizer(self, language: str, task: str) -> Tokenizer:
        return Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task=task,
            language=language,
        )

    def detect_language(
        self,
        audio: np.ndarray,
        allowed_languages: Optional[Sequence[str]] = None,
        return_probability: bool = False,
    ) -> Union[str, Tuple[str, float]]:
        if audio.shape[0] < N_SAMPLES and allowed_languages is None:
            logger.warning("Audio is shorter than 30s, language detection may be inaccurate")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(
            audio[:N_SAMPLES],
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=max(0, N_SAMPLES - audio.shape[0]),
        )
        encoder_output = self.model.encode(segment)
        candidates = self.model.model.detect_language(encoder_output)[0]
        if allowed_languages is not None:
            allowed = set(allowed_languages)
            candidates = [
                candidate
                for candidate in candidates
                if candidate[0][2:-2] in allowed
            ]
            if not candidates:
                raise ValueError(
                    f"Whisper returned none of the requested LID languages: {sorted(allowed)}"
                )
        language_token, language_probability = max(candidates, key=lambda item: item[1])
        language = language_token[2:-2]
        if allowed_languages is None:
            logger.info(
                f"Detected language: {language} ({language_probability:.2f}) "
                "in first 30s of audio"
            )
        if return_probability:
            return language, language_probability
        return language

    def split_vad_segments_by_language(
        self,
        audio: np.ndarray,
        vad_segments,
        chunk_size: float,
    ) -> List[dict]:
        """Split VAD speech at Whisper-encoder LID changes and merge equal languages."""
        languages = tuple(self.lid_options["languages"])
        window_size = float(self.lid_options["window_size"])
        probability_threshold = float(self.lid_options["probability_threshold"])
        min_language_duration = float(self.lid_options["min_language_duration"])
        max_merge_gap = float(self.lid_options["max_merge_gap"])
        if window_size <= 0:
            raise ValueError("lid window_size must be greater than zero")
        if len(languages) < 2:
            raise ValueError("lid languages must contain at least two language codes")

        labeled_pieces: List[dict] = []
        for vad_segment in vad_segments:
            start, end = float(vad_segment.start), float(vad_segment.end)
            duration = end - start
            if duration <= 0:
                continue

            if duration <= window_size:
                centers = [(start + end) / 2]
            else:
                centers = list(
                    np.arange(start + window_size / 2, end, window_size)
                )
                final_center = end - window_size / 2
                if not centers or final_center > centers[-1] + 1e-6:
                    centers.append(final_center)

            predictions: List[Tuple[str, float]] = []
            for center in centers:
                probe_start = max(start, min(center - window_size / 2, end - window_size))
                probe_end = min(end, probe_start + window_size)
                f1 = max(0, int(probe_start * SAMPLE_RATE))
                f2 = min(audio.shape[0], int(probe_end * SAMPLE_RATE))
                predictions.append(
                    self.detect_language(
                        audio[f1:f2],
                        allowed_languages=languages,
                        return_probability=True,
                    )
                )

            predictions = _smooth_language_predictions(
                predictions,
                probability_threshold=probability_threshold,
            )
            boundaries = [start]
            boundaries.extend(
                (centers[index - 1] + centers[index]) / 2
                for index in range(1, len(centers))
            )
            boundaries.append(end)

            for index, (segment_language, probability) in enumerate(predictions):
                piece_start, piece_end = boundaries[index], boundaries[index + 1]
                labeled_pieces.append(
                    {
                        "start": piece_start,
                        "end": piece_end,
                        "language": segment_language,
                        "language_probability": probability,
                        "segments": [(piece_start, piece_end)],
                    }
                )

        stable_runs = suppress_short_language_runs(
            labeled_pieces,
            min_duration=min_language_duration,
            max_gap=max_merge_gap,
        )
        return merge_language_segments(
            stable_runs,
            chunk_size=chunk_size,
            max_gap=max_merge_gap,
        )


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="default",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad]= None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model: Optional[WhisperModel] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    use_auth_token: Optional[Union[str, bool]] = None,
    multilingual_lid: bool = False,
    lid_options: Optional[dict] = None,
) -> FasterWhisperPipeline:
    """Load a Whisper model for inference.
    Args:
        whisper_arch - The name of the Whisper model to load.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
            Use "default" to automatically select based on device (float16 for GPU, float32 for CPU).
        vad_model - The vad model to manually assign.
        vad_method - The vad method to use. vad_model has a higher priority if it is not None.
        options - A dictionary of options to use for the model.
        language - The language of the model. (use English for now)
        model - The WhisperModel instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info(f"Compute type not specified, defaulting to {compute_type} for device {device}")

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         local_files_only=local_files_only,
                         cpu_threads=threads,
                         use_auth_token=use_auth_token)
    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        logger.info("No language specified, language will be detected for each audio file (increases inference time)")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": model.model.is_multilingual,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "chunk_size": 30, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Note: manually assigned vad_model has higher priority than vad_method!
    if vad_model is not None:
        print("Use manually assigned vad_model. vad_method is ignored.")
        vad_model = vad_model
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            if device == 'cuda':
                device_vad = f'cuda:{device_index}'
            else:
                device_vad = device
            vad_model = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
        multilingual_lid=multilingual_lid,
        lid_options=lid_options,
    )
