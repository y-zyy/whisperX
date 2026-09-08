from whisperx.asr import (
    _smooth_language_predictions,
    merge_language_segments,
    suppress_short_language_runs,
)


def _segment(start, end, language, probability=0.9):
    return {
        "start": start,
        "end": end,
        "language": language,
        "language_probability": probability,
        "segments": [(start, end)],
    }


def test_merge_language_segments_merges_only_same_language():
    segments = [
        _segment(0.0, 1.0, "ko"),
        _segment(1.1, 2.0, "ko"),
        _segment(2.0, 3.0, "en"),
        _segment(3.2, 4.0, "en"),
    ]

    merged = merge_language_segments(segments, chunk_size=30, max_gap=0.4)

    assert [(item["start"], item["end"], item["language"]) for item in merged] == [
        (0.0, 2.0, "ko"),
        (2.0, 4.0, "en"),
    ]


def test_merge_language_segments_honours_chunk_size_and_gap():
    segments = [
        _segment(0.0, 2.0, "ko"),
        _segment(2.1, 4.1, "ko"),
        _segment(5.0, 6.0, "ko"),
    ]

    merged = merge_language_segments(segments, chunk_size=4.0, max_gap=0.4)

    assert [(item["start"], item["end"]) for item in merged] == [
        (0.0, 2.0),
        (2.1, 4.1),
        (5.0, 6.0),
    ]


def test_smoothing_replaces_isolated_low_confidence_language():
    predictions = [("ko", 0.92), ("en", 0.41), ("ko", 0.88)]

    assert _smooth_language_predictions(predictions, 0.5) == [
        ("ko", 0.92),
        ("ko", 0.92),
        ("ko", 0.88),
    ]


def test_smoothing_preserves_confident_language_change():
    predictions = [("ko", 0.92), ("en", 0.95), ("ko", 0.88)]

    assert _smooth_language_predictions(predictions, 0.5) == predictions


def test_short_language_run_is_absorbed_into_surrounding_language():
    segments = [
        _segment(0.0, 4.0, "ko"),
        _segment(4.0, 5.5, "en", probability=0.91),
        _segment(5.5, 9.0, "ko"),
    ]

    stable = suppress_short_language_runs(
        segments,
        min_duration=3.0,
        max_gap=0.4,
    )

    assert [(item["start"], item["end"], item["language"]) for item in stable] == [
        (0.0, 9.0, "ko"),
    ]


def test_language_run_at_minimum_duration_is_preserved():
    segments = [
        _segment(0.0, 4.0, "ko"),
        _segment(4.0, 7.0, "en"),
        _segment(7.0, 11.0, "ko"),
    ]

    stable = suppress_short_language_runs(
        segments,
        min_duration=3.0,
        max_gap=0.4,
    )

    assert [item["language"] for item in stable] == ["ko", "en", "ko"]


def test_isolated_short_utterance_after_long_silence_is_preserved():
    segments = [
        _segment(0.0, 4.0, "ko"),
        _segment(6.0, 7.5, "en"),
    ]

    stable = suppress_short_language_runs(
        segments,
        min_duration=3.0,
        max_gap=0.4,
    )

    assert [item["language"] for item in stable] == ["ko", "en"]
