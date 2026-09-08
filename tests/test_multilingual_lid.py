from whisperx.asr import _smooth_language_predictions, merge_language_segments


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
