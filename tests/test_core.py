import numpy as np
import pytest

from live_translate.core import FRAME, RATE, Segmenter, SpeakerCluster


def feed(segmenter, start, seconds, speech):
    result = []
    for i in range(round(seconds * RATE / FRAME)):
        segment = segmenter.feed(np.ones(FRAME), start + i * FRAME / RATE, speech)
        if segment:
            result.append(segment)
    return result


def test_endpoint_preserves_speech_end_and_removes_silence():
    s = Segmenter("mic")
    assert not feed(s, 10, 0.6, True)
    segments = feed(s, 10.6, 0.6, False)
    assert len(segments) == 1
    assert segments[0].end == pytest.approx(10.6)
    assert len(segments[0].audio) == 9600
    assert s.flush() is None


def test_loopback_missing_callbacks_still_endpoint():
    s = Segmenter("system")
    feed(s, 10, 0.6, True)
    assert s.expire(10.8) is None
    assert s.expire(11.1).source == "system"


def test_silence_and_brief_clicks_do_not_transcribe():
    s = Segmenter("mic")
    assert feed(s, 0, 3, False) == []
    feed(s, 3, 0.06, True)
    assert feed(s, 3.06, 0.6, False) == []


def test_continuous_speech_is_bounded_and_flush_keeps_tail():
    s = Segmenter("system")
    segments = feed(s, 0, 6.6, True)
    tail = s.flush()
    assert len(segments) == 2
    assert all(len(segment.audio) == 3 * RATE for segment in segments)
    assert len(tail.audio) == round(0.6 * RATE)


def test_pause_shorter_than_endpoint_stays_in_phrase():
    s = Segmenter("mic")
    feed(s, 0, 0.3, True)
    feed(s, 0.3, 0.15, False)
    feed(s, 0.45, 0.3, True)
    segment = s.expire(1.3)
    assert segment.end == pytest.approx(0.75)
    assert len(segment.audio) == 12000


def test_session_labels_stay_stable_across_alternating_speakers():
    cluster = SpeakerCluster()
    assert cluster.identify([1, 0, 0]) == "Speaker 1"
    assert cluster.identify([0, 1, 0]) == "Speaker 2"
    assert cluster.identify([0.99, 0.01, 0]) == "Speaker 1"
    assert cluster.identify([0, 2, 0]) == "Speaker 2"
    assert SpeakerCluster().identify([0, 1, 0]) == "Speaker 1"


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf"), 0]])
def test_invalid_embedding_is_rejected(vector):
    with pytest.raises(ValueError):
        SpeakerCluster().identify(vector)
