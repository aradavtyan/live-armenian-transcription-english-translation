import numpy as np
import pytest

from live_translate.core import FRAME, Segment
from live_translate.pipeline import Framer, Processor, render


class Models:
    def transcribe(self, audio):
        return "Բարև։"

    def embedding(self, audio):
        return [1, 0]

    def translate(self, text):
        assert text == "Բարև։"
        return "Hello."


def test_mic_is_me_and_never_uses_speaker_model(capsys):
    models = Models()
    models.embedding = lambda _: pytest.fail("Mic must not use speaker identification")
    processor = Processor(models, origin=10, clock=lambda: 15)
    latency = processor.process(Segment("mic", np.zeros(16000), 11, 12))
    out = capsys.readouterr()
    assert out.out == "[00:00:01] Me  HY: Բարև։  EN: Hello.\n"
    assert latency == 3
    assert "[OK]" in out.err


def test_system_label_and_latency_include_queue_wait(capsys):
    processor = Processor(Models(), origin=0, clock=lambda: 20)
    processor.process(Segment("system", np.zeros(16000), 10, 12))
    out = capsys.readouterr()
    assert "Speaker 1" in out.out
    assert "[LATE] latency=8.000s" in out.err


def test_empty_asr_does_not_hallucinate_translation(capsys):
    models = Models()
    models.transcribe = lambda _: ""
    models.translate = lambda _: pytest.fail("Silence must not be translated")
    assert Processor(models, 0).process(Segment("mic", np.zeros(480), 0, 1)) is None
    assert not capsys.readouterr().out


def test_render_handles_long_session_and_embedded_newlines():
    assert render(3661, "Me", "Բարև։\n", "Hello.\nAgain.") == (
        "[01:01:01] Me  HY: Բարև։  EN: Hello. Again."
    )


def test_streaming_resampling_preserves_frame_sizes_and_timeline():
    framer = Framer(48000, 1.03, 1440)
    frames = []
    for i in range(100):
        frames.extend(framer.feed(np.zeros(1440, dtype=np.float32), 1.03 + i * 0.03))
    assert 97 <= len(frames) <= 100
    assert all(len(frame) == FRAME and not speech for frame, _, speech in frames)
    assert frames[0][1] == pytest.approx(1)
    assert frames[-1][1] == pytest.approx(1 + (len(frames) - 1) * 0.03)


class FakeBuffers:
    def __init__(self):
        self.sources = iter(("mic", "system"))

    def get(self, timeout):
        import time

        try:
            source = next(self.sources)
        except StopIteration:
            raise KeyboardInterrupt from None
        return source, np.ones(4800, dtype=np.float32), 16000, time.monotonic()

    def empty(self):
        return True


class FakeCapture:
    def __init__(self):
        import queue

        self.errors = queue.Queue()
        self.buffers = FakeBuffers()
        self.closed = False

    def start(self, mic, system):
        assert (mic, system) == (1, 2)

    def close(self):
        self.closed = True


class SpeechFramer:
    def __init__(self, rate, end, size):
        self.last_callback = end

    def feed(self, audio, end):
        self.last_callback = end
        for i in range(10):
            yield audio[i * FRAME : (i + 1) * FRAME], end - 0.3 + i * 0.03, True


def test_dual_source_session_flushes_active_speech_and_closes_on_interrupt(monkeypatch, capsys):
    from live_translate import pipeline

    monkeypatch.setattr(pipeline, "Framer", SpeechFramer)
    capture = FakeCapture()
    pipeline.run(capture, Models(), 1, 2, 0.86)
    output = capsys.readouterr().out
    assert "Me  HY:" in output
    assert "Speaker 1  HY:" in output
    assert capture.closed


def test_capture_start_failure_closes_devices_and_releases_worker():
    import threading

    from live_translate.pipeline import run

    capture = FakeCapture()

    def fail(*args):
        raise OSError("device unavailable")

    capture.start = fail
    with pytest.raises(OSError, match="device unavailable"):
        run(capture, Models(), 1, 2, 0.86)
    assert capture.closed
    assert not any(t.name == "inference" for t in threading.enumerate())


def test_inference_failure_propagates_after_closing_devices(monkeypatch):
    from live_translate import pipeline

    monkeypatch.setattr(pipeline, "Framer", SpeechFramer)
    capture = FakeCapture()
    models = Models()

    def fail(audio):
        raise RuntimeError("model failed")

    models.transcribe = fail
    with pytest.raises(RuntimeError, match="model failed"):
        pipeline.run(capture, models, 1, 2, 0.86)
    assert capture.closed
