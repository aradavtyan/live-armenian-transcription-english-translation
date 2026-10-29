import queue
import sys
import threading
import time

import numpy as np

from .core import FRAME, RATE, Segmenter, SpeakerCluster


def render(elapsed, speaker, armenian, english):
    seconds = max(0, int(elapsed))
    stamp = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"

    hy = " ".join(armenian.split())
    en = " ".join(english.split())
    return f"[{stamp}] {speaker}  HY: {hy}  EN: {en}"


class Processor:
    def __init__(self, models, origin, threshold=0.86, clock=time.monotonic):
        self.models = models
        self.origin = origin
        self.clock = clock
        self.speakers = SpeakerCluster(threshold)

    def process(self, segment):
        started = self.clock()
        armenian = self.models.transcribe(segment.audio)
        asr_end = self.clock()
        if not armenian:
            return None
        speaker = "Me"
        if segment.source == "system":
            speaker = self.speakers.identify(self.models.embedding(segment.audio))
        speaker_end = self.clock()
        english = self.models.translate(armenian)
        if not english:
            raise RuntimeError("Translation returned empty output")
        print(render(segment.start - self.origin, speaker, armenian, english), flush=True)

        finished = self.clock()
        latency = finished - segment.end
        status = "LATE" if latency > 5 else "OK"
        print(
            f"[{status}] latency={latency:.3f}s queue+endpoint={started - segment.end:.3f}s "
            f"asr={asr_end - started:.3f}s speaker={speaker_end - asr_end:.3f}s "
            f"translation+print={finished - speaker_end:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        return latency


class Framer:
    def __init__(self, rate, first_end, size):
        import soxr
        import webrtcvad

        self.resampler = soxr.ResampleStream(rate, RATE, 1, dtype="float32")
        self.vad = webrtcvad.Vad(2)
        self.pending = np.empty(0, dtype=np.float32)
        self.next_time = first_end - size / rate
        self.last_callback = first_end

    def feed(self, audio, end):
        self.last_callback = end
        self.pending = np.concatenate((self.pending, self.resampler.resample_chunk(audio)))
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
            speech = self.vad.is_speech(pcm, RATE)
            start = self.next_time
            self.next_time += FRAME / RATE
            yield frame, start, speech


def run(capture, models, mic, system, threshold):
    origin = time.monotonic()
    processor = Processor(models, origin, threshold)
    segments = queue.Queue(maxsize=4)
    failures = queue.Queue()
    stopping = threading.Event()
    segmenters = {source: Segmenter(source) for source in ("mic", "system")}
    framers = {}

    def submit(segment):
        if segment is not None:
            try:
                segments.put_nowait(segment)
            except queue.Full as exc:
                raise RuntimeError(
                    "Inference overloaded; segment queue full. Use faster hardware."
                ) from exc

    def worker():
        try:
            while True:
                segment = segments.get()
                if segment is None or stopping.is_set():
                    return
                processor.process(segment)
        except Exception as exc:
            failures.put(exc)

    thread = threading.Thread(target=worker, name="inference", daemon=True)
    thread.start()
    interrupted = False
    try:
        capture.start(mic, system)
        print("Listening to microphone + system audio. Ctrl+C to stop.", file=sys.stderr)
        while True:
            if not failures.empty():
                raise failures.get()
            if not capture.errors.empty():
                raise capture.errors.get()
            try:
                source, audio, rate, end = capture.buffers.get(timeout=0.03)
            except queue.Empty:
                for segmenter in segmenters.values():
                    submit(segmenter.expire(time.monotonic()))
                continue
            framer = framers.get(source)

            if framer is None or end - framer.last_callback > len(audio) / rate + 0.15:
                submit(segmenters[source].flush())
                framer = framers[source] = Framer(rate, end, len(audio))
            for frame, start, speech in framer.feed(audio, end):
                submit(segmenters[source].feed(frame, start, speech))

            for name, segmenter in segmenters.items():
                if name != source and capture.buffers.empty():
                    submit(segmenter.expire(time.monotonic()))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        try:
            capture.close()
            if interrupted and thread.is_alive():
                for segmenter in segmenters.values():
                    segment = segmenter.flush()
                    if segment is not None:
                        while thread.is_alive():
                            try:
                                segments.put(segment, timeout=0.1)
                                break
                            except queue.Full:
                                continue
            else:
                stopping.set()
        finally:
            while thread.is_alive():
                try:
                    segments.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            while thread.is_alive():
                thread.join(timeout=0.1)
        if not failures.empty():
            raise failures.get()
