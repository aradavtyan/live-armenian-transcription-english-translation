from dataclasses import dataclass

import numpy as np

RATE = 16000
FRAME = 480


@dataclass
class Segment:
    source: str
    audio: np.ndarray
    start: float
    end: float


class Segmenter:
    def __init__(self, source, silence=0.45, maximum=3.0, minimum=0.24):
        self.source = source
        self.silence = silence
        self.maximum = maximum
        self.minimum = minimum
        self.frames = []
        self.start = None
        self.last_speech = None
        self.speech_frames = 0

    def feed(self, audio, start, speech):
        if speech:
            if self.start is None:
                self.start = start
            self.last_speech = start + len(audio) / RATE
            self.speech_frames += 1
        if self.start is None:
            return None
        self.frames.append(audio.copy())
        end = start + len(audio) / RATE
        if end - self.start >= self.maximum:
            return self.flush()
        return self.expire(end)

    def expire(self, now):
        if self.last_speech is not None and now - self.last_speech >= self.silence:
            return self.flush()
        return None

    def flush(self):
        result = None
        if self.start is not None and self.speech_frames * FRAME / RATE >= self.minimum:
            count = round((self.last_speech - self.start) * RATE)
            result = Segment(
                self.source, np.concatenate(self.frames)[:count], self.start, self.last_speech
            )
        self.frames = []
        self.start = self.last_speech = None
        self.speech_frames = 0
        return result


class SpeakerCluster:
    def __init__(self, threshold=0.86):
        if not 0 < threshold < 1:
            raise ValueError("Speaker threshold must be between 0 and 1")
        self.threshold = threshold
        self.centers = []

    def identify(self, embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm < 1e-8:
            raise ValueError("Invalid speaker embedding")
        vector = vector / norm
        scores = [float(center @ vector) for center in self.centers]
        best = int(np.argmax(scores)) if scores else -1
        if best < 0 or scores[best] < self.threshold:
            self.centers.append(vector)
            best = len(self.centers) - 1
        else:
            center = self.centers[best] * 0.9 + vector * 0.1
            self.centers[best] = center / np.linalg.norm(center)
        return f"Speaker {best + 1}"
