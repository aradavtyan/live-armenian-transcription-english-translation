import queue
import sys
import time

import numpy as np


class Capture:
    def __init__(self):
        self.buffers = queue.Queue(maxsize=256)
        self.errors = queue.Queue()
        self.streams = []
        self.pa = None
        if sys.platform == "win32":
            import pyaudiowpatch as backend

            self.backend = backend
            self.pa = backend.PyAudio()
        else:
            import sounddevice as backend

            self.backend = backend

    def devices(self):
        if self.pa:
            return [self.pa.get_device_info_by_index(i) for i in range(self.pa.get_device_count())]
        return [dict(d, index=i) for i, d in enumerate(self.backend.query_devices())]

    def describe(self):
        for d in self.devices():
            channels = d.get("maxInputChannels", d.get("max_input_channels", 0))
            loop = " [loopback]" if d.get("isLoopbackDevice") else ""
            print(f"{int(d['index']):3}: inputs={channels} {d['name']}{loop}")

    def _put(self, source, audio, rate, status):
        if status:
            self.errors.put(RuntimeError(f"{source}: audio capture status {status}"))
        try:
            self.buffers.put_nowait((source, audio.copy(), rate, time.monotonic()))
        except queue.Full:
            self.errors.put(RuntimeError("Audio buffer overflow; capture cannot keep up"))

    def start(self, mic, system):
        if mic == system:
            raise ValueError("Microphone and system audio must be different devices")
        devices = {int(d["index"]): d for d in self.devices()}
        for source, index in (("mic", mic), ("system", system)):
            if index not in devices:
                raise ValueError(f"Unknown {source} device {index}; use devices command")
            dev = devices[index]
            channels = int(dev.get("maxInputChannels", dev.get("max_input_channels", 0)))
            if channels < 1:
                raise ValueError(f"Device {index} is not an input/loopback device")
            if self.pa and source == "system" and not dev.get("isLoopbackDevice"):
                raise ValueError("Select a WASAPI [loopback] device for --system")
            if self.pa and source == "mic" and dev.get("isLoopbackDevice"):
                raise ValueError("Select a microphone, not a loopback, for --mic")
            rate = int(dev.get("defaultSampleRate", dev.get("default_samplerate")))
            if self.pa:

                def callback(
                    data, count, timing, status, source=source, rate=rate, channels=channels
                ):
                    audio = np.frombuffer(data, dtype=np.float32).reshape(-1, channels).mean(1)
                    self._put(source, audio, rate, status)
                    return None, self.backend.paContinue

                stream = self.pa.open(
                    format=self.backend.paFloat32,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=index,
                    frames_per_buffer=round(rate * 0.03),
                    stream_callback=callback,
                    start=False,
                )
            else:

                def callback(data, count, timing, status, source=source, rate=rate):
                    self._put(source, data.mean(axis=1), rate, status)

                stream = self.backend.InputStream(
                    device=index,
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=round(rate * 0.03),
                    callback=callback,
                )
            self.streams.append(stream)
        for stream in self.streams:
            if self.pa:
                stream.start_stream()
            else:
                stream.start()

    def close(self):
        error = None
        for stream in self.streams:
            try:
                if self.pa:
                    stream.stop_stream()
                else:
                    stream.stop()
            except Exception as exc:
                error = error or exc
            try:
                stream.close()
            except Exception as exc:
                error = error or exc
        self.streams.clear()
        if self.pa:
            self.pa.terminate()
            self.pa = None
        if error is not None:
            raise error
