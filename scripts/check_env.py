from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> CheckResult:
    section("Python")
    version = ".".join(map(str, sys.version_info[:3]))
    in_venv = sys.prefix != sys.base_prefix
    print(f"Python      : {version} ({sys.executable})")
    print(f"OS          : {platform.platform()}")
    print(f"Virtual env : {'yes' if in_venv else 'NO (activate .venv first)'}")
    ok = sys.version_info[:2] == (3, 12)
    return CheckResult("Python 3.12", ok, version)


def check_torch() -> CheckResult:
    section("PyTorch")
    try:
        import torch
    except ImportError as exc:
        return CheckResult("PyTorch CUDA", False, f"not installed: {exc}")

    print(f"torch       : {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        return CheckResult("PyTorch CUDA", False, "torch.cuda.is_available() is False")

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU         : {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM")
    print(f"Capability  : {major}.{minor} ({arch})")
    print(f"Built archs : {', '.join(arch_list)}")
    if arch not in arch_list:
        return CheckResult(
            "PyTorch CUDA", False, f"{arch} missing from this build; use the cu128 wheel index"
        )

    try:
        x = torch.randn(512, 512, device="cuda")
        value = (x @ x).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return CheckResult("PyTorch CUDA", False, f"GPU computation failed: {exc}")
    print(f"Test matmul : ok (checksum {value:.1f})")
    return CheckResult("PyTorch CUDA", True, f"{props.name} ({arch})")


def check_ctranslate2() -> CheckResult:
    section("CTranslate2")
    try:
        import ctranslate2
    except ImportError as exc:
        return CheckResult("CTranslate2 CUDA", False, f"not installed or DLLs missing: {exc}")

    print(f"ctranslate2 : {ctranslate2.__version__}")
    count = ctranslate2.get_cuda_device_count()
    print(f"CUDA devices: {count}")
    if count == 0:
        return CheckResult("CTranslate2 CUDA", False, "no CUDA device visible")

    compute_types = sorted(ctranslate2.get_supported_compute_types("cuda"))
    print(f"Compute     : {', '.join(compute_types)}")

    return CheckResult("CTranslate2 CUDA", True, f"{count} device(s), model load verified later")


def open_pyaudio():
    import pyaudiowpatch as pyaudio

    return pyaudio, pyaudio.PyAudio()


def list_audio_devices() -> CheckResult:
    section("Audio devices")
    try:
        pyaudio, pa = open_pyaudio()
    except ImportError as exc:
        return CheckResult("Audio devices", False, f"PyAudioWPatch not installed: {exc}")

    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        print(f"{'idx':>4}  {'host API':<22} {'in':>3} {'out':>3} {'rate':>6}  name")
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            host = pa.get_host_api_info_by_index(dev["hostApi"])["name"]
            tag = "  [LOOPBACK]" if dev.get("isLoopbackDevice") else ""
            print(
                f"{i:>4}  {host:<22} {dev['maxInputChannels']:>3} {dev['maxOutputChannels']:>3} "
                f"{int(dev['defaultSampleRate']):>6}  {dev['name']}{tag}"
            )

        mic = pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        loopback = find_default_loopback(pa, wasapi)
        print(f"\nDefault WASAPI mic      : [{mic['index']}] {mic['name']}")
        if loopback is None:
            return CheckResult("Audio devices", False, "no loopback for the default output found")
        print(f"Default output loopback : [{loopback['index']}] {loopback['name']}")
        return CheckResult(
            "Audio devices", True, f"mic {mic['index']}, loopback {loopback['index']}"
        )
    finally:
        pa.terminate()


def find_default_loopback(pa, wasapi) -> dict | None:

    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for loopback in pa.get_loopback_device_info_generator():
        if speakers["name"] in loopback["name"]:
            return loopback
    return None


def record_both(seconds: float, mic_index: int | None, loopback_index: int | None) -> CheckResult:

    section(f"Simultaneous capture ({seconds:g} s)")
    try:
        import numpy as np

        pyaudio, pa = open_pyaudio()
    except ImportError as exc:
        return CheckResult("Simultaneous capture", False, f"missing dependency: {exc}")

    streams = []
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        mic = pa.get_device_info_by_index(
            wasapi["defaultInputDevice"] if mic_index is None else mic_index
        )
        loopback = (
            find_default_loopback(pa, wasapi)
            if loopback_index is None
            else pa.get_device_info_by_index(loopback_index)
        )
        if loopback is None:
            return CheckResult("Simultaneous capture", False, "no loopback device")

        sources = {"mic": mic, "loopback": loopback}
        frames: dict[str, list[bytes]] = {name: [] for name in sources}
        lock = threading.Lock()

        def make_callback(name: str):
            def callback(in_data, frame_count, time_info, status):
                with lock:
                    frames[name].append(in_data)
                return (None, pyaudio.paContinue)

            return callback

        for name, dev in sources.items():
            print(f"{name:<9}: [{dev['index']}] {dev['name']}")
            streams.append(
                pa.open(
                    format=pyaudio.paInt16,
                    channels=dev["maxInputChannels"],
                    rate=int(dev["defaultSampleRate"]),
                    input=True,
                    input_device_index=dev["index"],
                    frames_per_buffer=1024,
                    stream_callback=make_callback(name),
                )
            )

        print("Recording... speak into the mic and play some audio.")
        time.sleep(seconds)
        for stream in streams:
            stream.stop_stream()
        ok = True
        for name, dev in sources.items():
            raw = b"".join(frames[name])
            if not raw:
                print(f"{name:<9}: no data received (is audio playing?)")
                ok = False
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples**2)))
            dbfs = 20 * np.log10(max(rms, 1e-10))
            duration = len(samples) / dev["maxInputChannels"] / dev["defaultSampleRate"]
            print(f"{name:<9}: {duration:.1f} s captured, level {dbfs:.1f} dBFS")
        return CheckResult("Simultaneous capture", ok, "both sources delivered audio" if ok else "")
    except OSError as exc:
        return CheckResult("Simultaneous capture", False, f"stream error: {exc}")
    finally:
        for stream in streams:
            stream.close()
        pa.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Python, CUDA and Windows audio capture")
    parser.add_argument(
        "--record", type=float, metavar="SECONDS", help="capture mic and loopback simultaneously"
    )
    parser.add_argument("--mic", type=int, help="mic device index (default: WASAPI default input)")
    parser.add_argument(
        "--loopback", type=int, help="loopback device index (default: loopback of default output)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [check_python(), check_torch(), check_ctranslate2(), list_audio_devices()]
    if args.record:
        results.append(record_both(args.record, args.mic, args.loopback))

    section("Summary")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name:<22} {result.detail}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
