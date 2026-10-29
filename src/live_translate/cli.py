import argparse
import os
import sys


def parser():
    result = argparse.ArgumentParser(description="Local Armenian → English live translation")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="list microphone and loopback/monitor inputs")
    download = commands.add_parser("download", help="download model weights (requires internet)")
    download.add_argument("--models", default="models")
    live = commands.add_parser("run", help="run local inference on both audio sources")
    live.add_argument("--mic", type=int, required=True)
    live.add_argument("--system", type=int, required=True)
    live.add_argument("--models", default="models")
    live.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    live.add_argument("--speaker-threshold", type=float, default=0.86)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.command == "download":
            from .models import download_models

            download_models(args.models)
        elif args.command == "devices":
            from .audio.capture import Capture

            capture = Capture()
            try:
                capture.describe()
            finally:
                capture.close()
        else:
            if not 0 < args.speaker_threshold < 1:
                raise ValueError("--speaker-threshold must be between 0 and 1")

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from .audio.capture import Capture
            from .models import LocalModels
            from .pipeline import run

            print("Loading and warming local models...", file=sys.stderr)
            models = LocalModels(args.models, args.device)
            models.warmup()
            run(Capture(), models, args.mic, args.system, args.speaker_threshold)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
