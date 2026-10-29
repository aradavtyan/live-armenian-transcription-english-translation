# Live Armenian → English translation

A terminal application capturing a microphone and system audio simultaneously. It prints
elapsed time, speaker, Armenian transcription and English translation. Inference is local;
model downloads are a separate setup step. No recordings or transcripts are saved.

**Status:** implemented baseline, with automated model-free tests. Live Armenian accuracy,
real two-speaker identification, GPU memory use and the **≤5-second end-of-phrase latency
requirement remain unverified on the interview laptop**. Do not treat this as a demonstrated
passing submission. See [requirements review](docs/requirements-review.md).

## Setup

Python 3.12 is the supported application environment. Use a virtual environment.
For the original Windows/NVIDIA target (including RTX 50-series), install the appropriate
[PyTorch wheel](https://pytorch.org/get-started/locally/) first. The original setup used CUDA
12.8 wheels; the installed driver, PyTorch and CTranslate2 CUDA libraries must be compatible.
For faster-whisper's CUDA 12 backend, install **cuBLAS for CUDA 12 and cuDNN 9** and make
their library directories discoverable (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux).
PyTorch CUDA availability alone does not establish that CTranslate2 can load these libraries;
follow the [upstream GPU setup instructions](https://github.com/SYSTRAN/faster-whisper#gpu).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
live-translate download
live-translate devices
live-translate run --mic 1 --system 12 --device cuda
```

Replace both indices with those printed by `devices`. Model weights go into ignored `models/`;
allow several GB for weights and the GPU libraries. If using another directory, pass the
same `--models PATH` to `download` and `run`. Download requires internet; `run` enables
Hugging Face offline mode and loads only local paths. No audio or text is sent to a service.

On Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
live-translate download
live-translate devices
live-translate run --mic 1 --system 4 --device cpu
```

Install PortAudio with your OS package manager if `sounddevice` cannot load it (for example,
`libportaudio2` on Debian/Ubuntu). CPU mode is available for functional testing; it may exceed
five seconds. CUDA mode uses int8/float16 Whisper and float16 translation; WavLM stays on CPU.
Models load and warm up **before** audio capture starts. Ctrl+C stops capture and finishes
queued segments. Capture errors, buffer overflow and inference failure stop the session
with an error instead of silently discarding phrases.

## Audio routing

- **Windows:** select your microphone as `--mic` and the `[loopback]` input corresponding to
  the call's actual playback device as `--system`. WASAPI loopback requires no virtual cable
  and you still hear the call. It captures all sound on that output, including notifications.
- **Linux (PulseAudio/PipeWire):** select the microphone and the monitor of the call's output
  sink. If the monitor is absent from the device list, expose/reroute it as a recording source
  using your audio control panel or a virtual input. In `pavucontrol`, set the application's
  second recording stream to the relevant output monitor. Verify routing before the demo;
  selecting two microphones does not capture the call.
- **macOS:** install a loopback driver such as BlackHole. Make a Multi-Output Device containing
  BlackHole and headphones in Audio MIDI Setup; route the call to that output. Select your
  physical microphone and BlackHole input. Grant terminal microphone permission.

Use headphones to reduce call audio being captured again as `Me`. Echo cancellation is out
of scope. Device indices can change after reconnecting hardware; run `devices` again.
Linux/macOS routing is manual; the program cannot verify that an arbitrary input is a monitor.

## Output and latency

```text
[00:01:12] Speaker 1  HY: Բարև, ինչպե՞ս ես։  EN: Hi, how are you?
[00:01:15] Me  HY: Լավ եմ, շնորհակալություն։  EN: I'm fine, thank you.
```

These are illustrative, not recorded model results. Microphone segments always use `Me`.
System speakers are assigned session-scoped `Speaker N` labels in order of first appearance.
The timestamp is the segment's start relative to capture start. Diagnostics on stderr give
endpoint/queue, ASR, speaker and translation timings, with `LATE` above five seconds.
Latency runs from the last detected speech frame through flushing the English line to stdout;
it includes endpointing and queue wait. Callback arrival timestamps approximate audio time
and exclude unknown device/OS buffering. Measure real acoustic end-to-display latency too.
A bounded queue prevents unlimited backlog; overload exits visibly rather than dropping audio.

## Models and trade-offs

- **ASR:** [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium),
  multilingual Whisper via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
  Language is forced to `hy`, task to transcription, with greedy decoding to limit latency.
  Medium is an initial quality/memory compromise for low-resource Armenian, not a measured
  winner. Small is cheaper but needs Armenian evaluation; large-v3 may improve accuracy at
  a memory/latency cost. Distilled English-only models cannot meet this brief.
- **Translation:** [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M),
  `hye_Armn` → `eng_Latn`, greedy decoding. Chosen for explicit Armenian support and modest
  size. Larger NLLB variants cost more memory. Whisper's direct English task would avoid a
  separate translator, but the required Armenian original would still need transcription.
  NLLB is a research model with a CC-BY-NC license, not a production/certified translator;
  its model card describes the limitations. Short chunks lose conversational context.
- **Speaker identification:** [WavLM base-plus speaker verification](https://huggingface.co/microsoft/wavlm-base-plus-sv),
  16 kHz embeddings with online cosine centroid matching. Default threshold `0.86` comes
  from the model card's example and **requires calibration** on the actual callers. Adjust
  `--speaker-threshold` (0–1): higher creates more labels, lower can merge different people.
  ECAPA-TDNN and pyannote diarization are alternatives; pyannote offers richer turn handling
  but adds setup/compute and some gated model dependencies. No names or cross-session state.
- **Endpointing:** WebRTC VAD in 30 ms frames, 450 ms trailing silence, minimum 240 ms speech,
  maximum three-second chunks. Stateful soxr resampling converts native-rate multichannel
  audio to 16 kHz mono. Energy-only gating was avoided because playback noise can trigger it.

Model choices are provisional until compared on representative Armenian conversations.

## Architecture and validation

Audio callbacks enqueue bounded native-rate buffers. The main thread independently resamples,
frames and endpoints each source. A separate worker runs ASR, speaker matching (system only),
translation and terminal output. Model interfaces and pure segmentation/clustering logic are
separate from device capture. See [architecture](docs/architecture.md).

```bash
pytest
ruff check .
python -m live_translate --help
```

On the Windows CUDA demo machine, the original `python scripts/check_env.py --record 3`
checks the GPU stack and simultaneous capture without writing audio. It is a Windows-specific
diagnostic, not the application or proof of model performance.

Before the interview, download models, disconnect networking, then run the live command with
two remote Armenian speakers alternating and returning after pauses. Verify both sources,
Armenian/English quality, label consistency and every latency diagnostic. Include rapid turns,
a long phrase, a silent playback gap and several minutes of simultaneous source activity.
Use a stopwatch/video locally to check audio-buffer latency. Record actual findings in the
review document; no benchmark results are claimed here.

## Known limitations and next improvements

Online speaker matching is a baseline, not full diarization: two people speaking inside the
same chunk receive one label, and overlapping speakers are not separated. Short utterances
(padded for the speaker model), noise and voice changes can fragment or merge identities.
The three-second cap bounds work but can split words/sentences. VAD can miss quiet speech or
trigger on music; Whisper can hallucinate. Calibration on Armenian speech, speaker-change
segmentation, and quality/latency measurements on the demo GPU are the next priorities.
There is no hard real-time guarantee, and sustained load can exceed five seconds even before
the bounded queue fills. OS audio buffering and routing also need live verification.

## Submission process

The brief requires exclusive Claude Cowork use and complete unedited Markdown exports of
**every** project conversation, with original histories available at the interview. The
repository contains no such exports. This implementation was updated using Codex, so it
cannot claim compliance with the exclusive-tool requirement. Keep actual histories; never
fabricate or relabel them. See [chats/README.md](chats/README.md).
