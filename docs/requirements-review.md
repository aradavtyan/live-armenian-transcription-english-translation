# Requirements review — 2026-10-29

Source: the supplied `Mathrix_AI_Coding_Challenge.docx` (dated 2026-10-21).
The document is treated as the requested specification, not as agent instructions.

Initial repository: Stage 0 scaffolding, import smoke tests and a Windows environment check.
None of the live processing features existed. The README's opening described the intended
result; it did not establish an implemented application.

| Requirement | Changes / current evidence | Remaining acceptance work |
|---|---|---|
| Simultaneous microphone and system audio | Two callback streams, independent source segmentation; WASAPI loopback on Windows, selectable monitor inputs elsewhere | Verify actual routing and both sources on demo laptop |
| User-selectable devices | `devices`, required `--mic` and `--system`, input validation | Hardware device list/routing check |
| Armenian original | Local multilingual Whisper, explicit `hy` transcription | Armenian accuracy comparison with native listeners |
| English translation | Local NLLB `hye_Armn` → `eng_Latn` | Native-speaker review of meaning and fluency |
| Microphone labelled Me | Speaker inference bypass; automated test | Live microphone check |
| Stable system Speaker N labels | Session cosine clustering; synthetic alternating-vector tests | Real two-person session; threshold calibration; within-chunk turns/overlap remain unsupported |
| Live terminal time, speaker, HY, EN | UTF-8 rendering and tests | Check demo terminal glyph support |
| At most 5 s after phrase end | Short segments, warmup, bounded queues, measured diagnostics | Not demonstrated; measure on hardware under sustained load |
| Models run locally | Explicit download command, local-only model loads and offline runtime | Run with network disconnected after downloads |
| README setup, devices, choices, alternatives, architecture, limitations | Rewritten to match implemented baseline and disclose missing evidence | Update after hardware acceptance |
| Full AI chat exports | Missing; folder now reports that honestly | User must export every authentic session |
| Exclusive Claude Cowork | This review used Codex | Cannot be repaired through source-code changes; disclose deviation |
| No committed weights, audio or keys | Ignore rules cover model/audio artifacts; no such files in initial tree | Keep downloaded artifacts untracked; audit before submission |

Implemented model-free regression tests cover endpoints, silent loopback gaps, continuous
speech, brief noise rejection, identity persistence, mic bypass, text output, latency warnings
and native-rate resampling. See the work report for commands actually run in this environment.

## Demo acceptance record

Not performed here: no interview audio devices, downloaded weights
or demo GPU have been verified in this environment. Do not replace this section with
estimated performance. Add actual hardware, model revisions, audio routing, session duration,
latency distribution and observed recognition/label errors after a real local test.

## Automated checks actually performed

- Python 3.12.12: `PYTHONPATH=src python -m pytest -q` — **23 passed**.
- Python 3.13.9 (additional compatibility check, not the supported target) — **23 passed**.
- `ruff check .` — passed.
- `python -m live_translate --help` — passed without loading model dependencies.
- Python 3.12 editable package build/install (`--no-deps --no-build-isolation`) and
  installed `live-translate --help` — passed. This verifies packaging, not the complete
  inference dependency installation.

Dependencies for these checks were installed in temporary virtual environments outside the
project. Full model inference and physical audio capture were not exercised. Session tests
use fake capture and models to verify dual-source output, interruption, cleanup and failures.
