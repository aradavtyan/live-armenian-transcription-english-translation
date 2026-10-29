from pathlib import Path

ASR_ID = "Systran/faster-whisper-medium"
MT_ID = "facebook/nllb-200-distilled-600M"
SPEAKER_ID = "microsoft/wavlm-base-plus-sv"


def download_models(root):
    from huggingface_hub import snapshot_download

    for name, repo in (("asr", ASR_ID), ("translation", MT_ID), ("speaker", SPEAKER_ID)):
        snapshot_download(
            repo,
            local_dir=str(Path(root) / name),
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.model", "*.txt"],
        )


class LocalModels:
    def __init__(self, root, device="cpu"):
        import torch
        from faster_whisper import WhisperModel
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioXVector,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        root = Path(root).resolve()
        for name in ("asr", "translation", "speaker"):
            if not (root / name).is_dir():
                raise ValueError(f"Missing {root / name}; run live-translate download first")
        self.torch = torch
        self.device = device
        self.asr = WhisperModel(
            str(root / "asr"),
            device=device,
            compute_type="int8_float16" if device == "cuda" else "int8",
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "translation", src_lang="hye_Armn", local_files_only=True
        )
        self.mt = (
            AutoModelForSeq2SeqLM.from_pretrained(
                root / "translation",
                local_files_only=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            .to(device)
            .eval()
        )
        self.extractor = AutoFeatureExtractor.from_pretrained(
            root / "speaker", local_files_only=True
        )

        self.speaker = AutoModelForAudioXVector.from_pretrained(
            root / "speaker", local_files_only=True
        ).eval()

    def transcribe(self, audio):
        segments, _ = self.asr.transcribe(
            audio,
            language="hy",
            task="transcribe",
            beam_size=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def translate(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        if tokens.input_ids.shape[-1] > 512:
            raise ValueError("Translation input exceeds NLLB's 512-token training length")
        tokens = tokens.to(self.device)
        with self.torch.inference_mode():
            result = self.mt.generate(
                **tokens,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=256,
                num_beams=1,
            )
        if result.shape[-1] >= 257:
            raise RuntimeError("Translation reached its token limit; output may be incomplete")
        return self.tokenizer.decode(result[0], skip_special_tokens=True).strip()

    def embedding(self, audio):

        import numpy as np

        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        with self.torch.inference_mode():
            return self.speaker(**inputs).embeddings[0].cpu().numpy()

    def warmup(self):
        import numpy as np

        self.transcribe(np.zeros(16000, dtype=np.float32))
        self.translate("Բարև։")
        self.embedding(np.zeros(16000, dtype=np.float32))
