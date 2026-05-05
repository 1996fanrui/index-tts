"""IndexTTS2 HTTP service.

Tutorial:
    uv sync --extra http_service
    uv run python http_service.py

Endpoint:
    POST /synthesize  body: {"text": "..."}  returns {"wav_path": "...", "srt_path": "..."}

Paths in the response are relative to --output_root and meant to be served via filehub.
"""

import argparse
import asyncio
import datetime
import logging
import os
import sys
import uuid
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from indextts.infer_v2 import IndexTTS2

OUTPUT_ROOT = "/home/fanrui/agbox-paseo-shared/read-only/indextts2_service_voices"
DEFAULT_SPEAKER_AUDIO = (
    "/home/fanrui/code/tts-service/materials/"
    "en-US-AndrewMultilingual-Relieved-Audio.wav"
)
DEFAULT_MODEL_DIR = os.path.join(current_dir, "checkpoints")

logger = logging.getLogger("indextts2.http")


class SynthesizeRequest(BaseModel):
    text: str


class SynthesizeResponse(BaseModel):
    wav_path: str
    srt_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS2 HTTP Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=37861)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--speaker_audio", default=DEFAULT_SPEAKER_AUDIO)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--cuda_kernel", action="store_true", default=False)
    return parser.parse_args()


def validate_model_dir(model_dir: str) -> None:
    if not os.path.isdir(model_dir):
        raise SystemExit(f"Model directory not found: {model_dir}")
    required = ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]
    missing = [f for f in required if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        raise SystemExit(f"Missing model files in {model_dir}: {missing}")


def validate_speaker_audio(speaker_audio: str) -> None:
    if not os.path.isfile(speaker_audio):
        raise SystemExit(f"Default speaker audio not found: {speaker_audio}")


def build_app(tts: IndexTTS2, output_root: str, speaker_audio: str) -> FastAPI:
    app = FastAPI(title="IndexTTS2 HTTP Service")
    # Process-wide lock: GPU inference must run one at a time.
    inference_lock = asyncio.Lock()

    @app.post("/synthesize", response_model=SynthesizeResponse)
    async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text cannot be empty")

        task_id = uuid.uuid4().hex
        date_dir = datetime.datetime.utcnow().strftime("%Y%m%d")
        rel_dir = os.path.join(date_dir, task_id)
        abs_dir = os.path.join(output_root, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        wav_rel = os.path.join(rel_dir, "0.wav")
        srt_rel = os.path.join(rel_dir, "0.srt")
        wav_abs = os.path.join(output_root, wav_rel)
        srt_abs = os.path.join(output_root, srt_rel)

        logger.info("synthesize task_id=%s text_len=%d", task_id, len(text))

        # Serialize all GPU inference; offload to a thread so the event loop is free.
        async with inference_lock:
            try:
                await asyncio.to_thread(
                    tts.infer,
                    spk_audio_prompt=speaker_audio,
                    text=text,
                    output_path=wav_abs,
                    stream_return=False,
                    interval_silence=0,
                )
            except Exception as e:
                logger.exception("synthesize failed task_id=%s", task_id)
                raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e

        if not os.path.isfile(wav_abs):
            raise HTTPException(status_code=500, detail="wav file was not generated")
        if not os.path.isfile(srt_abs):
            raise HTTPException(status_code=500, detail="srt file was not generated")

        return SynthesizeResponse(wav_path=wav_rel, srt_path=srt_rel)

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    validate_model_dir(args.model_dir)
    validate_speaker_audio(args.speaker_audio)
    os.makedirs(args.output_root, exist_ok=True)

    logger.info("loading IndexTTS2 model from %s", args.model_dir)
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        use_cuda_kernel=args.cuda_kernel,
    )
    logger.info("model loaded")

    app = build_app(tts, args.output_root, args.speaker_audio)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
