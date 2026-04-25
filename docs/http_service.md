# IndexTTS2 HTTP Service

A synchronous HTTP service for text-to-speech synthesis with SRT subtitle output.

## Install

```bash
uv sync --extra http_service
```

## Start

```bash
uv run python http_service.py
```

Available options:

```
--host          bind address (default: 0.0.0.0)
--port          listen port (default: 37861)
--model_dir     path to model checkpoints (default: ./checkpoints)
--fp16          enable FP16 inference (default: on)
--deepspeed     enable DeepSpeed acceleration (default: off)
--cuda_kernel   enable compiled CUDA kernels (default: off)
```

## API

### POST /synthesize

Synthesize speech from text. The request blocks until synthesis is complete.

**Request**

```json
{ "text": "Hello, world!" }
```

**Response**

```json
{
  "wav_path": "20260425/3f1a.../0.wav",
  "srt_path": "20260425/3f1a.../0.srt"
}
```

Both paths are relative to `OUTPUT_ROOT` (`/home/fanrui/agbox-paseo-shared/read-only/indextts2_service_voices`) and are meant to be served via filehub.

**Errors**

| Status | Meaning |
|--------|---------|
| 400 | `text` is empty |
| 500 | synthesis failed or output files missing |

## Output Layout

```
<OUTPUT_ROOT>/
  <YYYYMMDD>/        ← UTC date
    <task_id>/       ← uuid4 hex
      0.wav
      0.srt
```

Output files are never deleted by the service; lifecycle is managed externally.
