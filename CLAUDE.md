# IndexTTS2 项目说明

本仓库维护一个 FastAPI TTS 服务（`http_service.py`），通过 HTTP 提供文本转语音 + SRT 字幕。详细使用参考 `docs/http_service.md`。

## 启动 HTTP 服务

默认使用 `uv run python http_service.py` 启动。
`uv run` 在项目目录下会使用当前项目 `.venv`，并在运行前按 lock 同步依赖；只有临时诊断且明确需要跳过 uv 同步时，才直接使用 `.venv/bin/python http_service.py`。
