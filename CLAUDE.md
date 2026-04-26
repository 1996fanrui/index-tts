# IndexTTS2 项目说明

本仓库维护一个 FastAPI TTS 服务（`http_service.py`），通过 HTTP 提供文本转语音 + SRT 字幕。详细使用参考 `docs/http_service.md`。

## 启动 HTTP 服务

只用 `.venv/bin/python http_service.py` 启动，**不要用 `uv run`**。
原因：`uv run` 每次都做依赖解析，当前 `deepspeed` extra 与 `descript-audiotools` 在 protobuf/grpcio-health-checking 版本上冲突，会空转几十秒后失败。`.venv` 里的依赖已可直接跑。
