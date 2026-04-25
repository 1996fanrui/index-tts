# IndexTTS2 HTTP Service 需求

## 背景

需要在工作流（n8n 等）中调用 IndexTTS2 做文本转语音并拿到细粒度 SRT 字幕。

- 工作流平台只支持 HTTP，gRPC 不可用。
- 业界 TTS 部署没有返回细粒度 SRT 的现成方案；细粒度 SRT 是本仓自定义扩展（`indextts/infer_v2.py:403` `generate_srt`）。
- 已有的 `tts-service`（gRPC）服务不在改造范围内，继续保留给后端轮询使用。
- 调用方仅限作者自己的工作流，请求量低，不开放给外部。

## 服务形态

在本仓库（`/home/fanrui/code/indextts2`）内新增一个 FastAPI 应用，与 IndexTTS2 同进程加载模型，直接调用 `from indextts.infer_v2 import IndexTTS2`，不跨进程。

- 同步 HTTP 接口：请求阻塞直到合成完成再返回。
- 不引入数据库、队列、回调、批量、鉴权。
- 不引入对 `tts-service` 的依赖。

## 接口

### POST /synthesize

请求体：

```json
{
  "text": "待合成的文本"
}
```

响应体：

```json
{
  "wav_path": "20260425/<task_id>/0.wav",
  "srt_path": "20260425/<task_id>/0.srt"
}
```

- 请求只接受 `text` 一个字段；其他参数（speaker audio、model_dir、cfg_path 等）在服务代码里硬编码，第一版不暴露。
- 响应只返回两个相对路径，不返回任何其他字段，不返回文件字节。
- 路径相对于配置的输出根目录，由 filehub 托管对外读取。

## 硬编码参数（v1）

- `default_speaker_audio`: `/home/fanrui/code/tts-service/materials/en-US-AndrewMultilingual-Relieved-Audio.wav`
- `model_dir`、`cfg_path`、`use_fp16`、`use_deepspeed`、`use_cuda_kernel` 复用本仓现有默认/已有约定，不通过 HTTP 暴露。

## 输出目录约定

输出根目录：`/home/fanrui/agbox-paseo-shared/read-only/indextts2_service_voices`

目录结构（参考 comfyui_service `executor.go:196`）：

```
<output_root>/
  <YYYYMMDD>/
    <task_id>/
      0.wav
      0.srt
```

- 日期取 UTC 当天，格式 `YYYYMMDD`。
- `task_id` 用 uuid4，保证并发不撞目录。
- `0.wav` 与 `0.srt` 同名同目录，由 IndexTTS2 推理一次产出。
- 接口返回 `wav_path` / `srt_path` 是相对 `<output_root>` 的相对路径，不带前导 `/`。

## 行为约束

- 模型在服务启动时加载一次，常驻；不做按请求加载。
- 一次请求只生成一组 wav+srt，不支持批量。
- 合成完成前 HTTP 连接保持打开；不返回任务 ID 让客户端轮询。
- 失败返回非 2xx 状态码与错误信息字符串；不做重试。
- 不清理输出目录；保留供 filehub 服务读取，由人工或外部脚本管理生命周期。
- 单 GPU 串行：handler 内用进程级锁（`asyncio.Lock`）保护推理调用，同一时刻只允许一个请求进入 `engine.infer`，其余请求在锁上排队等待。不限制队列长度。

## 不做的事

- 不做异步任务、任务队列、callback。
- 不做鉴权、限流、并发控制。
- 不做多 voice / 多 speed / 多 speaker 切换；这些参数 v1 不开放。
- 不做服务化部署脚本（systemd unit）；第一版本人工 `uv run` 启动即可。
- 不做与 `tts-service` 的兼容、迁移、调用关系。
