# 产品 v1 契约

2026-11-25核对：上传、异步预测、媒体预览/导出及可选部署状态均已有v1契约；2026-11-27为可选部署描述增加连接状态与本机演示能力字段，业务请求不变。保留原预测请求/响应及旧Mock jobs兼容API，现工作台使用uploads/tasks。D/E 以本目录 Schema 为边界；比赛导入及命令转换仅位于 `services/model_adapter/`。启动方式及端到端例子见 [联调交接](../docs/integration.md)。

## 预测接口（E 与 A/B）

Python 入口为 `ModelAdapter.predict(request) -> result`；HTTP 对接可创建预测任务，再取任务结果。命令适配器的请求与结果文件也使用同一结构。

请求见 `prediction-request.schema.json`：`api_version="1"`、非空 `video_id`、服务端解析的受控 `video_path`、正整数宽高对 `target_ratio`。浏览器通过上传文件取得标识，不向正式任务接口提交服务器路径。

响应见 `prediction-result.schema.json`：`api_version`、`video_id`、`target_ratio`、`engine`、`mock`、`predictions`。每条预测为原视频帧号 `frame` 和源像素 `[x,y,w]`；高度按目标比例计算。帧号非负、唯一、递增，坐标有限，x/y 非负且宽度为正。真实 runner 还须验证视频可解码和坐标、帧号在原视频范围内。

结果的视频标识和比例必须与请求一致，engine/mock 必须与任务绑定的适配器一致。没有高光可以成功返回空数组；推理失败必须以错误结束，不能伪造成功空预测。Mock 返回 `engine="mock-v1"`、`mock=true` 和空预测，不读取视频、不生成成片。

## 上传与异步任务接口（D/E）

| 方法与路径 | 成功响应 | 契约 |
|---|---|---|
| `POST /api/v1/uploads?filename=<URL编码文件名>` | 201 | `upload-result.schema.json` |
| `POST /api/v1/tasks` | 新任务 202；幂等重放 200 | 请求 `task-request.schema.json`，响应 `task.schema.json` |
| `GET /api/v1/tasks/{task_id}` | 200 | `task.schema.json` |
| `GET /api/v1/tasks/{task_id}/result` | 200，仅成功任务 | `prediction-result.schema.json` |
| `POST /api/v1/tasks/{task_id}/cancel` | 200，仅排队或已取消任务 | `task.schema.json` |
| `GET /api/v1/model` | 200 | `model-status.schema.json`；显示是否 Mock、模型是否已配置 |
| `GET /health` | 200 | `status="ok"`、`engine`；仅表示 HTTP 服务存活 |
| `GET /ready` | 就绪 200，未就绪 503 | `ready`、`engine`、`mock`；检查 worker 与适配器可用状态 |

上传请求体为原始文件字节，推荐 `Content-Type: application/octet-stream`，必须提供 Content-Length；不是 multipart。默认上限 512 MiB，可由服务端启动参数修改。支持 mp4/mov/mkv/webm/avi 后缀；上传成功只确认字节已保存和 SHA256，真实视频解码与模型支持性由 runner 检查。

上传响应包含 `upload_id`、显示文件名、实际字节数、SHA256、UTC 创建时间。路径由服务端生成，不在响应中暴露。创建任务仅接受：

```json
{"api_version":"1","upload_id":"替换为上传返回的32位标识","target_ratio":[9,16]}
```

正式 Schema 的 upload_id 为 32 位小写十六进制。任务中的 video_id 等于 upload_id；业务数据库可自行关联用户的视频名称和业务 ID。可选请求头 `Idempotency-Key` 长度 1–128，同一 key 与相同请求返回原任务，参数不同时返回 409。失败后重试应使用新 key，旧 key 不会创建新推理。

任务状态为 `queued → running → succeeded/failed`，排队状态可转为 `cancelled`。首次创建返回 queued 快照；查询时任务可能已经结束，前端无需等待观察到 running。

| 状态 | progress.stage | progress.percent | result / error |
|---|---|---|---|
| queued | queued | 0 | 均为 null |
| running | predicting | null | 均为 null |
| succeeded | completed | 100 | result 为预测结果，error 为 null |
| failed | failed | null | result 为 null，error 含 code/message/retryable |
| cancelled | cancelled | null | 均为 null |

预测任务 succeeded 只表示预测完成，不代表视频已渲染。2026-10-27 新增独立预览和导出资源，见下文；已有 tasks 状态及返回结构保持兼容。推理中没有真实细粒度进度回报，因此 percent=null，D 应展示阶段和等待状态。

## 持久化与模型固定

参考实现使用 runtime 下的 SQLite 和文件目录，单 worker 顺序执行。排队任务、终态结果和幂等 key 会保留到重启后。启动 worker 时，把上次遗留的 running 任务标记为 failed/worker_interrupted，供调用方显式重试；不宣称可以从中断位置续算。每个 runtime 通过进程锁只允许一个 worker，避免误恢复其他运行中的任务。

任务绑定创建时的模型配置和资产清单指纹。升级后旧模型不可用的排队任务返回 model_version_unavailable，不自动改用新模型。部署时应由业务层暂停新提交并排空旧任务，再切换版本。可用模型不等于任意素材效果保证。已有F1短样片GPU证据见f1-video-inference.md；新的模型/环境仍须独立验收。

## 错误和兼容

HTTP 错误体统一为 `{"error":{"code":"...","message":"..."}}`。400 为非法请求，404 为未知文件/任务，408 为上传超时，409 为幂等冲突、结果未就绪或任务不可取消，413 为超出大小，415 为不支持的文件后缀，500 为服务内部错误。任务创建后的执行失败通过任务状态返回，查询 HTTP 仍为 200。

主要执行错误码：`model_unavailable`、`model_version_unavailable`、`runner_unavailable`、`inference_failed`、`inference_timeout`、`inference_interrupted`、`invalid_prediction`、`worker_interrupted`。retryable=true 表示同一模型和输入可考虑重试，不表示服务会自动重试。

旧 `POST /api/v1/jobs` 保留原 201/立即完成行为，`GET /api/v1/jobs/{job_id}` 保留进程内查询，仅用于Mock兼容API，现工作台不调用它；配置真实模式时旧创建入口返回410。新开发统一使用 tasks，不改写旧 jobs 的返回含义。不兼容改变新建 v2，模型变化由适配层吸收。


## CPU 媒体接口（2026-10-27）

新增 `media-metadata.schema.json`、`segment.schema.json`、`preview-request.schema.json`、`preview.schema.json`、`export-request.schema.json` 和 `export.schema.json`。逐字段语义、全部路由、错误码与 HTTP 示例以 [A/B CPU 开发交接](../docs/ab-cpu-preparation.md) 为准；`examples/` 的媒体响应为合成示例。

2026-11-06：按产品要求移除固定帧率准入限制。`validation_profile` 在保留 `cpu-cfr-v1` 的基础上新增 `cpu-vfr-v1`；后者使用容器平均帧率（缺失时使用解码帧数/时长）表达现有帧时钟，保留原文件和解码帧号。它不表示已实现逐帧真实时间戳映射或变帧率音画同步。

上传后可 GET `/api/v1/uploads/{id}/metadata` 检查媒体；预测成功后 GET `/api/v1/tasks/{id}/preview`，或 POST 同一路径预览完整编辑方案。POST `/api/v1/exports` 创建导出快照，GET `/api/v1/exports/{id}` 查询，成功后 GET `/api/v1/exports/{id}/file` 下载。预览的 source_url 与成片 URL 均支持单区间 Range。编辑不修改原预测；导出以 plan_id 绑定规则、帧、框与剪辑快照，artifact 另记录源 SHA 绑定后的实际文件摘要。

导出请求可选 `segments`；省略采用原建议，提供则使用整份方案，不是增量补丁。每段 `suggestion_id,start_frame,end_frame_exclusive,enabled`，严格连续默认规则为 `consecutive-inward-v1`。导出新建返回 202，同 Idempotency-Key/同请求返回 200，不同请求返回 409；同预测另有 queued/running 导出时返回 export_active。成功任务包含 artifact，失败包含 error，其余为 null。

单段独立导出复用上述 v1 请求：只提供目标片段且 `enabled=true`，`format=mp4`。多个已选片段仍按源视频顺序提交完整方案，未选片段 `enabled=false`；服务端在同一 MP4 中拼接启用范围、丢弃中间未选区间并剪接对应原声。导出请求的范围独立于页面上的其他预览请求，不修改模型预测；不增加新的路由或 Schema 字段。

JSONL 每行是完整 prediction v1 对象，以原帧号表达实际启用范围及对齐裁剪框。无高光可导出空 JSONL，MP4 无启用帧返回 422。Mock 默认禁止导出，显式开发选项才开放；对外页面仍须显示 engine/mock。

HTTP 错误体保持原 `error.code/message`，执行错误使用 `code/message/retryable`。CPU 媒体限制与未来正式业务 v2 的归属/编辑版本要求分开记录；v2仍是扩展草案，不能将这些v1路由称为v2已实现；大学生示例项目本轮继续使用v1，不要求实现整套v2。

## 保存配置的启动器：可选传输能力（2026-11-27）

`GET /api/v1/deployment` 沿用 `deployment.schema.json`，保留原必填的 `location/mode/transport`，新增可选 `connection`（`connecting/connected/unavailable`）和 `local_demo`（boolean）。旧连接器仍可只返回原三字段。`remote/real` 表示配置的远程目标，不能单凭该描述认定连接或模型就绪；前端同时检查 `/api/v1/model` 与 `/ready`。

仅保存配置的 `start_demo.py` 网关提供 `POST /api/v1/deployment`，请求体必须为 `{}`。它异步重连本机保存的目标并返回同一部署描述，不接受主机、路径或认证字段；已有可用连接直接复用。普通后端与旧连接器不必实现此控制路由。

该网关在同一回环地址上，以 `/demo/` 前缀将请求路由到独立本机模拟服务（例如 `/demo/api/v1/tasks`、`/demo/ready`），移除前缀后使用现有 v1 路由及完全相同的业务请求/响应。响应里的 `source_url` 和 artifact URL 仍是 v1 地址，前端根据任务的演示模式为原生媒体、Range 与下载请求加上前缀。这是本机网关的传输选路，不增加任务字段或模型能力。任务恢复时沿用记录的执行位置，不因远程连接状态改变选路。

网关 `/health` 检查本机页面服务；未加前缀的 `/ready` 检查远程处理。远程未连接时，未加前缀的 model 返回 `available=false,mock=false`，ready 返回 503，业务请求返回 `503 remote_unavailable`；不会自动交给本机 Mock。`/demo/` 下始终保留真实的模拟标记和自己的任务库。
