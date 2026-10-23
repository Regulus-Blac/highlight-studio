# 产品 v1 契约

业务后端通过 `ModelAdapter.predict(request)` 获取统一结果；比赛实现只能由适配层引用。Python 数据结构见 `services/model_adapter/adapter.py`，JSON Schema 见相邻文件。

请求：`api_version=1`、非空 `video_id`、非空 `video_path`（由业务端解析的受控文件路径）、`target_ratio`（正整数宽高对）。后续前端上传的是文件，不能直接向生产系统提交服务器任意路径。

响应：`api_version=1`、`video_id`、`target_ratio`、`engine`、`mock`、`predictions`。每条预测包含原视频帧号 `frame` 和源像素 `[x,y,w]` 裁剪框；高度由目标比例计算。不选中高光返回空列表，不能据此推断视频分析失败。

模拟联调 HTTP API：

| 方法与路径 | 响应 |
|---|---|
| `GET /health` | `status`、`engine` |
| `POST /api/v1/jobs` | 201；`job_id`、`status=succeeded`、`result` |
| `GET /api/v1/jobs/{job_id}` | 200；同一任务，进程内保存；未知任务 404 |

当前模拟适配器立即完成并返回空预测，不读取视频，不生成剪辑，不是推理质量演示。后续真实任务状态建议 `queued/running/succeeded/failed`，异步实现与存储由后端负责。

非法请求 400，未知路径 404，服务内部失败 500；错误体 `{"error":{"code":"...","message":"..."}}`。不兼容修改新建 v2；比赛分支变化由适配层吸收。
