# 合成契约示例

JSON 文件按同名 `../<文件名>.schema.json` 校验。所有 ID、时间和 SHA 都是示例值；preview 的预测来自合成 fixture，engine/mock 明确标识，不表示真实 F1 已运行。

先从上传响应和任务响应取得真实 ID；预览 GET/POST 使用 prediction task ID，导出 POST 使用完整的可选 segments，导出 GET/file 使用 export ID。完整路由、错误与启动示例见 [A/B CPU 交接](../../docs/ab-cpu-preparation.md)。
