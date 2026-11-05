# 模型接入边界（A/B）

`ModelAdapter` 定义固定的 `predict` 接口。当前提供 `MockAdapter`、无配置时显式报错的 `UnavailableAdapter` 和调用外部 GPU runner 的 `CommandAdapter`。产品可独立联调；命令协议已用 CPU 夹具验证，但完整 F1 runner、权重和 GPU 验收尚未完成，不能将其视为 F1 已上线。

真实接入时 A/B 通过本目录的命令适配器连接独立 runner，或实现遵循相同契约的适配器：

1. 从 `model.lock.json` 获取比赛版本；独立准备对应 GPU Python 环境与实际权重。
2. 将产品请求转换为所选分支的推理命令、索引或模型服务请求；研究脚本和路径只出现在本目录。
3. 将预测 JSONL 转换为 v1 响应，保留原帧号和 `[x,y,w]`，`mock=false`，`engine` 标识比赛提交与实际模型版本。
4. 校验返回字段、原视频边界及失败状态；不要把推理异常转换成成功的空预测。
5. 在自有公开视频上跑通，然后给业务后端注入适配器。生产环境应采用独立 GPU worker 或模型服务，限制并发；视频导出属于产品渲染模块。

比赛分支目前是研究工程，并无覆盖所有路线的统一生产 API。`sym` 与 `zyb` 可以自由切换源码引用，但不同的 checkpoint、依赖和推理参数需要同步确认；不保证两分支天然兼容。

模型资产模板见 `model-assets.example.json`，命令配置见 `adapter-config.example.json`。机器路径配置只保存在仓库外或 `runtime/`，不提交实际权重。版本清单的路径相对模型目录，每个文件单独记录 SHA256；命令适配器启动时要求 complete=true、比赛提交匹配锁文件，且全部文件校验通过，LFS 指针会被拒绝。

命令使用参数数组而非 shell。runner 读取 `{request}`，向 `{result}` 写 v1 预测；可用 `{asset_root}` 和 `{manifest}` 获取模型与清单。命令非零退出、超时、缺失或非法输出、engine/mock 不匹配均报明确错误，不回退 Mock。每次执行的输入、输出和 runner.log 保存在被忽略的运行目录。

`manifests/hf-legacy-ybox3.json` 是 `aicsym/aic_weights` 固定版本的两个旧 ybox3 小头及配置校验样本，complete=false；它不能作为完整 F1 发布。模型资产验证不下载权重、不执行 PyTorch 或训练代码。服务、直接 Python 调用和验证命令见 [联调交接](../../docs/integration.md)。

后续以 [架构与发布设计](../../docs/architecture.md) 为集成方向：第一版由独立 GPU worker 调用本适配层，模型在部署或更新时准备并固定版本，视频任务复用服务器本地模型。业务后端管理异步任务，渲染模块生成成片；前端继续只依赖产品契约。更换权重或模型实现需验证适配器兼容性，当前 Mock 检查不能代替真实推理验收。


2026-10-27：增加显式 `SyntheticMediaAdapter`（CLI `mock-media`，始终 mock=true）供 CPU 媒体联调；默认 Mock 空预测保持原行为。`manifests/f1-expected-assets.json` 与 `readiness.py` 覆盖已知 F1 补充材料及主干/父级账本/recipe 等缺口，完整逐文件 inventory 仍待取回。

`python -m services.model_adapter.f1_runner --check --asset-root <目录>` 不加载模型，未就绪退出 2。[单视频 F1 候选交接](../../docs/f1-video-inference.md) 说明如何生成候选 recipe、验证实际 GPU 和接入 E 的 worker；候选不默认启用。历史比赛结果不能作为上传视频的预测。
