# D/E 开发与 baseline 验收交接

更新：2026-11-25。现场演示已有[SSH连接器](demo-runbook.md)。当前事实见[状态页](STATUS_20261125.md)，ABDE按[最终分工计划](FINAL_IMPROVEMENTS_PLAN_20261125.md)共同收尾；A/B不再训练。

## 当前可用能力

| 模式 | 已验证结果 | 使用条件 |
| --- | --- | --- |
| CPU `mock-media` | 浏览器上传、模拟选帧、编辑、裁剪预览、带音频 MP4 和 JSONL 下载 | Python、FFmpeg/FFprobe，普通克隆产品仓库即可 |
| GPU `f1-derived-video-v0` | RTX 4090 上 16:9 和 9:16 都从新视频生成真实预测并导出；9:16 另通过 Edge 上传、预览和 MP4 下载 | 开启带 GPU 的实例，使用已配置的推理环境和固定资产 |

GPU 测试基于产品代码 `42f9ef6`、固定源码快照 `beefb57` 和 50 项校验资产。12 秒样片的独立推理耗时为 87 秒（16:9）和 86 秒（9:16），两次采样显存峰值为 19,282 MiB。详细条件和证据见 [F1 验收记录](f1-video-inference.md#rtx-4090-验收记录2026-11-05)。固定源码位于 [`references/competition-f1`](../references/competition-f1)，D/E 读取这部分只需产品仓库权限。

## 实例停机时照常开发

远程实例可能关机。关机时两个服务均不可访问。无卡开机后可启动 `8001` 的 CPU 模拟服务。`8002` 的真实 F1 服务需要有卡开机。日常开发从 `main` 各自切工作分支，优先在自己的电脑运行联调，避免依赖远程实例的开机时间。

在产品仓库根目录使用 Python 3.11+，先将 FFmpeg/FFprobe 6.1+ 放入 `PATH`。仅做 D/E 模拟开发时，`git clone` 不需要 `--recurse-submodules`。

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/create_demo_video.py
python -m apps.backend.server --port 8001 --runtime runtime/de-baseline --adapter mock-media --allow-mock-media
```

打开 <http://127.0.0.1:8001>，选择「本机模拟联调」，可用 `runtime/demo/source.mp4` 验证操作。模拟预测固定标记为 `mock=true`，可导出真实 MP4 文件用于检查流程。模型效果验收用真实 F1，或使用后续能根据视频内容选帧的 CPU 适配器。

有实例访问权限的队友在新机数据盘上的 `highlight-studio/f1/candidate-v0` 目录执行对应启动命令。服务监听实例回环地址，通过团队授权的 SSH 中转访问。

```bash
# 无卡开机后启动模拟服务
bash mock-service.sh start
bash mock-service.sh status

# 有卡开机后启动已验收的真实 F1 联调服务
bash sj-f1-service.sh start "$(cat gpu-runtime/de-service/acceptance-dir.txt)"
bash sj-f1-service.sh status
```

## ABDE本轮下一步

A负责媒体体验与集成验收，B负责启动/环境检查和既有GPU部署，D负责模式入口与编辑导出反馈，E负责单机任务恢复与安全离线清理。上述核心功能多数已存在，不从零重做；具体工作包、估时与验收见[最终分工计划](FINAL_IMPROVEMENTS_PLAN_20261125.md)。

当前真实路径复用已验收F1，不以训练、CPU替代模型、多用户公网、完整业务v2或本机模型下载安装器作为本轮任务。无GPU时诚实展示mock-media；GPU恢复后按记录有限复验，不能以模拟冒充真实效果。

## 最小 baseline 验收

以一个有权使用、含音频且有明显事件的短视频做现场测试。从计划交付的网页上传，自动分析产生至少一个由输入内容选出的片段，可预览并下载可播放的 MP4。核对任务和导出都成功、画幅正确、原音频保留，并记录 `engine` 和 `mock`。同时用一段不支持的素材验证页面能给出明确错误。

- **允许验收时启用 GPU**：用现有 F1 服务，在最终验收机器和素材上按上述步骤复跑。这条技术链路已经实测。
- **要求关机也能开发，且无卡时要验收真实高光**：本机开发使用 `mock-media`，再补齐上面的 CPU 适配器并以 `mock=false` 完成同一验收。CPU 模式不需要部署到这台可停机的 GPU 主机。
