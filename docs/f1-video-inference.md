# 新视频推理候选：A/B 与 E 交接

`f1-derived-video-v0` 把比赛 F1 的已锁定阶段用于新上传的视频。它从新视频计算全部特征、时间概率和裁剪框，没有比赛索引、缓存、父预测或训练步骤。它是产品候选版本；不能把比赛 F1 43.37 当作其成绩，也不能宣称逐字节复现历史提交。

| 阶段 | 输入与处理 | 产物 |
| --- | --- | --- |
| `extract_features` | FFprobe 校验 CFR 与原帧时钟；固定 TimeLens 和 processor 提取单视频特征 | `video.json`、`raw.jsonl`、`temporal/upload.npz` |
| `select_frames` | 原四组时间头及融合树；普通横/纵裁剪使用 base pair 与 YOLO11s AM，近等比例使用 full pair 和固定 shift3 | `selected.json` 原视频帧号 |
| `predict_crops` | 448/672 双路特征；x 使用 RetargetVid，y 使用 G1 的 4 fps 读出，近等比例 16:9 使用 F1 二维头；近等比例 9:16 尚无相应二维头，使用最大合法居中框 | `crops.json` |
| `assemble_result` | 验证选帧、裁剪边界和 v1 契约 | `result.json`，标记实际 engine 与 `mock: false` |

只接收 16:9 和 9:16，视频需满足现有 CPU/CFR 媒体规范。当前配方不执行历史 full92 规则，也不依赖历史检测缓存。失败或缺失概率显式报错，不能降级为全选或伪造成功。真实选帧为空是合法结果，不会导出 MP4。

## A/B：部署准备与候选验收

在 Linux 产品目录提供已验证的 Python 推理环境、FFmpeg/FFprobe 与资产。Python 需要 PyTorch/CUDA、transformers、accelerate、NumPy、Pillow、OpenCV headless、PyAV、decord 和 Ultralytics。运行时无需 Hub 凭据，模型文件都读本机。

比赛解码器要求 OpenCV wheel 内的 libswscale 提供 `sws_test_transfer`。在独立推理环境安装 Ultralytics 的其余依赖后，将 `opencv-python` 替换为 `opencv-python-headless==4.13.0.92`，配合 `av==18.1.0`。两个 OpenCV wheel 共用 `cv2` 路径，应只保留 headless 版。Ultralytics 的包元数据仍声明 `opencv-python`，`pip check` 会报告这项差异。用 `aic.data.reliable_video.decoder_metadata` 实测样片解码，再开始 GPU 验收。

- 源码使用[固定 F1 参考 PR #4](https://github.com/Regulus-Blac/highlight-studio/pull/4) 的 `references/competition-f1`，可单独从产品仓库获取。准备程序校验来源清单及全部 129 个原文件，不能拿任意比赛 checkout 替代。E/D 不需要比赛仓库权限。
- `manifests/f1-downloaded-assets.json` 保存 50 项资产 SHA，匹配清单中的 `models/...` 相对布局。已经分散保存的同盘资产可用 `--receipt <路径映射 JSON> --allowed-root <资产所在根目录>` 建立新的硬链接布局；使用 SHA 已核对的本机路径，不复制大权重。不要修改硬链接文件内容。
- 机器路径、GPU 输入、输出和日志留在服务器的独立目录。用不同的目录保存每次部署与验收，失败的运行留存供排查。

从产品代码根目录执行（路径按机器修改）：

```bash
python -m services.model_adapter.f1_setup \
  --source-dir /deployment/competition-f1 \
  --asset-root /deployment/f1-assets \
  --python /inference-env/bin/python \
  --output-dir /deployment/config-f1-v0
```

这一步核对全部资产，生成 `model-assets.json`、`recipe.json` 和 `adapter-config.json`；仍保持 `complete: false`，不会加载模型。独立 GPU 验收时，按 [v1 请求契约](../contracts/prediction-request.schema.json) 写 `request.json`，其中 `video_path` 是本次测试视频的绝对路径：

```json
{"api_version":"1","video_id":"smoke-portrait","video_path":"/deployment/samples/sample.mp4","target_ratio":[9,16]}
```

用该推理环境执行：

```bash
/inference-env/bin/python -m services.model_adapter.f1_runner --run --validate-candidate \
  --asset-root /deployment/f1-assets \
  --manifest /deployment/config-f1-v0/model-assets.json \
  --recipe /deployment/config-f1-v0/recipe.json \
  --request /deployment/samples/request.json \
  --result /deployment/runs/smoke-portrait/result.json
```

`--validate-candidate` 只接受固定的 F1 候选清单。输出目录必须是新运行目录，四个阶段各有日志。对至少一个有权使用的新视频分别验证 9:16 和 16:9，记录引擎、输入与结果 SHA、预测帧数、耗时、峰值显存。通过后再用产品后端验证预览、编辑与真实成片。没有 GPU 验收证据时，保留未就绪状态。

### RTX 4090 验收记录（2026-11-05）

产品代码 `42f9ef6`，F1 源码快照 `beefb57`，50 项模型资产逐项通过 SHA256 校验。环境为 Ubuntu 24.04、Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、transformers 4.57.6、accelerate 1.10.1、Ultralytics 8.3.222、FFmpeg 6.1.1，以及前述解码依赖。测试样片为 12 秒、640×360、25 fps、带音频。

| 画幅 | 真实预测帧数 | 独立推理耗时 | 未编辑成片 |
| --- | ---: | ---: | --- |
| 16:9 | 83 | 87 秒 | 608×342、3.32 秒、含音频 |
| 9:16 | 91 | 86 秒 | 198×352、3.64 秒、含音频 |

两个结果均返回 `engine=f1-derived-video-v0`、`mock=false`。在串行执行的两次独立测试中，`nvidia-smi` 按秒采样的显存峰值为 19,282 MiB。

另用隔离的 `command` 服务通过产品 HTTP API 重新上传样片，两种画幅均完成真实预测、片段调整、Range 预览及 JSONL、带音频 MP4 导出和下载。两次导出均核对文件 SHA256，MP4 的画幅、帧数和音频与编辑方案一致。样片、预测结果、成片、逐阶段日志和验收回执保存在验收主机的数据盘。

## E：接入已审核发布

E 使用现有 `CommandAdapter`，不需要手工拼装四阶段。A/B 审核并发布 `complete: true` 的清单后，在同一台能访问模型和媒体文件的机器上启动：

```bash
python -m apps.backend.server --adapter command \
  --adapter-config /deployment/config-f1-v0/adapter-config.json \
  --runtime /deployment/product-runtime
```

独立机器上的正式业务服务仍需 E 提供共享存储或任务传输。D/E 在发布前使用 `mock-media` 验证前后端；真实候选配置不会作为生产服务自动启用。GPU 主机文件与凭据无需发送给 D/E。
