# 现场演示：网页到远程 F1

使用 SSH 加密连接把远程回环服务映射为本机网页。适用于算家交互式服务器选择入口，不依赖网关支持 TCP 端口转发。浏览器无需安装 SSH 插件，也不会接触服务器密码。

## 准备一次

本机安装 Python 3.11+，拉取 main，运行：

```sh
python -m pip install -r requirements-demo.txt
```

远程保持已验收的 candidate-v0 布局：app、assets、.venv-f1、.venv-mock、mock-service.sh、sj-f1-service.sh，以及 gpu-runtime/de-service/acceptance-dir.txt 指向的真实产品验收目录。将当前版本的 app/scripts 和 app/apps/frontend 同步到部署目录；不要覆盖权重、模型配方或验收目录。现有服务每次请求读取网页文件，不必为页面更新中断任务。

## 每次演示

在平台选择有 GPU 的实例开机，然后在本机项目目录运行（用自己的 SSH 地址和用户名）：

```sh
python scripts/connect_demo.py --host <SSH主机> --user <SSH用户名> --ssh-port 2020 --instance 0 --remote-root /root/sj-fs/highlight-studio/f1/candidate-v0 --mode real
```

按提示输入密码。首次连接新主机需核对 SSH 指纹并确认；已有 known_hosts 中的主机密钥不匹配时拒绝连接。连接器不会保存密码或自动开通付费 GPU。保持该终端开启，浏览器打开 http://127.0.0.1:18000 。页面显示“远程服务已连接”“远程 F1”，上传视频后开始分析。建议使用已验证的12秒样片，分析约90秒，以实际任务状态为准。

无显卡时把 `--mode real` 改为 `--mode mock`，可以演示上传、任务、片段编辑、预览与下载，页面会明确标记模拟。真实模式缺少 GPU 时直接拒绝启动，不降级成模拟。

直接 SSH 登录的 GPU 主机不需要 `--instance`。若系统未安装 `/usr/bin/python3`，用 `--remote-python` 指定已有解释器，例如 `--remote-python /root/miniconda3/bin/python`。可通过 `--port 18002` 与原服务器的本机演示端口区分。远程启动脚本仍需按该机器的推理环境、FFmpeg 和已验收配置部署。

## 验证顺序

1. 页面确认引擎和远程状态，选一个有权使用的含音频 H.264 CFR MP4。
2. 分别选择9:16或16:9，上传并等待任务成功。
3. 预览裁剪，调整片段起止时间，点击应用修改。
4. 生成MP4并下载，检查画幅和音频；必要时另导出JSONL。
5. 刷新页面，从记录恢复任务，再下载已有成片。

素材限制沿用媒体规范：不超过512 MiB、10分钟、4K、120fps，固定帧率、无旋转及非零起始时钟。真实模型可能选帧为空，界面应保留空结果，而不是伪造高光。

## 故障与恢复

- `GPU_NOT_READY`：平台仍为无卡模式，切换GPU后重新连接。
- `CUDA_NOT_READY`：检查远程.venv-f1和驱动；不要用模拟结果替代真实结果。
- 启动超时：查看gpu-runtime/de-service/server.log。首次启动可能校验模型资产，连接器最多等待240秒。
- 连接中断：重新执行相同命令，保持相同本机端口，浏览器刷新或从任务记录恢复；已经完成的服务端结果不会因为连接器退出而删除。
- 上传中断：重新选择视频并上传；不要把未完成上传视为任务成功。

连接器仅绑定本机127.0.0.1，拒绝不匹配的Host及跨源请求，不是公开多用户网关。单条SSH连接按顺序转发请求，适合单人课程演示；没有增加并行GPU推理能力。平台关机或切换实例会中断SSH，必须重新运行连接命令。真实F1模型仍使用已验收配置，不重新下载模型、不启动训练、不改动比赛代码。

连接器提供可选的GET `/api/v1/deployment`，契约见`contracts/deployment.schema.json`。普通本机服务返回404时，前端继续沿用本机行为。模拟与真实远程任务在浏览器中分开存储，避免切换模式后恢复另一套服务的任务。SSH载荷采用分块Base64，避免交互式网关改写二进制视频；最终HTTP响应仍保留原文件字节和Range语义。
