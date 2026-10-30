# D/E 联调与 A/B 模型接入交接

更新：2026-10-25。已提供上传、SQLite 持久预测任务、单 worker、幂等提交、排队取消、模型固定及可配置命令适配器。完整 F1 runner、GPU 效果与性能验收、渲染成片和对公网发布仍未完成。

## D：不准备权重即可开发

需要产品源码与 Python 3.11+。无需初始化比赛子模块、安装 PyTorch、取得 HF 权限或连接 GPU 主机。运行：

```sh
python -m apps.backend.server
```

默认监听 `127.0.0.1:8000`，使用 mock-v1。根页面已提供本地剪辑工作台，按 [v1 契约](../contracts/README.md) 对接 uploads/tasks、预览和导出。启动与验证说明见 [前端工作台](../apps/frontend/README.md)。如果 D 的前端开发服务在另一端口，可显式允许其准确 origin：

```sh
python -m apps.backend.server --cors-origin http://localhost:5173
```

以自有视频跑上传和任务流程，默认仅生成明确标记的 Mock 预测：

```sh
python scripts/demo_task.py /path/to/owned-video.mp4 --ratio 9:16
```

浏览器上传示例（file 为文件选择框取得的 File 对象）：

```javascript
const base = 'http://127.0.0.1:8000';
async function api(path, options = {}) {
  const response = await fetch(base + path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error.message);
  return body;
}
const upload = await api('/api/v1/uploads?filename=' + encodeURIComponent(file.name), {
  method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file
});
let task = await api('/api/v1/tasks', {
  method: 'POST',
  headers: {'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID()},
  body: JSON.stringify({api_version: '1', upload_id: upload.upload_id, target_ratio: [9, 16]})
});
while (task.status === 'queued' || task.status === 'running') {
  // 用 task.progress.stage 更新界面；percent=null 时显示等待，不伪造百分比。
  await new Promise(resolve => setTimeout(resolve, 1000));
  task = await api('/api/v1/tasks/' + task.task_id);
}
if (task.status === 'failed') throw new Error(task.error.message);
// succeeded 时读取 task.result；cancelled 时显示取消。Mock 结果需明确标记。
```

同一次创建任务的网络重试复用同一个 Idempotency-Key。前端持有 task_id 即可在页面刷新后恢复查询。参考实现暂不提供任务列表；业务后端可将 task_id 与用户、视频记录关联。

## E：可以复用服务，也可以只依赖预测接口

本参考 HTTP 服务与 GPU worker 暂同进程运行，一次执行一个任务，适合本机和受控联调。E 可复用 `services/task_service/`，或者在自己的业务服务中按同样契约管理数据库、队列及存储，调用 `ModelAdapter.predict`。比赛路径和命令不能散落到业务代码。

```python
from services.model_adapter.command import create_adapter

adapter = create_adapter('command', 'runtime/adapter.json')
try:
    result = adapter.predict({
        'api_version': '1', 'video_id': 'your-business-video-id',
        'video_path': '/controlled/input/video.mp4', 'target_ratio': [9, 16],
    })
finally:
    adapter.close()
```

HTTP 对接则只需要任务服务地址，上传文件后创建任务并获取预测结果，不需要获得 SSH 或 HF 凭据。业务鉴权、用户隔离、视频渲染、访问受控的成片存储、分布式队列和独立 GPU 服务由 E 与 A/B 后续集成；当前参考服务未实现这些能力，应保持本机或受控开发网络使用。

为验证缺模型时的失败界面，可以运行：

```sh
python -m apps.backend.server --adapter command --port 8001 --runtime runtime/unavailable
python scripts/demo_task.py /path/to/owned-video.mp4 --base-url http://127.0.0.1:8001
```

该模式不会回退 Mock：ready 返回 503，任务失败为 model_unavailable，mock=false。

## A/B：接入真实 runner

命令适配器不会启动训练或下载模型。它在启动时校验完整资产声明、与 `model.lock.json` 一致的比赛提交，以及每个实际文件 SHA256；遇到 LFS 指针、缺文件或不一致时拒绝启动。历史清单 complete=false 不能用作推理发布。

1. 在独立 GPU 环境准备完整 F1 组件及其推理依赖，明确继承主干、时间头、空间头、二维头、处理器和后处理；补齐已有交接记录中的缺项。
2. 将 [模型清单模板](../services/model_adapter/model-assets.example.json) 复制到 runtime/model-assets.json。列出所有所需文件的相对路径及 SHA256，记录环境版本，经真实验证后声明 complete=true。
3. 将 [适配器配置模板](../services/model_adapter/adapter-config.example.json) 复制到 runtime/adapter.json，设置 GPU Python、runner、模型目录和超时。配置中的 manifest/asset_root 相对配置文件所在目录解析；runner 可执行文件和脚本建议使用绝对路径。
4. runner 读取 `{request}` 的 v1 JSON，执行真实推理，向 `{result}` 写完整 v1 JSON；engine 使用清单 release_id，mock=false。进程非零退出表示失败。没有结果文件、非法结果、超时和模型身份不符均导致任务失败。
5. runner 须解码视频并验证原视频帧号和裁剪边界；适配器只校验 v1 结构、有限坐标、递增帧号及请求/模型身份，不替代视频级语义验收。

启动已配置的服务：

```sh
python -m apps.backend.server --adapter command --adapter-config runtime/adapter.json
```

命令使用参数数组执行，不经过 shell；支持 `{request}`、`{result}`、`{asset_root}`、`{manifest}`。每次执行的请求、结果和 runner.log 位于 runtime/inference 的独立目录，均不入 Git。runner 应等待其计算完成后再退出，并管理自己启动的子进程资源。

当前缺少可用 GPU 主机、完整 F1 模型目录与实际 runner，本轮只验证了命令协议的 CPU 测试夹具，不能据此宣称真实 F1 已上线。D/E 可以先使用 Mock 服务完成接口开发。

## 现有 HF 资产的管理尝试

[历史清单](../services/model_adapter/manifests/hf-legacy-ybox3.json) 固定 `aicsym/aic_weights` 的 `f0cfb12f0b11255a4dac89cb463e63b5c53527da`，只收录本地已核验的两个 ybox3 小头及配置，不宣称与 F1 完整绑定。普通校验只读取文件，不反序列化权重、不下载特征。

```sh
python scripts/model_assets.py services/model_adapter/manifests/hf-legacy-ybox3.json --root /mnt/e/aic_weights_audit/verified_samples
```

Windows 环境把 root 改为 `E:/aic_weights_audit/verified_samples`。本地审计样本可通过 SHA256 检查；指针克隆 `E:/aic_weights` 会被识别为尚未下载。加 `--require-inference` 后历史清单会被拒绝，防止把两套旧小头误当作完整推理模型。

前端和业务开发不需要约 50 GB 的历史资源；其中大部分是训练特征。HF 读取凭据仅由资产准备或部署负责人管理，发布代码不保存 token。

## 验证与交付范围

```sh
python scripts/check_repository.py
python scripts/smoke.py
python -m unittest discover -s tests -v
```

CPU 检查覆盖兼容接口、持久任务、幂等并发、模型不可用、版本固定、中断恢复、排队取消、错误结果和命令超时。它不证明 F1 分数、GPU 可用性或可渲染成片。测试夹具的 runner 只验证文件协议，不是实际模型。


## 2026-10-27 CPU 预览/导出交接

媒体核心与稳定联调路由已完成，具体启动命令、请求/响应与边界见 [A/B 无 GPU 前期开发与交接](ab-cpu-preparation.md)。原任务/预测 v1 与旧 Mock jobs 保持兼容；文中此前“无渲染”的记录由本节补充更新。

D/E 安装 FFmpeg/FFprobe 6.1+ 与 `requirements-dev.txt` 后，可以运行 `python scripts/create_demo_video.py`，再以 `--adapter mock-media --allow-mock-media` 启动参考服务，完成上传、预测、预览、JSONL/MP4 导出和下载。合成结果始终 `mock=true`；无需 HF、SSH、GPU 或 torch。

E 接入正式后端时复用 `services/media/` 规则与渲染函数，再建立项目、用户归属与不可变 EditRevision；需求中的业务 v2 仍待实现。A/B 用 `python -m services.model_adapter.f1_runner --check --asset-root <目录>` 查看全量缺口，不将旧 HF 两个小头的验证结果当作 F1 已就绪。
