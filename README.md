# Highlight Studio · 视频高光剪辑软件工程项目

ABDE 维护产品；ABC 在原比赛仓库训练、改进算法。本仓库通过 Git Submodule 引用完整比赛源码，当前跟踪分支及精确版本见 `model.lock.json`。

## 首次使用

需要 Git 和 Python 3.11+。D/E 可按 [开发与 baseline 验收交接](docs/de-baseline-handoff.md) 普通克隆并在本机做 CPU 媒体联调。下面的完整初始化流程适用于需要检出比赛子模块且有读取权限的成员。

```sh
git clone --recurse-submodules https://github.com/Regulus-Blac/highlight-studio.git
cd highlight-studio
python scripts/bootstrap.py
python scripts/check_repository.py
python scripts/smoke.py
python -m pip install -r requirements-dev.txt
# 完整媒体测试还需在 PATH 中安装 FFmpeg/FFprobe 6.1+
python -m unittest discover -s tests -v
python -m apps.backend.server
```

当前提供网页工作台、视频上传、SQLite 持久任务、片段预览、JSONL/MP4 导出和 Range 播放。默认启动返回明确标记的 Mock 结果。完整模拟成片请用 `mock-media`。真实 F1 候选已在 RTX 4090 验证 16:9 和 9:16 的新视频推理与成片。预测和导出分别查询任务状态。[接口交接](docs/integration.md) · [F1 验收记录](docs/f1-video-inference.md)

```sh
python scripts/demo_task.py /path/to/owned-video.mp4 --ratio 9:16
```

该命令在另一个终端运行，连接已经启动的服务。产品联调只需要 Python 标准库；参考服务默认仅监听本机。

## 需求文档（初稿）

[需求说明书与阅读导航](docs/requirements/README.md)集中说明首版产品范围、用户场景、功能与交互、UML 图、数据与接口草案，以及需求对应的验收计划。文档区分现有能力和拟实现功能，尚待确定的限制与排期标为建议值。

## 固定开发结构

| 路径 | 职责 | 建议负责 |
|---|---|---|
| `apps/frontend/` | 上传、比例选择、任务进度、预览和下载页面 | D |
| `apps/backend/` | 文件、用户、任务、存储及业务 API | E |
| `contracts/` | 产品 v1 请求/响应契约；前后端共同依据 | ABDE |
| `services/model_adapter/` | 唯一模型接入点；隔离比赛实现差异 | A/B |
| `services/media/` | 元数据、片段/裁剪规则、CPU 渲染与参考导出队列 | A/B 提供核心，E 业务化 |
| `vendor/competition/` | 原比赛仓库完整源码子模块 | ABC 在原仓库维护 |
| `scripts/` | 初始化、模型引用更新/切换、验证 | A/B |
| `docs/` | 架构、分工、开发流程与验收计划 | ABDE |
| `.github/workflows/` | 产品 CI 检查 | A/B |

分工是建议，可在 `docs/team.md` 填入真实成员。产品仓库保存固定 F1 源码快照和完整比赛仓库的子模块引用。需要完整比赛源码时使用 `--recurse-submodules`，并确认有比赛仓库权限。权重和数据保存在仓库外。

## 后续发布方向

后续 v1.0 开发以 [baseline应用开发计划书](docs/baseline应用开发计划书.md) 为执行基线，详见 [开发说明](docs/development.md)。当前可用参考工作台验证上传、预测和导出。正式云端入口、用户与文件归属、业务存储和对外部署由 D/E 后续接入。前端依赖产品契约，模型升级由适配层与部署配置处理。

权重通过私有模型资产库（可选 Hugging Face）归档和分发，在服务器部署或更新时下载固定版本并校验，视频任务复用服务器本地模型。Git 只保存代码、源码引用及不含凭据的版本清单。用户明确要求本地处理时，按计划书单独交付本机服务与首次下载/导入、持久缓存的模型包，仍使用网页交互；该可选模式当前尚未实现。

详细职责、接口兼容、版本回退和验收方向见 [架构与发布设计](docs/architecture.md)。

## 更新与切换比赛分支

```sh
git switch -c chore/model-upgrade
python scripts/model_ref.py update
# 或切换到成果更好的分支：
python scripts/model_ref.py switch zyb
python scripts/check_repository.py
python scripts/smoke.py
git add .gitmodules model.lock.json vendor/competition
git commit -m "chore: update competition model reference"
git push -u origin HEAD
```

命令不会向比赛仓库提交或推送，也不会自动创建父仓库提交。可手动切换到 `zyb`，或用同一命令切回 `sym`。分支接口不一致时由 A/B 修改适配层。切换不等于自动转换权重、训练环境或推理命令。

2026-11-24 已停用并移除 **Sync competition reference** 自动同步工作流。产品使用仓库内 `references/competition-f1` 的固定源码快照，后续有算法升级需求时由 A/B 手动更新并验收。完整比赛子模块仍固定在 `model.lock.json` 指定的提交；产品 CI 与本地递归克隆需要比赛仓库读取权限，见开发文档。

## 为什么采用这种方式

源码管理采用固定 F1 源码快照、固定提交的比赛子模块及模型适配层，升级由 A/B 手动审核。产品运行采用独立 GPU worker 或模型服务，与源码组织分别管理；部署与模型发布体系按设计文档逐步实现。

子模块本身可写，固定的是提交而非实时镜像。本项目通过本地禁止提交/推送、独立克隆和版本一致性检查防止误操作；真正的访问控制仍由 GitHub 仓库权限决定。A/B 在原仓库仍保有写权限，但产品开发只在本仓库进行。C 无需进入产品仓库。

详见 [架构决策](docs/architecture.md)、[开发与同步说明](docs/development.md)、[接口契约](contracts/README.md)、[模型接入](services/model_adapter/README.md) 和 [分工与验收](docs/team.md)。

Git 行为参考：[官方子模块文档](https://git-scm.com/docs/git-submodule)。上游许可证保留在 `vendor/competition/LICENSE`；产品代码的开源许可证尚未指定。
