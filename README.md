# Highlight Studio · 视频高光剪辑课程项目

更新：2026-11-26。ABDE共同维护大学生示例项目，A/B本阶段不再训练模型。当前使用固定F1新视频推理配方；重点完善启动、网页使用和演示，不建设商用多用户平台。

[当前能力/限制](docs/STATUS_20261125.md) · [ABDE最终改进分工计划](docs/FINAL_IMPROVEMENTS_PLAN_20261125.md) · [真实演示启动](docs/demo-runbook.md) · [文档更正清单](docs/DOCS_RECONCILIATION_20261125.md)

## 普通电脑启动：完整流程模拟

Python 3.11+，FFmpeg/FFprobe 6.1+在PATH中。普通克隆即可，无需私有比赛仓库权限、GPU或模型权重：

```sh
git clone https://github.com/Regulus-Blac/highlight-studio.git
cd highlight-studio
python -m pip install -r requirements-dev.txt
python -m apps.backend.server --port 8001 --runtime runtime/local-demo --adapter mock-media --allow-mock-media
```

打开http://127.0.0.1:8001，选择“体验演示模式”。页面可上传视频、选择16:9/9:16、查看任务、编辑片段、预览、导出原声MP4或JSONL。模拟选段为固定规则，页面保留演示标记，API持续标记mock=true，不能作为AI效果。

`python scripts/create_demo_video.py`可生成runtime/demo/source.mp4合成样片。前端使用原生HTML/CSS/JavaScript，不需要npm构建。仅运行`python -m apps.backend.server`默认是空Mock：可以检查接口，但没有高光成片。

## 真实F1演示：连接已有GPU部署

已保存服务器连接的电脑可双击 `Start-Demo.cmd`、运行 `Start-Demo.ps1` 或 `python scripts/start_demo.py`。配置位于 [config/demo.json](config/demo.json)，认证使用本机 OpenSSH 密钥，不需要每次填写地址。更换服务器时更新配置；该文件中的地址不保证长期在线。

其他部署可在平台启动已部署的GPU实例，再在本机项目目录运行：

```sh
python -m pip install -r requirements-demo.txt
python scripts/connect_demo.py --host <SSH主机> --user <SSH用户名> --ssh-port 2020 --instance 0 --remote-root /root/sj-fs/highlight-studio/f1/candidate-v0 --mode real
```

按提示认证，保持终端开启，浏览器打开http://127.0.0.1:18000。连接器会启动已有远程服务；主机参数/路径按部署修改，直接SSH主机省略--instance，缺系统Python时加--remote-python。它不自动开通GPU，也不负责空服务器的环境/权重安装。

真实f1-derived-video-v0已在4090/4090D完成新12秒样片的双画幅分析、网页编辑及原声MP4/JSONL验收。独立预测71～87秒、显存采样约19.2GB，仅适用于记录条件。详细证据、固定50项资产、环境和新机部署步骤见[F1说明](docs/f1-video-inference.md)。本次文档更新没有开启GPU或复验远端在线状态。

## 当前边界

- 适合单个受信任操作者/课堂演示；服务默认回环监听，没有通用公网入口、账号注册或多用户隔离。
- 单runtime一个worker，预测/导出串行，排队预测可取消；运行中取消尚未实现。
- SQLite保存任务/导出；浏览器保存最近任务与已应用编辑。没有服务端Project/EditRevision或业务v2。
- CFR/VFR 均可通过媒体准入；VFR 暂用平均帧率换算时间，音画同步与真实时间戳映射暂未处理。无旋转、音视频零起始、单视频流等检查保留；默认上限512 MiB、10分钟、4K像素、120fps/72000帧，上限不等于性能保证。
- 真正的本机F1需手动准备完整资产/CUDA环境；模型下载器、设备检测、缓存管理尚未交付，本轮推迟。
- 已增加保存连接的启动入口、可拖动播放指针、裁剪预览逐段暂停和失败后重新导入。统一模拟/真实模式启动器、维护清理等其余改进仍按分工推进。

## 开发结构

| 路径 | 职责 |
|---|---|
| apps/frontend | 上传、模式、任务、片段编辑、预览和下载 |
| apps/backend | HTTP API、上传与业务入口 |
| services/task_service | SQLite持久任务、单worker与恢复 |
| services/media | 媒体校验、片段/裁剪规则、MP4/JSONL导出 |
| services/model_adapter | 唯一模型接入点，资产校验和四阶段F1 runner |
| contracts | 已实现v1接口契约与示例 |
| references/competition-f1 | 129项固定F1源码/配置，新视频runner验证后使用 |
| vendor/competition | 完整比赛源码子模块，只读引用 |
| scripts、docs | 开发/演示工具、部署说明、当前状态与分工 |

## 初始化与验证

普通产品检查：

```sh
python scripts/smoke.py
python -m unittest discover -s tests -v
```

完整媒体回归需要FFmpeg/FFprobe、Pillow与jsonschema；CPU夹具不证明AI质量。需要完整比赛源码且有读取权限的成员另行执行：

```sh
git submodule update --init --recursive
python scripts/bootstrap.py
python scripts/check_repository.py
```

完整Product checks CI也需要比赛仓库只读密钥，普通页面/CPU联调不需要。开发流程、引用升级/回滚与边界见[开发说明](docs/development.md)。

## 固定模型与资料导航

2026-11-24已移除Sync competition reference定时同步。产品固定F1配方，不自动跟随sym/zyb，不自动吸收半决赛Z4等更新。未来有独立算法升级任务时才由A/B使用model_ref.py update/switch，连同子模块指针、.gitmodules、model.lock.json评审；本轮不升级模型。

权重、视频、特征、预测、成片、凭据不进入Git，实物在仓库外或忽略的runtime。Git只保存源码和非敏感版本清单；已有源码不等于已有运行环境与资产。比赛历史分数不作为新上传视频效果保证。

[baseline计划](docs/baseline应用开发计划书.md) · [架构](docs/architecture.md) · [接口联调](docs/integration.md) · [需求导航](docs/requirements/README.md) · [团队分工](docs/team.md)。历史初稿/评审保留为来源，当前事实以状态页、代码与验收记录为准；本轮P0范围以最终改进计划为准。

[本次改动技术说明](docs/changes/2026-11-26-video-preview-and-media-compatibility.md) · [操作系统实验 PR 提交记录](docs/experiments/2026-11-26-pull-request-submission.md)。

上游源码许可保留在vendor/competition/LICENSE；产品代码的开源许可证尚未指定。
