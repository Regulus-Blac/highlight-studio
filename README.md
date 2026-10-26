# Highlight Studio · 视频高光剪辑软件工程项目

ABDE 维护产品；ABC 在原比赛仓库训练、改进算法。本仓库通过 Git Submodule 引用完整比赛源码，当前跟踪分支及精确版本见 `model.lock.json`。

## 首次使用

需要 Git 和 Python 3.11+。默认联调仅用 Python 标准库，不安装比赛训练环境，也不需要 GPU。

```sh
git clone --recurse-submodules https://github.com/Regulus-Blac/highlight-studio.git
cd highlight-studio
python scripts/bootstrap.py
python scripts/check_repository.py
python scripts/smoke.py
python -m apps.backend.server
```

打开 http://127.0.0.1:8000 查看联调页。当前提供模拟任务创建、查询和结果接口；返回明确标记的模拟结果。视频上传、持久化任务队列、真实 GPU 模型接入和视频渲染属于后续产品开发，不代表已完成。

## 需求文档（初稿）

[需求说明书与阅读导航](docs/requirements/README.md)集中说明首版产品范围、用户场景、功能与交互、UML 图、数据与接口草案，以及需求对应的验收计划。文档区分现有能力和拟实现功能，尚待确定的限制与排期标为建议值。

## 固定开发结构

| 路径 | 职责 | 建议负责 |
|---|---|---|
| `apps/frontend/` | 上传、比例选择、任务进度、预览和下载页面 | D |
| `apps/backend/` | 文件、用户、任务、存储及业务 API | E |
| `contracts/` | 产品 v1 请求/响应契约；前后端共同依据 | ABDE |
| `services/model_adapter/` | 唯一模型接入点；隔离比赛实现差异 | A/B |
| `vendor/competition/` | 原比赛仓库完整源码子模块 | ABC 在原仓库维护 |
| `scripts/` | 初始化、模型引用更新/切换、验证 | A/B |
| `docs/` | 架构、分工、开发流程与验收计划 | ABDE |
| `.github/workflows/` | CPU 检查、每日模型更新 PR | A/B |

分工是建议，可在 `docs/team.md` 填入真实成员。大仓库保存产品代码及比赛源码引用；权重和数据不保存在 Git 中。克隆时需要 `--recurse-submodules`，GitHub 的普通源码 ZIP 不会包含子模块内容。

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

命令不会向比赛仓库提交或推送，也不会自动创建父仓库提交。切换到 `zyb` 后，自动同步会跟踪 `zyb`；可用同一命令切回 `sym`。分支接口不一致时由 A/B 修改适配层。切换不等于自动转换权重、训练环境或推理命令。

每天北京时间 10:00（GitHub 调度可能延迟）检查所选分支，有更新时创建/更新一个 PR，验证后由 A/B 合并。可在 Actions 手动运行 **Sync competition reference**。原比赛仓库是私有的，CI 已使用专用只读部署密钥；本地递归克隆需要成员自己的比赛仓库读取权限，见开发文档。

## 为什么采用这种方式

采用“子模块固定提交 + 适配层 + 自动更新 PR”。直接复制容易分叉；subtree 更新容易引入合并冲突；共享 worktree 不适合独立交付。独立模型服务/版本化包适合后续部署，但现在仍需要交付比赛源码，增加服务发布体系收益有限。

子模块本身可写，固定的是提交而非实时镜像。本项目通过本地禁止提交/推送、独立克隆和版本一致性检查防止误操作；真正的访问控制仍由 GitHub 仓库权限决定。A/B 在原仓库仍保有写权限，但产品开发只在本仓库进行。C 无需进入产品仓库。

详见 [架构决策](docs/architecture.md)、[开发与同步说明](docs/development.md)、[接口契约](contracts/README.md)、[模型接入](services/model_adapter/README.md) 和 [分工与验收](docs/team.md)。

Git 行为参考：[官方子模块文档](https://git-scm.com/docs/git-submodule)。上游许可证保留在 `vendor/competition/LICENSE`；产品代码的开源许可证尚未指定。
