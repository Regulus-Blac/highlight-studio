# 开发、同步与边界

## v1.0 开发基线

后续应用开发必须按照 [baseline应用开发计划书](baseline应用开发计划书.md) 的功能范围、M/L 阶段与验收标准推进：首先交付网页＋云端推理/渲染服务，用户明确要求时提供首次下载并缓存模型的可选本地模式。普通用户不以 CLI 或桌面客户端为主入口。

`需求分析文档-初步.md` 与 `软件设计文档-初步.md` 保留为需求/设计来源，其中与当前形态冲突的单机离线和 CLI 优先约束按计划书调整。实际请求/响应遵循 `contracts/`；新增功能先评审契约，不直接破坏已有 v1 接口。开发任务及 PR/交接说明应列出计划书中的阶段、需求、完成条件和验证证据；范围变化先更新计划书与关联文档。

## 日常开发

首次克隆后执行 `python scripts/bootstrap.py`。之后从 `main` 建立 `feat/<功能>` 分支，只改自己负责的产品目录，提交 PR。拉取别人合并的版本后执行：

```sh
git pull --ff-only
git submodule update --init --recursive
python scripts/bootstrap.py
```

普通拉取使用父仓库已经确认的固定版本。只有 A/B 或同步工作流执行 `model_ref.py update/switch` 来升级。不要在子模块里直接 `git pull`，也不要让 D/E 修改研究脚本。

D/E 无权重联调、上传与持久任务接口、命令适配器配置和资产校验见 [联调交接](integration.md)。新的客户端使用 uploads/tasks，旧 jobs 仅保留 Mock 兼容。完整 CPU 回归先安装 FFmpeg/FFprobe 6.1+ 与 `python -m pip install -r requirements-dev.txt`，再运行 `python -m unittest discover -s tests -v`；测试使用合成媒体，不下载模型，也不启动训练。CPU 媒体、F1 资产与 runner 交接见 [A/B 开发记录](ab-cpu-preparation.md)。

## 子模块防误操作

bootstrap 为本克隆的子模块配置独立 hooksPath，禁止子模块提交和 push，并设置不可用的 push URL。它不会修改原比赛工作副本的配置。重新克隆后必须重新 bootstrap；这些都是本地防误操作措施，可以被绕过，不是远端权限。

父仓库提交前检查子模块无未提交修改、锁文件与子模块 HEAD 一致、指针不是本地未发布提交。CI 同样检查。不要用忽略 dirty 的设置隐藏子模块改动。

GitHub `CODEOWNERS` 当前由仓库所有者作为模型引用临时审核人，A/B 的真实账号确定后更新。CODEOWNERS 只指定审核人；强制审核还需仓库套餐支持的分支保护/规则集。此初始化不邀请未知成员，也不改变原比赛仓库权限。

## 自动升级

工作流每天 UTC 02:00 检查一次所选分支，或手动触发。先跑与 CI 相同的 CPU 检查，再创建升级 PR；不会自动合并。分支切换后从 `.gitmodules` 读取新分支，无需改工作流。

已开启 Actions 创建 PR（Settings → Actions → General）。同步工作流仅获得父仓库内容写入及 PR 写入；比赛源码使用专用只读部署密钥，不持有比赛仓库写凭据。`GITHUB_TOKEN` 创建的 PR 通常不会再次触发其他工作流，所以同步任务在创建 PR 前已执行两项检查；仍须人工真实推理验收。

原比赛仓库目前是私有的，已添加名为 `highlight-studio CI read-only` 的只读 deploy key，其私钥加密保存在父仓库的 Actions Secret `COMPETITION_READ_KEY`。`scripts/ci_submodule.py` 仅将私钥临时写到 Runner 临时目录，以 HTTPS 获取 GitHub 已公布主机公钥并强制 SSH 主机校验，仅在子模块配置 HTTPS→SSH 读取映射。父仓库的默认 token 不具备跨仓库读取权限；没有复制个人 token 到 CI。

D/E 本地若没有比赛仓库读取权限，可先普通克隆父仓库，直接运行 mock 服务和 `smoke.py`，完整子模块初始化需由有权限的成员执行；完整源码交付与可独立递归克隆要求团队另行授予比赛仓库读取权限。此任务未邀请成员或改变成员写权限。部署密钥只供 CI 使用，不应分发给个人。

## 切换、回滚和离线交付

`python scripts/model_ref.py switch zyb` 会拉取远端 zyb、切换 detached HEAD、改 `.gitmodules` 分支并刷新锁文件。脚本拒绝覆盖子模块未提交改动。提交这三个路径后，其他成员按固定版本更新即可。

回滚已合并的升级时，对升级提交使用 `git revert <提交号>`，然后 `git submodule update --init --recursive`。这样同时恢复分支、锁文件和指针。不要只编辑锁文件。

GitHub 下载 ZIP 不含子模块。需要离线代码交付时，在已递归克隆的工作目录另行打包，排除 `.git`、`.env`、视频、权重和缓存，并核对上游许可证。部署 GPU 端时单独安装所选比赛版本的环境。
