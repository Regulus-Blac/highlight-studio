# 开发、同步与边界

更新：2026-11-25。本轮按照[最终改进分工计划](FINAL_IMPROVEMENTS_PLAN_20261125.md)进行大学生示例项目收尾；[baseline计划](baseline应用开发计划书.md)记录媒体与接口基线。A/B不再训练模型，四人均开发产品。当前能力见[状态页](STATUS_20261125.md)，不能把历史初稿或尚未执行的计划当作实现。

## 普通克隆即可开发

只做页面、上传/任务和CPU媒体联调，无需访问私有比赛仓库：

```sh
git clone https://github.com/Regulus-Blac/highlight-studio.git
cd highlight-studio
python -m pip install -r requirements-dev.txt
# 将FFmpeg/FFprobe 6.1+安装到PATH后：
python scripts/smoke.py
python -m apps.backend.server --port 8001 --runtime runtime/local-demo --adapter mock-media --allow-mock-media
```

网页为原生HTML/CSS/JS，无npm构建步骤。基础Mock只需Python标准库；完整媒体模拟还需要Pillow与FFmpeg，测试依赖见requirements-dev.txt。当前没有一键启动器，该项由B在本轮实现。

拉取代码后使用`git pull --ff-only`，从main创建`feat/<功能>`短分支，例如A的`feat/final-media-acceptance`、B的`feat/final-launcher`、D的`feat/final-workbench`、E的`feat/final-runtime-maintenance`。不维持长期AB/DE主线，不重新选择框架。

## 可选的完整比赛子模块

需要核对完整比赛源码并具备读取权限时，再执行：

```sh
git submodule update --init --recursive
python scripts/bootstrap.py
python scripts/check_repository.py
```

bootstrap对子模块配置本地禁止提交/push的防误操作钩子，不是远端权限；它不修改原比赛工作副本。父仓库固定gitlink、.gitmodules和model.lock.json。不要在子模块中直接git pull或编辑研究代码；没有权限不妨碍普通产品联调。完整子模块检查和CI仍需要比赛仓库读取权限，不能把这种权限当成普通用户使用前提。

Windows创建的worktree可能在.git记录Windows绝对路径，WSL直接调用Git会失败。优先在创建它的Windows环境使用，或为WSL另建普通clone；不要为阅读/文档更新重置工作树或改子模块指针。

## 固定源码与手动升级

**2026-11-24已移除Sync competition reference工作流；没有每日UTC 02:00自动检查或自动升级PR。** 当前产品使用references/competition-f1固定快照，完整比赛子模块保持锁定提交。

仅有明确算法升级任务时，由A/B使用`python scripts/model_ref.py update`或`switch <branch>`，审核子模块指针、.gitmodules、model.lock.json以及真实资产/环境兼容性，一起提交评审。本轮不执行这些命令。切换源码不等于切换权重或线上模型，不自动合并模型升级PR。

普通Product checks CI通过COMPETITION_READ_KEY只读部署密钥读取私有比赛子模块，随后做引用、CPU接口和媒体检查。密钥只在Runner临时目录使用，不分发给个人；不要把旧自动同步工作流的写PR权限说明混到当前CI。CODEOWNERS只指定审核人，不等于已经强制分支保护或分配目录写权限。

## 按改动验证

- 文档：检查事实、文件/相对链接、命令与当前代码一致；不为文档变更启动GPU或下载模型。
- 页面/接口：`python scripts/smoke.py`、相关现有unittest；JS改动增加`node --check apps/frontend/app.js`。
- 媒体：准备FFmpeg/FFprobe和requirements-dev.txt后运行相关媒体测试；完整回归为`python -m unittest discover -s tests -v`。
- 完整引用：有子模块权限的环境运行`python scripts/check_repository.py`。
- 真实推理：复用已验收部署，按[f1说明](f1-video-inference.md)和[演示手册](demo-runbook.md)有限复验；记录真实环境和结果，不以Mock代替。

本地检查与远端Actions状态分别记录，不因本地通过声称CI通过。功能PR包含问题、最终改动与真实验证，精确暂存自己的文件；权重、视频、特征、运行结果和凭据不进Git。

## 单机演示与后续边界

服务保持回环或受控演示访问，使用SQLite/单worker与本地文件；不新增分布式队列或商用账号体系。协议仍为contracts v1，业务v2是扩展草案。新增API先评审，不能为界面方便更改严格Schema。

新机器的GPU环境、权重和远端服务脚本需要单独部署，不会随Git自动出现；日常开发不运行训练、不自动下载模型。模型回滚需恢复匹配的代码、配置与资产；只改model.lock.json不能完成模型回滚。Git ZIP不包含子模块或模型；固定参考源码可直接阅读，但模型部署资源仍另行准备。
