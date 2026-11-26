# 操作系统实验 PR 提交记录

状态：初稿。整理日期：2026-11-26。

本次以 Highlight Studio 的真实功能改动完成“创建分支、提交代码、推送远程、发起 Pull Request”的任务。技术内容和验证依据见 [视频预览与媒体导入改进技术说明](../changes/2026-11-26-video-preview-and-media-compatibility.md)。本记录不替代课程指定的实验报告格式，姓名、学号等个人信息由提交者按任务书填写。

## 1. 仓库与提交对象

| 项目 | 内容 |
|---|---|
| 仓库 | [Regulus-Blac/highlight-studio](https://github.com/Regulus-Blac/highlight-studio) |
| 目标分支 | `main` |
| 功能分支 | `codex/video-preview-and-media-compatibility` |
| 同步时主分支 | `e0e6cdb`，保留 11 月 25 日的范围、分工与文档更新 |
| 功能提交 | `4652eba`，`feat: improve video preview, media admission and saved demo startup` |
| 改动范围 | 播放指针与逐段暂停、导入兼容、预览布局、失败恢复、保存演示连接与技术文档 |

PR 链接和最终提交摘要在创建后补充到本页。PR 的创建、CI 通过、代码审查和合并是不同状态；本任务以创建可供评审的 PR 为交付，不自动合并。

## 2. 实际提交过程

1. 用 `git status --short` 核对工作区，将此前完成但尚未提交的相关改动逐项纳入本次提交，排除 runtime、视频、模型和凭据。
2. 执行 `git fetch origin --prune`，确认远程比原本工作区新增 3 个提交。
3. 创建 `codex/video-preview-and-media-compatibility` 分支，使用当前已连接 GitHub 账户对应的提交身份；Git 身份仅配置在本仓库。
4. 精确暂存本次修改并创建功能提交。仓库提交钩子执行引用检查；为该进程提供项目 Python 路径，没有跳过钩子。
5. 将功能分支 rebase 到 `origin/main`。处理 README、前端说明和两份需求说明的文档冲突，保留远程新范围，同时更新本次功能事实。
6. 重新运行基本检查、33 项定向 Python 回归及浏览器测试，补写技术说明和本实验记录。
7. 推送功能分支，以 `main` 为目标创建 PR，保留问题、改动、验证、限制和文档链接。

提交前原工作区没有配置作者身份，首次提交没有成功；提供仓库级身份后，提交钩子又发现系统 Python 快捷方式不可用。最终通过项目虚拟环境执行钩子并成功提交。这两次未成功尝试没有生成额外功能提交。

## 3. 可以展示的操作系统相关内容

本项目改动涉及用户态进程与线程：Python 创建 SSH 子进程，通过标准输入输出管道通信；多个 HTTP 处理线程以互斥锁共享中继；退出时回收子进程；视频播放通过浏览器事件与帧回调驱动。可结合技术说明中的结构图讲解这些机制与实际代码的对应关系。

本次提交不声称实现了内核调度、页表或文件系统算法；若课程还要求这些实验，需要另按任务书完成。

## 4. 提交材料与截图建议

| 材料 | 查看方式 | 证明内容 |
|---|---|---|
| PR 页面 | GitHub 的 Conversation、Commits、Files changed、Checks | 发起人、源/目标分支、提交、代码差异和实际检查状态 |
| 功能演示 | 裁剪预览时间轴 | 指针可拖动，每段结束暂停并保留末帧 |
| 技术文档 | 本 PR 的 `docs/changes/` | 问题原因、设计、接口变更、验证和限制 |
| 本地提交记录 | 下列只读命令 | 分支、提交和改动统计 |

```sh
git branch --show-current
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --stat
git status --short
```

截图由提交者在最终 PR 状态下取用，避免把“已创建 PR”截图写成“已审核合并”。不截取或上传 SSH 密钥、访问令牌及服务器环境中的秘密。

## 5. 验证摘要

本地引用检查、Smoke、JavaScript 语法、33 项定向 Python 测试、6 项媒体 Schema/片段规则测试，以及播放/兼容回调、失败恢复、横竖方形视频布局和主流程浏览器回归均通过。浏览器中的合成视频和 API 夹具只验证交互，不作为模型识别效果证据。

全量 CPU 媒体回归以 GitHub Actions 实际结果为准；本次没有执行模型训练、下载权重或重新跑 GPU 分析。既有 11 月 6 日媒体检查记录与本次复验分开列在技术说明中。
