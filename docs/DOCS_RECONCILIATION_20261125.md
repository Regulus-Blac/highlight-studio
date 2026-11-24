# 文档逐项更正记录（2026-11-25）

核对依据：当前`apps/`、`services/`、`scripts/`和`contracts/`，固定源码/模型清单，以及11月5日/6日GPU验收记录。当前事实集中在[状态页](STATUS_20261125.md)，本轮待办在[四人计划](FINAL_IMPROVEMENTS_PLAN_20261125.md)。本次只改文档/协作说明，不新增产品功能、训练、下载模型或升级子模块。

## 更正清单

| 文件 | 原过期/混淆内容 | 更正与证据 |
|---|---|---|
| README.md | 初始化默认要求私有子模块；课程/云端商业范围混杂 | 将普通CPU模拟与已有GPU真实入口分开，列实物依赖、现有能力及课程范围；不宣称一键模型安装 |
| AGENTS.md | A/B继续训练；以原云端/本机安装器范围为当前任务 | 指向本轮四人计划，A/B转产品；普通联调与完整子模块检查分开 |
| docs/development.md | 段落仍描述每日UTC 02:00自动同步、自动升级PR权限 | 明确11月24日工作流已移除；Product checks只读CI与手动升级分开；补可选子模块与WSL worktree边界 |
| docs/architecture.md | 首页用旧jobs；完整recipe/GPU/渲染未完成；发布清单未知 | 按实际v1工作台、SQLite/单worker、SSH中继、f1_setup/f1_stage、50项资产和GPU记录重写，商用扩展推迟 |
| docs/team.md | 从零建立网页/模型/导出的旧排期，AB仅模型工作 | 当前进度与ABDE四人具体工作分开；不虚填GitHub账号或发送邀请 |
| docs/baseline应用开发计划书.md | 预览/渲染未实现；首页旧jobs；需重建模型；JSONL仅双字段 | 对照server/worker/media/f1代码更正；JSONL为完整prediction v1；缩小课程范围，保留帧/框/音频语义 |
| services/model_adapter/README.md | “完整F1 runner、权重和GPU验收尚未完成” | 固定四阶段已实现，50项资产与历史GPU样片已验收；实物部署和完整发布审批仍需分开 |
| references/competition-f1/README.md | “不会用于运行”“没有新视频recipe” | f1_stage验证manifest后在独立runner使用固定源码，f1_setup生成候选recipe；129项绑定原文件不改 |
| apps/frontend/README.md | 仅本机、不能远程、云端完全未接入 | 本机服务与SSH连接器两条路径并列；只有未连接的通用云端入口仍禁用，本机模型下载安装器未实现 |
| docs/frontend-execution-modes.md | 改动只本地未推送；缺远端主机/runner；云端均禁用 | 对照app.js的model/ready/deployment条件和connect_demo明确本机/远程真实/模拟/不可用状态 |
| docs/integration.md | 缺GPU/完整runner、无渲染；要求E建设v2/分布式设施 | 新视频runner/渲染/SSH链路已有；当前复用v1/单机结构；早期readiness槽位与新候选资产检查分开 |
| docs/de-baseline-handoff.md | 仍按D/E单独补主流程、云端待接入安排任务 | 链接本轮ABDE分工，保留可复用的CPU/GPU启动步骤，不重复开发已存在功能 |
| docs/demo-runbook.md | 未明确现有命令与本轮拟建启动器的区别 | 标记当前命令可用、本轮工具待实现；保留GPU/路径预置条件，不声称空服务器自动部署 |
| docs/f1-video-inference.md | 同机课程使用也要求跨机共享存储；历史验收容易当实时状态 | 本轮同机runtime即可；跨机另行设计；历史硬件/耗时保留，本次未复验远端在线 |
| contracts/README.md | 将jobs称当前兼容页；媒体/v2状态容易混淆 | 当前页面用tasks；保留兼容API；Schema不变，完整prediction v1 JSONL与v2草案明确区分 |
| docs/requirements/README.md | “只有空Mock、不读视频、不生成成片”及旧基线当当前 | 更新当前网页/媒体/GPU/SSH事实、文档优先级和扩展边界 |
| docs/requirements/01-product-scope.md | 静态页、内存任务、无视频；500MiB/180秒建议值当现状、第二比例P1 | 更正SQLite/网页/成片和实际512MiB/10分钟等校验上限，双画幅已有样片；Project旧场景标扩展 |
| docs/requirements/02-functional-requirements.md | 全部Project、服务端编辑版本冲突/删除都算首版必做 | 保留FR编号，逐项列当前v1行为与待办；离线清理与复杂Project删除分开 |
| docs/requirements/03-system-models.md | 图示“排队不能取消”、retry_of像当前接口 | 增加当前v1实际状态图，明确queued预测可cancel、导出无cancel、当前无retry_of；旧图标v2历史概念 |
| docs/requirements/04-data-and-interfaces.md | Project/EditRevision/v2伪接口像已实现；无高光不能导出一概而论 | 改为实际Upload/Task/Plan/Export与v1路由；空JSONL可导出，空MP4拒绝；v2仅后续边界 |
| docs/requirements/05-quality-and-acceptance.md | “真实模型/媒体/性能未知”“全部AT未执行”、旧三周从零排期 | 已有CPU/GPU证据与未覆盖规模分开；收敛AT/FR范围，不把历史记录或计划当新验收通过 |
| docs/requirements/diagrams/README.md | 五张扩展图可能误读为当前数据库/接口 | 标明10月27日Project/v2概念；实际v1另见03开头，不改旧图/SVG冒充当前架构 |
| docs/ab-cpu-preparation.md | 10月27日缺项仍像“下一步”：补全资产/recipe/GPU；旧readiness结果当全盘缺失 | 保留历史检查结果与日期，逐处补后续完成事实；v1复用，不要求E包装v2 |
| docs/reviews/2026-10-27-ab-cpu-preparation.md | 历史“尚待完成”没有有效期 | 明确当时评审范围与后续完成事实；不篡改历史测试数量或声称当时已有GPU结果 |
| docs/需求分析文档-初步.md | CLI/离线/批量长视频、旧格式/取整/异常回退等假设 | 页首逐项更正表，原稿明确归档，不再当当前承诺；当前规范转至v1/状态页 |
| docs/软件设计文档-初步.md | 同上，以及旧API/布局与当前产品不一致 | 同样逐项校正并保留历史设计来源；没有对原稿伪API做业务实现承诺 |

## 新增的执行入口

- [STATUS_20261125.md](STATUS_20261125.md)：当前能力、运行条件、限制、事实来源。
- [FINAL_IMPROVEMENTS_PLAN_20261125.md](FINAL_IMPROVEMENTS_PLAN_20261125.md)：大学生项目P0/P1/范围外，ABDE可落实工作包、约4工作日示例、依赖与待执行验收清单。
- 本页：修订追溯，不表示新功能已完成。

未修改比赛子模块、`.gitmodules`、`model.lock.json`、129项绑定源码、任何权重或业务实现。图中尚未落地的扩展保留历史身份；未把新方案写成已通过验收。后续功能PR按分工逐项更新事实和真实验证记录。


## 本次验证

- 仓库引用检查11项通过：锁文件/分支/remote/commit、子模块内容、gitlink、暂存引用和上游历史一致。此工作树由Windows创建；WSL检查使用临时Git路径适配，并按Windows的CRLF检出规则核对，未修改子模块内容/指针、未关闭钩子或忽略真实改动。
- 固定F1 SOURCE_MANIFEST和129项绑定原文件验证通过，参考README不属于绑定文件。
- HTTP Mock冒烟通过；20项接口、前端资源、F1 runner边界与SSH二进制/Range回归通过。
- 本地Markdown路径与当前新增锚点检查通过，Git diff whitespace检查通过。
- 本次未启动GPU、未下载模型、未进行完整FFmpeg媒体回归或真实模型复验；历史GPU记录与本次检查分别表述，远端CI以推送后的Actions实际状态为准。
