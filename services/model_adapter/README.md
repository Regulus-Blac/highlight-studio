# 模型接入边界（A/B）

更新：2026-11-25。真实F1新视频runner已实现并有4090/4090D双画幅短样片验收，见[部署与验收记录](../../docs/f1-video-inference.md)。仓库包含代码和固定清单，实际模型、环境与已审核部署配置仍需独立准备，克隆源码不会自动具备真实推理能力。

## 已实现适配器

| 模式 | 实现 | 行为 |
|---|---|---|
| mock | MockAdapter | 空预测，mock=true，无AI分析 |
| mock-media | SyntheticMediaAdapter | 固定合成选段，可用于媒体演示；导出需显式allow-mock-media |
| command无配置 | UnavailableAdapter | mock=false、available=false，明确模型不可用 |
| command有配置 | CommandAdapter | 校验真实发布清单与资产，调用独立runner，要求合法v1结果与真实身份 |

业务仅依赖ModelAdapter.predict；比赛源码导入、命令、路径与结果转换集中在本目录。原帧号、源坐标[x,y,w]、请求video_id/比例、engine/mock均检查；实际runner还校验源SHA、帧钟和视频边界。命令使用参数数组，非零退出、超时、缺失/非法结果均失败，不静默回退Mock或全选。

## 固定F1新视频链路

f1_setup.py核对固定129项参考源码与50项资产，生成候选recipe和adapter配置。f1_runner.py按extract_features → select_frames → predict_crops → assemble_result串行调用f1_stage.py；从本次上传视频真实提取特征，不使用比赛旧预测或测试缓存。

11月5日/6日已在24GB 4090/4090D验收12秒新样片的16:9、9:16推理、网页编辑、预览、原声MP4/JSONL与下载。此证据不保证任意视频质量，也不代表普通电脑CPU能运行F1。当前运行仍需要已部署的模型与环境；本次文档更新未复验远端在线。

- manifests/f1-downloaded-assets.json：完整候选50项文件SHA/大小，固定主干/processor版本。
- manifests/f1-expected-assets.json：10月27日旧槽位/缺项审计；默认readiness检查仍用于该旧布局，不能单凭其结果否定已验收的新候选。
- manifests/hf-legacy-ybox3.json：两个旧小头和配置样本，complete=false，不能作为完整F1。
- model-assets.example.json、adapter-config.example.json：文件协议模板，不是实物模型或已审核发布。

仓库候选complete=false用于明确验证；独立候选执行需显式--validate-candidate，正式CommandAdapter要求经审核的complete=true发布、比赛提交匹配锁文件、全部资产通过SHA校验且不是LFS指针。不能手改标志代替GPU验收。

## 开发与演示

无权重联调用CPU mock-media。真实演示复用[SSH连接器](../../docs/demo-runbook.md)，由演示者启动已有GPU部署；任务读取本地已准备资产，不自动下载权重或训练。请求、结果、逐阶段日志保存在受控runtime，不进Git。

产品本轮保持已有F1。A/B转入[四人改进计划](../../docs/FINAL_IMPROVEMENTS_PLAN_20261125.md)的媒体/部署工作，不继续训练或自动跟随sym/zyb变化。新算法若未来独立安排，应在原比赛仓库维护，再同时核对接口、源码、权重与环境，不在vendor/competition里修改。
