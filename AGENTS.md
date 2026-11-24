# 本仓库协作边界

- 用中文交流。产品由 ABDE 开发；A/B 本阶段不再训练模型，四人均投入产品改进。比赛算法为固定外部依赖，本轮不训练或升级。
- 当前事实先读 [状态页](docs/STATUS_20261125.md)，本轮按 [ABDE最终改进计划](docs/FINAL_IMPROVEMENTS_PLAN_20261125.md) 推进大学生示例项目：单个受信任操作者、现有网页与F1、简化启动和演示。不把商用公网、多租户、分布式设施或本机模型下载器列为本轮必交；媒体/接口规则沿用 baseline 计划。
- `vendor/competition/` 为外部比赛子模块。不要在其中改文件、提交或推送；需要的算法变更回原仓库处理。
- 前后端只依赖 `contracts/` 的 v1 接口。比赛导入和命令转换只允许放在 `services/model_adapter/`。
- 升级或切换模型引用使用 `python scripts/model_ref.py update` 或 `switch <branch>`，一起提交子模块指针、`.gitmodules` 和 `model.lock.json`。
- 不自动合并模型升级 PR。检查通过后由 A/B 确认算法与模型资产兼容。
- 权重、视频、特征、运行结果和凭据不入 Git；使用仓库外路径或被忽略的 `runtime/`。
- 普通产品联调无需比赛子模块：`python scripts/smoke.py`；媒体检查需 FFmpeg/FFprobe 和 requirements-dev.txt。完整子模块初始化后再运行 `python scripts/check_repository.py`；不要为无权限成员强制初始化私有源码。不要在本项目启动训练或自动下载模型。
