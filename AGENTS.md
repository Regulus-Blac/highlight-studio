# 本仓库协作边界

- 用中文交流。产品由 ABDE 开发；比赛训练和算法由 ABC 在原仓库维护。
- `vendor/competition/` 为外部比赛子模块。不要在其中改文件、提交或推送；需要的算法变更回原仓库处理。
- 前后端只依赖 `contracts/` 的 v1 接口。比赛导入和命令转换只允许放在 `services/model_adapter/`。
- 升级或切换模型引用使用 `python scripts/model_ref.py update` 或 `switch <branch>`，一起提交子模块指针、`.gitmodules` 和 `model.lock.json`。
- 不自动合并模型升级 PR。检查通过后由 A/B 确认算法与模型资产兼容。
- 权重、视频、特征、运行结果和凭据不入 Git；使用仓库外路径或被忽略的 `runtime/`。
- 初始化验证：`python scripts/check_repository.py` 与 `python scripts/smoke.py`。不要在本项目启动训练或自动下载模型。
