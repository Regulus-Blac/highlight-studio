# A/B 无 GPU 前期开发与交接（2026-10-27）

执行基线为 [baseline应用开发计划书](baseline应用开发计划书.md)。本次交付 M0/M1 的接口与 CPU 联调支持、M2 的资产/runner 准备、M3 的 CPU 渲染参考实现。真实模型端到端验收、正式网页、业务 v2、云端部署和 M4 发布仍待完成。合成视频和合成预测只证明产品处理链路，不证明 F1 效果。

## 已完成的七项工作

| 工作 | 实现与验收方式 |
|---|---|
| 预览/导出契约 | `contracts/` 增加媒体元数据、片段、预览、导出请求与状态 Schema；保留预测 v1、上传/tasks 和旧 Mock jobs 语义 |
| 帧到片段、裁剪规则 | `services/media/plan.py`；连续段、单帧、禁用/缩短/扩展、边界检查、精确比例与偶数尺寸；快照哈希绑定规则版本 |
| 视频元数据与语义检查 | FFprobe 解码时间戳检查 CFR、帧数和尺寸；校验源 SHA、原帧号与裁剪边界；明确拒绝旋转/VFR 等不支持输入 |
| CPU 渲染与联调 | FFmpeg + Pillow，逐帧裁剪/缩放、按源时间剪原声、MP4/JSONL、Range 播放；合成画面和分段音调验证内容与同步 |
| F1 入口准备 | 审查锁定源码的继承关系；`f1_runner` 提供离线检查及四阶段命令编排，缺失的算法 recipe 不作默认猜测 |
| F1 资产就绪检查 | 16 个已知权重、7 个原配置和 10 个待确认发布槽位；逐项报告缺失、指针、未知哈希、错误哈希与已验证状态 |
| 回归与交接 | CPU 自动化检查、无模型演示入口、CI 安装媒体依赖；本页与 [联调文档](integration.md) 说明 D/E 接入方法和后续前提 |

## 无 GPU 联调准备

D/E 需要 Python 3.11+。上传/空 Mock 任务仍仅依赖标准库；媒体预览/导出需要 FFmpeg/FFprobe **6.1+** 和 Pillow，运行完整测试还需要 jsonschema。无需 HF 权限、远程 GPU 或比赛训练依赖。

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
# 安装 FFmpeg 后，将 ffmpeg 与 ffprobe 加入 PATH。
# 也可分别设置 FFMPEG_BIN、FFPROBE_BIN 为可执行文件绝对路径。
python scripts/create_demo_video.py
python -m apps.backend.server --adapter mock-media --allow-mock-media --cors-origin http://localhost:5173
```

在另一终端上传测试视频并获取预测任务：

```sh
python scripts/demo_task.py runtime/demo/source.mp4 --ratio 16:9
```

默认 `mock` 保持空预测；`mock-media` 为明确的合成演示，固定选择首尾各四分之一帧并居中裁剪，返回 `engine=synthetic-media-v1,mock=true`。这不是比赛推理。默认拒绝所有 Mock 导出；只有本地联调显式指定 `--allow-mock-media` 后才允许。正式发布不启用该选项。

本次 WSL 没有系统安装权限，验证时将 Ubuntu 官方 FFmpeg 包解压到被忽略的 `runtime/media-tools/`，依赖环境放在 `runtime/cpu-venv/`；这些本机工具不提交 Git。团队按自己的系统安装同类工具即可。

## 给 D 的媒体 API

| 方法与路径 | 用途/响应 |
|---|---|
| `GET /api/v1/uploads/{upload_id}/metadata` | 上传后的独立解码检查；`media-metadata.schema.json`；不改变已有上传响应 |
| `GET /api/v1/tasks/{task_id}/preview` | 成功预测的原始建议、片段、逐帧实际整数裁剪框、输出尺寸和源视频 URL；`preview.schema.json` |
| `POST /api/v1/tasks/{task_id}/preview` | 请求 `{"api_version":"1","segments":[...]}`，预览指定编辑；不修改原预测或保存业务编辑版本 |
| `GET /api/v1/tasks/{task_id}/source` | 受控源文件，支持单段 Byte Range；须由 E 的正式服务增加归属校验 |
| `POST /api/v1/exports` | 按 `export-request.schema.json` 创建不可变导出快照；新任务 202、同幂等键重放 200 |
| `GET /api/v1/exports/{export_id}` | `export.schema.json`；queued/running/succeeded/failed；失败查看 error |
| `GET /api/v1/exports/{export_id}/file` | 成功后读取 MP4 或 JSONL，支持 200/206/416；排队、执行或失败返回 409 |

```json
{
  "api_version": "1",
  "task_id": "替换为预测任务的32位标识",
  "format": "mp4",
  "segments": [
    {"suggestion_id":"0","start_frame":0,"end_frame_exclusive":25,"enabled":true}
  ]
}
```

`segments` 可省略，表示采用全部原始建议。提供时代表完整剪辑快照；遗漏或 disabled 的建议不输出。每个建议最多引用一次，按源顺序提供互不重叠的半开区间，边界必须在原视频内。空方案可导出空预测 JSONL，MP4 返回 422/no_enabled_segments。

预览返回 `source_sha256`、`rules_version` 和 `plan_id`。D 使用源坐标绘制框：先按播放器实际画面尺寸等比缩放，再加画面留白偏移；原视频帧位置按 FPS 映射，不以播放器容器尺寸作为坐标系。逐帧 `crop=[x,y,width,height]` 是渲染实际使用的整数框。开启编辑时，POST 预览与 POST 导出提交相同 segments；导出返回的 plan_id 可用于核对。

JSONL 是一行完整的 prediction v1 对象，包含启用范围的原帧号和实际对齐后的 `[x,y,w]`，不是 MP4 成片，也不替换保存的原模型结果。engine/mock 保留。MP4 的清单记录在导出任务的 artifact 字段，包含 URL、SHA、字节数、输出尺寸和时长。

主要媒体错误：422 `invalid_media/unsupported_media/unsupported_media_profile/invalid_prediction/invalid_segments/crop_too_small/no_enabled_segments/source_changed`；409 `mock_export_disabled/export_active/idempotency_conflict/artifact_not_available`；未知资源 404。执行后的错误记录在任务中，查询仍为 200。Range 仅支持一个字节区间，多段或非法区间返回 416。

## 给 E 的复用边界

参考服务在同一 worker 中轮流领取预测与导出任务，SQLite 原子领取、持久化快照和幂等键；重启将遗留 running 标为失败，不自动重跑。导出期间固定源 SHA、模型结果和剪辑方案，生成临时文件并检查后原子发布；任务终态不被迟到结果覆盖。同预测最多一个活跃导出。

E 可直接复用 `probe()`、`make_plan()`、`render_mp4()` 与 `render_jsonl()`，在业务后端建立 Project、AnalysisResult、EditRevision 和鉴权存储。现有 `services/task_service`/`services/media/service.py` 是单机参考持久化，不替代业务域模型。

[需求 v2 草案](requirements/04-data-and-interfaces.md) 的项目、归属、编辑版本、删除生命周期和正式 v2 路由**尚未实现**。本次新增的是 v1 联调端点，不把草案标为已上线；E 后续用业务 v2 包装这些核心函数，D 保持 v1/v2 页面接入的明确区分。真实产品不直接公开当前无鉴权的参考服务。

## 固定媒体规则与限制

规则版本 `consecutive-inward-v1`：默认前后缓冲均为 0，最短片段为 1 帧，仅合并严格连续选帧；不填空隙，不删短段，不自动平滑。编辑扩展到建议外时只继承该建议自己的首尾框。纯函数 `interpolate_box()` 供未来稀疏模型适配测试使用，当前密集 v1 结果不调用它，不用插值增加选帧。

合法浮点框先向内取整数坐标，在其中取满足精确目标比例且宽高为偶数的最大尺寸；不修复原本非法的框。所有启用帧取最大公共输出尺寸，每帧先裁剪后缩小至该尺寸。Pillow Lanczos 只进行尺寸缩放，不加入轨迹平滑。视频帧与音频按同一源半开时间区间裁剪、重置时间戳并拼接；无音轨才输出无声。

CPU 参考输入范围为：单视频流、方形像素、零旋转、视频/音频从 0 开始、实际解码为 CFR、最长 10 分钟、最多 4K 像素/120 fps/72000 帧，上传默认最多 512 MiB。VFR 与带旋转输入显式拒绝；后续规范化需另定源身份与坐标/时间映射。默认保留首条音轨，其他音轨/字幕不导出。MP4 使用 libx264/yuv420p 与 AAC。

这是 CPU 参考实现：顺序解码、逐帧处理、内存只持有少量原图；长视频的执行时间、批量预览数据量、取消/超时与部署吞吐仍需产品资源实测。渲染有 900 秒上限，worker 关闭会终止正在执行的渲染；排队导出暂不提供取消接口。业务前端应按原视频与 overlays 预览，避免每次操作都重新渲染。

## F1 审计与就绪条件

锁定比赛提交仍为 `4cfe92894953b006752cd539ec4815945212c0f6`，包含 10 月 23 日 F1；本次没有改子模块、升级模型引用、训练或下载模型。

- `tools/run_final.py` 是较早的 YOLO 路线，不能作为最高分 F1 的真实入口。
- `research_crop_scale_apply.py` 固定检查 X1 源预测 SHA，读取预先准备的 rows/features，保留所有历史选帧，仅替换 full16:9 框；它是历史包生成/重放链的一部分。
- 上游 F1 文档明确尚未证明整套原视频起步的全神经流水线；167 条新框和其余继承行的验证不能等同于任意新视频推理。
- 上游补充材料的 `missing_required_weights=[]` 只表示补充包完整。完整 F1 声明参数为 8,926,602,839，仍需父级全部主干、时间头、检测器、固定配置和环境。

`manifests/f1-expected-assets.json` 保存已知 16 权重、7 配置的原 SHA/字节数；主干、父级全量账本和 recipe 等 10 个路径是拟定发布槽位，未知 SHA 明确为 null，`complete=false`。它覆盖需求与缺口，尚不是逐文件齐全的完整发布清单；取得父级原始账本后继续展开所有分片/继承组件，不能自行凑齐哈希或用近似模型代替。

```sh
# 全量检查，不下载、不导入 torch；未就绪退出码为 2，并不是脚本异常。
python -m services.model_adapter.f1_runner --check --asset-root /path/to/f1-assets --report runtime/f1-readiness.json
# 在锁定源码元数据变化后，由 A/B 审核并重新生成已知清单：
python scripts/build_f1_inventory.py
```

2026-10-27 本机对现有 `E:\aic_weights` 按上述发布布局检查，33 个 F1 槽位均缺失，ready=false。该结论只适用于固定布局，不等于扫描所有磁盘断言没有模型。既有 HF 仓库是更早的资产/特征归档，不能用它的旧 ybox3 小头充当 F1。之前 `E:\aic_weights_audit` 留存的审计记录与部分样本不纳入新发布包，也不上传 Git。

实际 runner 按 `extract_features → select_frames → predict_crops → assemble_result` 调用显式命令数组，分别保存日志；只有完整发布 manifest（complete=true、提交匹配、全部 SHA 验证）才能进入执行。参数占位为 `{request}/{result}/{work}/{asset_root}/{manifest}`；末阶段输出原预测 v1，engine 必须等于发布版本且 mock=false。这个编排器不提供缺失的 F1 算法 recipe，CPU 测试只验证命令次序和接口。

下一步 A/B 需要：取得全部原始 F1 运行资产及冻结参数账本；复原对新视频的采样、时间集成、解码与裁剪配方；固定 GPU 环境；在自有/授权视频验证真实端到端结果与耗时。对外软件发布还需处理上游已记录的组件用途/分发许可限制，不能因为资产可下载就认为可公开再分发。HF 是可选分发源，不提供默认 GPU 算力；资产下载/缓存/回滚与可选用户本地模式按基线另行交付。

## 验证命令

```sh
python scripts/check_repository.py
python scripts/smoke.py
python -m unittest discover -s tests -v
```

测试使用自建合成画面和音调、临时目录及假模型文件，覆盖 Schema、旧 API、任务与导出幂等/恢复、框和帧边界、片段编辑、动态裁剪、音视频时间片、单帧/无音轨、旋转/VFR、Range、源 SHA、LFS 指针与 runner 编排。CI 安装媒体工具并运行同样检查，不将缺工具静默视为通过。

本次本机验证：仓库引用完整性、旧 Mock HTTP 冒烟、全部 32 个 CPU 回归用例通过；包括 16:9 与 9:16 MP4、分段音调对应和播放文件检查。未运行真实 F1/GPU/训练。测试使用的工具与产物保留在忽略目录或临时目录，不随提交分发。
