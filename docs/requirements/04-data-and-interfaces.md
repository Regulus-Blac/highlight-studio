# 04 数据、媒体规则与已实现v1接口

更新：2026-11-25。[需求导航](README.md) · [Schema与完整路由](../../contracts/README.md)。本轮沿用当前v1与SQLite，不实现旧草案的Project/EditRevision/v2业务域。

## 1. 当前数据对象

| 对象 | 实际位置/语义 |
|---|---|
| Upload | 服务生成32位小写十六进制upload_id，保存原文件、SHA、文件名/大小/时间 |
| Prediction Task | SQLite中保存请求、状态、结果/错误、固定适配器身份与幂等键；video_id=upload_id |
| Prediction Result | v1原帧号与[x,y,w]，含api_version、video_id、target_ratio、engine、mock、predictions |
| Plan | 服务端由固定预测与segments算出的预览/导出方案，含rules_version/plan_id；不覆盖原预测 |
| Export | SQLite持久记录、不可变source/result/plan快照，成功后artifact含下载URL、SHA、大小、尺寸与时长 |
| 浏览器恢复记录 | 当前浏览器最近20个任务摘要/已应用编辑/导出ID；不是服务端Project/EditRevision |

上传成功只确认字节保存，不能等同于已通过媒体解码。浏览器提交upload_id，服务端生成video_path；不能提交任意服务器路径。实际文件/模型实物不入Git。

## 2. 核心业务规则

### BR-01 源帧与坐标

frame从0开始，基于上传原视频显示帧；框为源像素[x,y,w]，h=w×目标高/目标宽。当前只接受CFR、零旋转/起始、方形像素与单视频流。VFR/旋转规范化需另定源身份、时钟和坐标映射，本轮推迟。

### BR-02 校验与真实状态

帧号唯一递增、数值有限、框和帧在视频范围内，标识/比例/engine/mock与请求和发布匹配，源SHA不变。非法结果失败，合法空结果成功但无高光。默认禁止Mock导出，仅显式开发allow-mock-media允许带mock=true的合成成片。

### BR-03 连续帧到片段

consecutive-inward-v1只合并连续选帧。30fps下30…59与90…119成为[30,60)、[90,120)，共2秒，不填空隙或丢单帧。用户输入秒数对齐最近帧边界，恰半帧向较大帧号取整，越界/重叠/空区间拒绝。

### BR-04 实际整数裁剪

合法浮点框向内取整，在框内取满足精确比例、偶数宽高的最大合法编码尺寸，不修补原本非法的预测。所有启用帧取公共输出尺寸，逐帧裁剪后缩小；预览使用相同实际框，不添加未确认平滑。

### BR-05 编辑扩展

每个suggestion_id最多一次，按源顺序非重叠；省略segments采用原建议，提供数组表示整份方案。扩展区间只继承该建议自己的首/末框，不借其他片段。当前不支持从空建议任意新增片段。

### BR-06 原声同步

视频/音频按同一源半开区间剪切、重置时间戳后拼接。有音轨保留原声，无音轨才输出无声。MP4使用H.264/AAC；当前不导出其他音轨或字幕。

### BR-07 编辑与导出快照

编辑不改变模型结果。当前已应用方案由浏览器保存供本浏览器恢复；导出将source/result/plan固定到SQLite快照，不读取变化中的草稿。没有服务端edit_revision/base_revision API，也没有跨浏览器版本冲突控制，本轮不新增。

### BR-08 幂等与恢复

同Idempotency-Key/同请求返回同任务，异请求冲突；失败重试用新key创建新任务，现有请求无retry_of字段。同预测最多一个活跃导出。单runtime一个worker串行执行；重启遗留running变失败，排队项继续处理，不自动续算。排队预测可cancel，运行中或导出取消尚无。

### BR-09 文件生命周期

通过受控ID读取文件；当前是单一受信任工作区，没有用户/项目授权域，不直接匿名公网发布。本轮拟新增的离线清理命令必须默认dry-run、活跃服务拒绝、确认后只处理指定runtime；尚未实现，不能宣称已有DELETE项目路由。

## 3. 当前HTTP边界

| 入口 | 当前行为 |
|---|---|
| POST uploads?filename=... | 原始字节+Content-Length；201返回Upload，非multipart |
| POST tasks | 严格api_version/upload_id/target_ratio；202，新任务幂等重放200 |
| GET tasks/{id}、/result、POST /cancel | 持久状态/成功预测/仅排队预测取消 |
| GET uploads/{id}/metadata | 媒体检查，不改变原上传响应 |
| GET/POST tasks/{id}/preview | 原方案或整份编辑方案的实际plan |
| POST exports，GET exports/{id}、/{id}/file | 独立持久导出、查询和成功下载/Range |
| GET tasks/{id}/source | 源视频Range播放 |
| GET model、/health、/ready | 模型/Mock、HTTP存活、worker/适配器就绪 |
| 可选GET deployment | 仅SSH连接器提供remote/mode/transport；普通服务404是正常边界 |

除health/ready外路径前缀/api/v1；表中model/deployment也使用该前缀。正式字段/状态/error以contracts Schema和代码为准；这里不是新增路由说明。running.percent=null，不伪造真实进度。

JSONL导出一行完整prediction v1，保留engine/mock与实际启用原帧/对齐框；它不等同于旧比赛双字段JSONL。无高光可下载空predictions JSONL，MP4无启用帧报no_enabled_segments。

## 4. 后续扩展边界

旧Project、SourceVideo、EditRevision、完整业务v2、版本冲突与项目级删除是10月27日概念设计，当前未实现且不列本轮P0。相关历史概念图保留供追溯；不能按其伪API调用当前服务。将来确有任务再定义兼容性/Schema与实现，不为了文档完整把课程项目扩大为商用系统。
