# 需求图表（初稿）

[返回需求导航](../README.md) · [阅读图解](../03-system-models.md)

2026-11-25说明：这五张图保留10月27日的Project/v2扩展概念，当前v1并未实现完整项目域。本轮不要求建设它们；现有排队预测可取消、重试无retry_of，实际状态见03开头的新v1图。原图/SVG未改，不把扩展图误称当前架构。

| 图 | 类型 | 可编辑源文件 | 可阅读图像 |
|---|---|---|---|
| 用例与边界 | Mermaid 流程图形式的用例总览 | [01-use-cases.mmd](01-use-cases.mmd) | [SVG](01-use-cases.svg) |
| 用户主要流程 | 活动流程 | [02-workflow.mmd](02-workflow.mmd) | [SVG](02-workflow.svg) |
| 分析与导出交互 | UML 时序图 | [03-sequence.mmd](03-sequence.mmd) | [SVG](03-sequence.svg) |
| 任务生命周期 | UML 状态图 | [04-job-state.mmd](04-job-state.mmd) | [SVG](04-job-state.svg) |
| 核心数据关系 | UML 概念类图 | [05-domain.mmd](05-domain.mmd) | [SVG](05-domain.svg) |

SVG 已随文档保存，可以直接显示；`.mmd` 为唯一的图内容来源，文字要求以对应需求编号为准。重新生成图像后一起提交源文件和 SVG，避免图文不同步。

## 生成与检查

本初稿使用 Mermaid **11.12.0** 渲染，并采用 [统一配置](mermaid-config.json)。生成在本机完成，未把项目文档提交给外部在线绘图服务。临时工具和检查截图保存在被 Git 忽略的运行目录，不是产品依赖。

维护时可以在独立工具环境使用 Mermaid CLI，针对五份源文件分别运行如下命令形状：

```text
mmdc -i 01-use-cases.mmd -o 01-use-cases.svg -c mermaid-config.json -b white
```

应固定并记录所使用的 Mermaid/CLI 版本，配置支持中文的系统字体。该示例是维护方式，不代表仓库已经安装 CLI，也不是业务启动命令。

每次更新后检查：中文是否可见、标题/箭头是否遮挡、状态与需求是否一致、关系基数是否正确。SVG 在其他字体环境中的布局可能略有变化；可直接打开图像放大阅读。

官方语法参考：[时序图](https://mermaid.js.org/syntax/sequenceDiagram.html)、[状态图](https://mermaid.js.org/syntax/stateDiagram.html)、[类图](https://mermaid.js.org/syntax/classDiagram.html)。
