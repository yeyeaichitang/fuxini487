# FuXi 光伏功率预测项目

给组长审阅可先看 [模型审阅说明](REVIEW_GUIDE.md)，其中整理了完成情况、对比结果和运行依赖。

当前模型将完整 FuXi 气象网络、内部 LoRA 与光伏功率头结合，接收温度、气压、湿度、辐照度及历史功率，预测下一 15 分钟光伏有功功率（MW）。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 当前模型、评估范围及预测方法 | [模型使用说明](models/full_fuxi_pv_available_data_v1/README.md) |
| 当前微调权重和功率头 | [model.pt](models/full_fuxi_pv_available_data_v1/model.pt) |
| 实际运行过的预测示例 | [输入](models/full_fuxi_pv_available_data_v1/example_input.json) / [输出](models/full_fuxi_pv_available_data_v1/example_output.json) |
| 微调前后对比 | [预测曲线](models/full_fuxi_pv_available_data_v1/comparison_curves.png) / [误差图](models/full_fuxi_pv_available_data_v1/comparison_metrics.png) |
| 训练与预测代码 | [full_fuxi_pv](full_fuxi_pv/README.md) |
| 数据整理入口 | [README_PV_DATA.md](README_PV_DATA.md) |
| 项目说明书和输入输出约定 | [输入输出约定](output/fuxi_full_retry_20261108/MODEL_INPUT_OUTPUT_SPEC.md)；同目录保留原始技术说明书 DOCX |
| 旧版本及试验记录 | [归档说明](archive/2026-11-09/README.md) |

## 运行一次光伏预测

在项目根目录用 PowerShell 执行：

```powershell
.\output\python312-runtime\python.exe full_fuxi_pv/predict_available_model.py --input models/full_fuxi_pv_available_data_v1/example_input.json --context models/full_fuxi_pv_available_data_v1/example_context.json --cache-directory output/pv_prediction_cache_new --output output/pv_prediction_result.json
```

需要 NVIDIA CUDA 环境；缓存目录使用新的路径。示例是真实历史时段的离线预测，不是对当前日期的实时预报。当前增量权重需要与 `fuxi-2.1/fuxi-2.1.pt2` 一起加载。

## 当前目录分工

| 目录 | 内容 |
|---|---|
| `full_fuxi_pv/` | 完整 FuXi、LoRA、光伏头的训练、预测和检查代码 |
| `fuxi-2.1/` | 原始 FuXi 主模型、标准化参数、原始示例及必要模块 |
| `models/full_fuxi_pv_available_data_v1/` | 当前交付：权重、指标、对比图和预测示例 |
| `scripts/` | 当前两套光伏数据的下载、清洗及校验脚本 |
| `data/raw/` | 原始光伏数据；香港科大公开数据单独分类 |
| `data/processed/pv_all_sites_v2/` | 原来的 8 套去重站点数据 |
| `data/processed/hkust_rooftop_pv/` | 香港科大清洗数据、样本和核查记录 |
| `data/weather/` | 已下载的全球气象数据与来源记录 |
| `output/` | 当前日志、说明书、核查证据，以及当前运行环境和天气解码依赖 |
| `archive/2026-11-09/` | 旧模型、旧脚本、旧数据、失败及早期试验，按原相对路径归档 |

`.venv/`、`output/python312-runtime/`、`output/weather_dependencies/` 和 `.vscode/` 保留原路径：当前独立 Python 会复用 `.venv` 中的依赖，请不要将这些目录当作旧输出删除。

## 结果与边界

已完成现有数据条件下的完整图 LoRA 训练、模型重载及下一 15 分钟光伏预测。测试站点从训练中完整留出；当前只有一个同期 ERA5 初始化时段，时区、校园代表点和气压仍存在已记录的假设。

本次微调对测试误差的改善极小，简单持续性预测在这批数据上更好。模型为研究交付版，尚未完成 DG1～DG3 实站、实时硬件或生产精度验证。接口封装、滚动优化和联调不属于本次数据与训练代码的交付范围。
