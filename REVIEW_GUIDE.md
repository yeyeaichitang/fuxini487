# 光伏预测模型审阅说明

本仓库用于审阅现有数据训练的 FuXi 光伏 LoRA 研究模型，包括训练与预测代码、微调权重、输入输出示例及前后对比结果。硬件特征已经进入模型计算，尚未接入实际采集设备。

## 建议查看顺序

1. [模型说明与结果范围](models/full_fuxi_pv_available_data_v1/README.md)
2. [微调前后预测曲线](models/full_fuxi_pv_available_data_v1/comparison_curves.png)
3. [微调前后误差对比](models/full_fuxi_pv_available_data_v1/comparison_metrics.png)
4. [测试指标](models/full_fuxi_pv_available_data_v1/metrics.json)与[权重重载验证](models/full_fuxi_pv_available_data_v1/validation.json)
5. [输入示例](models/full_fuxi_pv_available_data_v1/example_input.json)与[实际离线预测输出](models/full_fuxi_pv_available_data_v1/example_output.json)
6. [训练代码](full_fuxi_pv/train_available_data.py)与[预测代码](full_fuxi_pv/predict_available_model.py)

## 当前完成情况

- 已整理历史光伏数据，明确温度、气压、湿度、辐照度与历史实测功率五个输入字段。
- 使用连续 16 个、间隔 15 分钟的历史点，预测下一 15 分钟光伏有功功率，输出单位 MW。
- 已完成 12 轮完整 FuXi 计算图 LoRA 更新，保存微调权重并重载执行完整图预测。
- 已提供相同功率头下微调前后的对比图与 125 条留出测试样本的指标。

## 结果与限制

本次 LoRA 对测试误差的改善极小，持续性基线在这批样本上更好，不能声称微调已经明显提升精度。同期全球天气仅覆盖一个 ERA5 初始化时段，尚未完成跨季节、目标 DG1～DG3 实站或生产验证。

当前路线是 FuXi＋LoRA＋光伏功率头。现有技术说明书中的 IWOA–CEEMDAN–VMD–BiLSTM 路线需由团队统一修订，不能把两种实现的实验结果混用。

## 仓库包含的模型与外部依赖

`models/full_fuxi_pv_available_data_v1/model.pt` 是 LoRA 增量与光伏功率头，不包含冻结的 FuXi 主干。推理需要同时准备 `fuxi-2.1/fuxi-2.1.pt2`（约 3.7 GB）、对应时段的全球天气数据，以及 NVIDIA CUDA 环境。

原始主模型、大体积气象与原始光伏数据、本机 Python 环境、缓存及历史归档不随 Git 上传。克隆仓库后不能仅凭增量权重直接运行完整推理。原始 FuXi 的来源和说明保留在 [上游说明](fuxi-2.1/README.md)，应遵守上游许可及署名要求。

模型运行说明中的 `output/python312-runtime/python.exe` 是原开发机路径，不包含在仓库中；其他机器需自行准备兼容环境。示例 `example_context.json` 引用的全球天气文件也需单独准备。

第三周接入前还需确认目标站点参数、历史功率采集与缺失策略、同期天气供应、部署依赖及与后端的实际联调。
