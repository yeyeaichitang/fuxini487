# IWOA CEEMDAN VMD BiLSTM 训练结果

本模型用连续 16 个、间隔 15 分钟的温度、气压、湿度、辐照度及实测功率预测下一时刻光伏功率。训练使用现有 `site_1_b`，容量按源文件声明为 50 MW。数据按原时间顺序划分，训练、验证、测试分别为 14,169、3,501、8,849 条日间样本。

## 测试结果

| 方法 | MAE MW | RMSE MW | R² |
|---|---:|---:|---:|
| IWOA–CEEMDAN–VMD–BiLSTM | 1.549444 | 2.682969 | 0.956209 |
| CEEMDAN–VMD–BiLSTM | 1.603532 | 2.692347 | 0.955902 |
| IWOA–VMD–BiLSTM | 1.617769 | 2.702700 | 0.955562 |
| BiLSTM | 1.595845 | 2.710401 | 0.955308 |
| 持续性预测 | 2.210479 | 3.200810 | 0.937673 |

完整组合模型在本次测试中误差最低，相较基础 BiLSTM 的提升较小。全部测试预测保存在 `test_predictions.csv`，图中展示按时间排列的前 160 个测试样本，横轴省略夜间间隔。

## 可交付文件

- `model.pt` 保存完整的组合模型 BiLSTM 权重、分解配置与预处理参数。
- 另外三个 `.pt` 文件保存消融模型。
- `preprocessing.json` 与 `feature_schema.json` 记录字段、单位、标准化、时间窗口和筛选规则。
- `example_input.json` 与 `example_output.json` 来自实际保存权重的 CPU 推理。`example_reference.json` 单独保存对应真值。
- `metrics.json`、`validation.json` 记录指标及独立核对结果。
- 两份参数搜索日志、四份训练日志记录验证集搜索和模型选择过程。
- `comparison_curves.png` 与 `comparison_metrics.png` 为同一测试集的曲线和误差比较。

## 加载与预测

在项目根目录安装 [依赖](../../iwoa_pv/requirements.txt)，然后运行以下命令。

```powershell
python iwoa_pv/predict.py --input models/iwoa_ceemdan_vmd_bilstm_v1/example_input.json --output prediction.json
```

Python 代码入口为 `iwoa_pv/predict.py` 中的 `predict(request, checkpoint)`，返回下一 15 分钟的 MW 功率。完整模型可以在 CPU 上运行，分解所需 EMD-signal 与 vmdpy 也须安装。硬件字段和历史功率均须有实测值，当前模型适用于最近观测辐照度大于 20 W/m² 的预测起点。

已完成独立指标重算、测试样本与源表对齐、保存模型预测入口验证，以及未知站点、替换容量、缺失气压和不连续时间四种异常输入检查。CPU 示例与 CUDA 批量预测差约 0.0000067 MW。

源时间缺少已核实时区，示例保留原始时间。正式接口要求携带时区，接入时应先确认源时间含义并统一转换。目标 10 MW 电站、DG1～DG3与硬件联调需要对应站点数据和资料。

说明书没有给出 IWOA 改进公式，本次采用明确记录的反向初始化、二次收敛因子与精英扰动变体。CEEMDAN 噪声试验次数为 16。完整算法、搜索预算和数据防泄漏处理见 [训练说明](../../iwoa_pv/README.md)。
