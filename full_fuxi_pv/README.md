# 完整 FuXi 光伏 LoRA：当前代码入口

当前交付为 `models/full_fuxi_pv_available_data_v1/`，详见 [模型使用说明](../models/full_fuxi_pv_available_data_v1/README.md)。此前本文件的版本已保存到 [历史说明](../archive/2026-11-09/full_fuxi_pv/README_before_cleanup.md)。

## 常用入口

| 文件 | 用途 |
|---|---|
| `train_available_data.py` | 使用现有原站点数据及香港科大配对样本训练当前版本 |
| `predict_available_model.py` | 加载当前权重与原始完整 FuXi，预测下一 15 分钟光伏功率 |
| `verify_available_model.py` | 核对留出指标并重载当前模型执行真实完整图前向 |
| `model.py` | 原始完整计算图、内部 decoder29 Wq LoRA、功率头及时间特征 |
| `offload.py` | 低内存权重映射和反向激活暂存 |
| `contract.py`、`data.py` | 历史特征、缺失值、时间及全球天气维度校验 |
| `download_era5.py` | 按需下载 ERA5 开发样本，不在预测时自动下载天气 |
| `test_contract.py` | 输入、缺失处理和时间约束测试 |

当前模型输入为 16 条连续 15 分钟历史记录，每条包括温度、气压、湿度、GHI 和真实历史功率；完整网络还需要同期全球气象及站点位置。输出为下一 15 分钟有功功率，单位 MW。模型增量权重与原始 `fuxi-2.1/fuxi-2.1.pt2` 组合加载。

## 运行示例

```powershell
.\output\python312-runtime\python.exe full_fuxi_pv/predict_available_model.py --input models/full_fuxi_pv_available_data_v1/example_input.json --context models/full_fuxi_pv_available_data_v1/example_context.json --cache-directory output/pv_prediction_cache_new --output output/pv_prediction_result.json
```

命令从项目根目录运行，缓存目录需为新路径。例子是现有电站的历史预测；改变日期必须同时提供对应全球天气。未知 DG 站点不能通过改站名或容量来替换训练电站。

## 其他保留代码

- `train.py`、`prepare.py`、`predict.py`、`stations.json`：要求元数据已核实、各时期均有天气配对的严格流程；不是当前 available-data 权重的运行入口。
- `train_pv_pilot.py`、`train_weather_check.py`：早期小样本机制/监督检查，可供复现。
- `smoke.py`：完整图检查，同时提供当前训练需要的 `frozen_digest`，是有效依赖。
- `feature_schema.json`：通用输入设计；当前实际预处理参数以交付目录及 checkpoint 的 schema 为准。

历史试验输出已移动至 `archive/2026-11-09/output/`，清单保留原路径与哈希。机制检查脚本的默认输出路径仍可用于新试验；查看过去结果应从归档目录读取。

## 当前训练结果

原来的 8 套站点及 23 个香港科大来源站点共 854,424 条历史训练样本用于功率头预训练；462 条已有同期配对样本用于完整图 LoRA 更新。5 个验证站、5 个测试站的全部功率标签未进入训练。保存模型已重载并执行真实完整图预测。

当前测试只有一个全球天气初始化时段，LoRA 的额外误差改善极小，尚无生产可用或 DG 实站精度结论。数据、模型和图的解释见交付目录，不将旧 smoke、bridge 或单矩阵试验当作当前模型。
