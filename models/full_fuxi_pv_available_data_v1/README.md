# 现有数据训练的完整 FuXi 光伏 LoRA 模型

本模型使用项目已有光伏历史数据及已有 ERA5 输入训练，没有新下载数据。它是完成训练和留出站点评估的研究模型；时区、校园代表点和气压差异仍有假设，未标记为生产可用。

## 模型与文件

| 文件 | 用途 |
|---|---|
| `model.pt` | 验证集选择的第 11 轮 LoRA 权重与光伏功率头、站点参数及训练元数据 |
| `baseline.pt` | 同一个已训练功率头、原始 FuXi 与零 LoRA，用于公平比较 |
| `feature_schema.json`、`preprocessing.json` | 特征顺序、容量、每站或共享的标准化参数 |
| `example_input.json`、`example_output.json` | 已实际运行的 16 点历史输入及下一 15 分钟功率输出 |
| `example_context.json` | 模型侧同期全球天气路径及站点位置，和外部历史输入分开保存 |
| `comparison_curves.png` | 同一留出测试集的真实功率、微调前和微调后预测曲线，单位 kW |
| `comparison_metrics.png` | 微调前后 MAE、RMSE，占站点容量百分比 |
| `heldout_predictions.csv` | 全部 125 条测试真值、前后预测、持续性预测和站点容量 |
| `metrics.json` | 全部评估指标和适用范围 |
| `training_log.json`、`training_config.json` | 训练损失、验证选择及配置 |
| `source_usage.json`、`paired_split_manifest.jsonl` | 数据在各阶段的用途与站点划分 |
| `validation.json` | 独立指标重算、站点隔离和保存权重重载检查结果 |

`model.pt` 采用标准的 LoRA 增量保存方式，文件较小；包含内部 A/B 矩阵和功率头，**不是单独一个投影矩阵模型，也没有在文件内重复存储冻结主干**。计算时必须同时使用项目原模型 `fuxi-2.1/fuxi-2.1.pt2`，通过 `FullFuXiPV` 实例加载增量。完整 FuXi 计算图实际参与前向与反向；LoRA 注入位置为 decoder 第 29 层 Wq，其他原始参数冻结。

四个环境特征为温度、气压、相对湿度、总水平辐照度，另有历史真实功率。输入历史为 16×5，预测下一 15 分钟功率。它们在历史数据中是真实气象观测，硬件尚未连接。

## 本次训练

1. 使用原来的 **8 套去重站点数据**中的 225,538 条训练样本，以及香港科大 **23 个训练来源站点**的 628,886 条训练样本，合计 **854,424 条**，训练功率头 3 轮。没有同期全球气象的样本仅训练历史特征分支，全球天气上下文分支置零作为消融；不伪造天气，不向完整 FuXi 发送错期全球输入。
2. 香港科大 5 个验证站、5 个测试站的所有功率记录均排除在功率头和 LoRA 训练之外。其天气标准化参数使用共享的、2021-12-01 之前的气象观测拟合，不使用它们的功率标签。
3. 利用现有一组 ERA5 初始化输入，按电站划分 462 条训练、125 条验证、125 条测试样本。对应 19 / 5 / 5 个站点。先在训练样本上校准功率头，再固定该头，执行 **12 轮完整图 LoRA 更新**。相同天气初始化和校园网格的样本共享一次完整 FuXi 前向，平均真实功率损失反向传播到内部 LoRA。
4. 只按验证集选择第 11 轮，最后一次性计算测试集对比。微调前后使用相同功率头、预处理和同一批测试样本，区别是是否加载学到的内部 LoRA。
5. 从原始 FuXi 与新功率头开始，未使用此前可能见过留出站点的试验权重初始化。原始冻结参数哈希保持一致。

## 实际结果

下表均为按站点额定容量归一化后的误差，使用同一批 125 条留出测试样本：

| 方法 | MAE，容量百分比 | RMSE，容量百分比 |
|---|---:|---:|
| 微调前：原始 FuXi＋已训练功率头 | 1.519558% | 2.209096% |
| 微调后：FuXi LoRA＋同一个功率头 | 1.519555% | 2.209082% |
| 持续性基线：下一点等于最近实测功率 | 0.884859% | 1.946949% |

**本次 LoRA 对测试误差的改善极小，不能视为明确精度提升；持续性预测在这批样本上更好。** 曲线图中前后预测几乎重合是实际结果，没有放大或修改预测差异。黑线为真实功率，蓝色虚线为微调前，橙线为微调后。图中仅展示实测功率峰值较高的 3 个测试站，全部测试站的预测保留在 CSV 中。

`metrics.json` 中的历史时序测试指标对应“功率头预训练结束、同期配对校准之前”的历史分支消融评估，不能当作最终模型的全局气象 LoRA 测试精度。正式的 LoRA 前后比较是 `before_lora`、`after_lora`，按 5 个完整留出站点评估。

## 限制与使用范围

- 全局天气只有 **一个 ERA5 初始化时段**。测试覆盖同一天凌晨至早晨，夜间样本较多；这是站点留出评估，不能声称完成跨天气、跨季节或全日独立预测验证。
- UTC+8 为有公开证据支持的时间假设；校园代表坐标不等于每个屋顶独立测量的 GPS。现有 SLP 与 ERA5 的约 12.6 hPa 差异未擅自订正。
- 原数据第 8 站使用文件声明的 30 MW；与其重复文件的 50 MW 标注冲突保留在元数据中，没有复制样本或虚构容量确认。
- “训练完成”表示已按现有数据训练协议完成优化、保存权重和评估，不表示设备接入、DG1/DG2/DG3 实站验证或生产部署完成。
- 本次交付属于数据处理和模型训练，不包含接口封装、滚动优化或联调。

## 复现与权重检查

### 实际进行一次光伏预测

已补齐可执行的模型侧预测脚本 `full_fuxi_pv/predict_available_model.py`。在项目根目录运行：

```powershell
.\output\python312-runtime\python.exe full_fuxi_pv/predict_available_model.py --input models/full_fuxi_pv_available_data_v1/example_input.json --context models/full_fuxi_pv_available_data_v1/example_context.json --cache-directory output/pv_prediction_cache_new --output output/pv_prediction_result.json
```

`--cache-directory` 必须是新的目录；它是临时权重缓存，进程结束后可删除。此命令真正加载原始完整 FuXi、微调权重和功率头，并处理输入历史；不读取未来实测目标来生成预测。

当前固定示例站点为 `hkust_zone_a6`（实际容量 0.05 MW），使用截至 **2021-12-02 08:00 +08:00** 的 16 条历史观测，输出 **08:15 +08:00 的预测功率 0.0127389237 MW，约 12.739 kW**。这是已发生时段上的离线预测示例，不是对当前日期的实时预报，也不是 DG1 的 10 MW 预测结果。

`example_input.json` 与约定一致，包含站点、容量、15 分钟间隔及温度、气压、湿度、辐照度、历史功率。`example_output.json` 返回 `predictions` 数组，每点含 timestamp、power_mw，并标识模型版本。

换一组历史数据预测时，需要在 checkpoint 的站点 schema 中找到对应电站，使用其真实容量和训练预处理参数；完整 FuXi 还需要与预测起点同期的全球天气输入及站点位置，由模型侧 context 提供。只有四个环境值或全段缺失历史功率，不能满足当前 16×5 模型输入。不能将 2021 年天气用于今天，也不能将现有站点改名为 DG1 或把其容量改为 10 MW 后声称完成目标站验证。

模型侧已能执行光伏功率推理；DG1～DG3 实站输入适配、实际气象供应、采集系统与保护计算联调仍不等于已经完成。应用服务接口与滚动优化不在这个脚本中实现。

### 训练与验证复现

训练脚本为 `full_fuxi_pv/train_available_data.py`，验证脚本为 `full_fuxi_pv/verify_available_model.py`。使用项目的 `output/python312-runtime/python.exe`。训练脚本拒绝覆盖已有结果；复现实验需先为脚本设置新的独立输出目录。临时权重映射缓存可在进程结束后删除，不影响原始主模型或这里的增量权重。

加载方式仍是已有训练模型类的 `FullFuXiPV(原始主模型路径, weight_cache=新的缓存目录)`，然后将 `torch.load(model.pt, map_location='cpu', weights_only=True)['state']` 传给 `load_adapter_state`。训练数据的站点 schema 在 checkpoint 的 `schema_by_site` 中。该文件使用研究模型格式 `full_fuxi_pv_available_data_v1`；应用层的接口适配属于后续独立工作。

训练日志：`output/full_fuxi_pv_available_data_training.log`。保存权重重载验证日志：`output/full_fuxi_pv_available_data_validation.log`。
