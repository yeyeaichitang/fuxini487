# 光伏功率预测模型输入输出约定（讨论稿）

> 用途：统一数据处理、模型训练和后端接口的输入输出，避免模型完成后因字段名称、单位、时间粒度不一致而返工。
>
> 当前约定依据现有项目数据制定；标有“待确认”的内容需要双方确认后定稿。

## 1. 当前项目基础约定

| 项目 | 约定 |
|---|---|
| 光伏站编号 | `DG1`、`DG2`、`DG3` |
| 单站额定容量 | `10.0 MW` |
| 数据时间粒度 | 15 分钟 |
| 每日数据点数 | 96 点 |
| 时间格式 | ISO 8601，必须携带时区，例如 `2026-11-01T10:15:00+08:00` |
| 模型目标 | 预测光伏有功功率 |
| 功率单位 | `MW` |

## 2. 单个时间点的数据格式

模型使用的每个时间点统一为以下结构：

```json
{
  "timestamp": "2026-11-01T10:15:00+08:00",
  "temperature_c": 18.6,
  "pressure_hpa": 929.7,
  "humidity_percent": 53.0,
  "irradiance_w_m2": 742.0,
  "power_mw": 4.21
}
```

字段说明：

| 字段 | 类型 | 单位 | 是否必需 | 说明 |
|---|---|---|---:|---|
| `timestamp` | string | - | 是 | 数据采集时间，精确到秒并携带时区 |
| `temperature_c` | number | °C | 是 | 环境温度 |
| `pressure_hpa` | number | hPa | 是 | 大气压强 |
| `humidity_percent` | number | % | 是 | 相对湿度，取值范围 0～100 |
| `irradiance_w_m2` | number | W/m² | 是 | 总太阳辐照度，不小于 0 |
| `power_mw` | number/null | MW | 建议保留 | 当前实测有功功率；硬件暂时不能提供时允许为 `null` |

说明：现有项目的数据集本身包含以上四个环境特征和实际功率，因此建议训练和实时推理沿用相同名称与单位。

## 3. 模型输入

模型输入为某个光伏站最近连续的一段历史数据，而不是只有当前一个时间点。

```json
{
  "station_id": "DG1",
  "capacity_mw": 10.0,
  "as_of": "2026-11-01T10:15:00+08:00",
  "interval_minutes": 15,
  "history": [
    {
      "timestamp": "2026-11-01T09:45:00+08:00",
      "temperature_c": 17.8,
      "pressure_hpa": 929.9,
      "humidity_percent": 55.2,
      "irradiance_w_m2": 680.0,
      "power_mw": 3.62
    },
    {
      "timestamp": "2026-11-01T10:00:00+08:00",
      "temperature_c": 18.2,
      "pressure_hpa": 929.8,
      "humidity_percent": 54.1,
      "irradiance_w_m2": 715.0,
      "power_mw": 3.94
    },
    {
      "timestamp": "2026-11-01T10:15:00+08:00",
      "temperature_c": 18.6,
      "pressure_hpa": 929.7,
      "humidity_percent": 53.0,
      "irradiance_w_m2": 742.0,
      "power_mw": 4.21
    }
  ]
}
```

输入规则：

1. `history` 按 `timestamp` 从早到晚排列。
2. 相邻数据间隔应为 15 分钟。
3. `as_of` 应等于 `history` 最后一条数据的时间。
4. 数值归一化不由接口调用方完成，由模型服务按照训练时的预处理参数统一完成。
5. 缺失值不能擅自填 0。缺失时传 `null`，由模型服务按双方约定的规则处理。
6. `history` 的固定长度暂时待确认，候选值为 4 点（1 小时）、16 点（4 小时）或 96 点（24 小时）。

## 4. 模型输出

第一阶段建议每次预测未来 15 分钟的一个功率点，便于尽快跑通滚动预测。模型接口统一输出数组，为后续扩展多步预测预留空间。

```json
{
  "station_id": "DG1",
  "based_on": "2026-11-01T10:15:00+08:00",
  "interval_minutes": 15,
  "model_name": "pv-forecast",
  "model_version": "baseline-v1",
  "predictions": [
    {
      "timestamp": "2026-11-01T10:30:00+08:00",
      "power_mw": 4.58
    }
  ]
}
```

输出规则：

1. `predictions` 按时间从早到晚排列。
2. `power_mw` 为非负数。
3. 正常情况下，预测值应限制在 `0～capacity_mw`；本项目默认范围为 `0～10 MW`。
4. `based_on` 表示本次预测使用的最新实测数据时间。
5. 必须返回 `model_version`，便于联调时确认实际调用的是基线模型还是微调模型。
6. 第二阶段可将 `predictions` 扩展为未来 4 个点，即未来 1 小时，但不改变字段结构。

## 5. 前端所需数据

后端会把模型结果与真实功率整理为当前前端使用的格式，于琬倩不需要让模型直接输出前端格式：

```json
{
  "timestamp": "2026-11-01T10:30:00+08:00",
  "predicted": 4.58,
  "actual": 4.46
}
```

其中：

- `predicted` 来自模型输出的 `power_mw`；
- `actual` 来自硬件或实测数据；
- 后端负责组合，模型只负责输出预测功率。

## 6. 于琬倩交付模型时需要一起提供

不能只提供一个模型权重文件。建议完整交付以下内容：

```text
model/
├── model.onnx / model.pt / model.pkl
├── preprocessing.json
├── feature_schema.json
├── example_input.json
├── example_output.json
├── metrics.json
└── README.md
```

具体说明：

| 文件 | 内容 |
|---|---|
| 模型文件 | 可直接加载执行推理的模型 |
| `preprocessing.json` | 归一化、标准化、缺失值填补等参数 |
| `feature_schema.json` | 特征名称、顺序、类型和单位 |
| `example_input.json` | 一组能够运行的固定测试输入 |
| `example_output.json` | 对应的预期输出，用于验证部署结果 |
| `metrics.json` | MAE、RMSE、MAPE 等测试集指标 |
| `README.md` | Python 版本、依赖版本、加载和推理方法 |

## 7. 需要双方尽快确认的事项

请于琬倩确认以下内容，确认后本文件作为接口定稿依据：

1. 模型是否需要历史实际功率 `power_mw` 作为输入特征？
2. 历史窗口固定使用多少个 15 分钟数据点？
3. 第一阶段是预测下一个 15 分钟，还是一次预测未来 1 小时的 4 个点？
4. 多步预测时，是否需要未来时段的天气预报数据？如果需要，应明确其来源和字段格式。
5. 最终模型导出格式是 ONNX、PyTorch `.pt`，还是 Python `.pkl`？

## 8. 建议的第一阶段验收标准

1. 使用 `example_input.json` 能成功加载开源基线模型并完成一次预测。
2. 返回结果符合第 4 节格式。
3. 预测功率单位为 MW，且结果位于 `0～10 MW`。
4. 输入中已经包含温度、气压、湿度和辐照度四个硬件字段。
5. 替换微调模型时，只需要替换模型文件和预处理配置，不修改外部接口。

