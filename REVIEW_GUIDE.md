# 光伏预测模型审阅说明

当前已按组合模型路线训练 IWOA–CEEMDAN–VMD–BiLSTM。模型接收温度、气压、湿度、辐照度及历史实测功率，预测下一 15 分钟功率，输出单位 MW。

## 查看入口

1. [模型使用说明与测试结果](models/iwoa_ceemdan_vmd_bilstm_v1/README.md)
2. [预测曲线](models/iwoa_ceemdan_vmd_bilstm_v1/comparison_curves.png)
3. [误差对比](models/iwoa_ceemdan_vmd_bilstm_v1/comparison_metrics.png)
4. [独立验证记录](models/iwoa_ceemdan_vmd_bilstm_v1/validation.json)
5. [训练方法与 IWOA 配置](iwoa_pv/README.md)
6. [模型权重](models/iwoa_ceemdan_vmd_bilstm_v1/model.pt)

## 本次完成情况

四个硬件环境字段实际进入模型，输入历史窗口为 16 点。已完成 CEEMDAN 分解、高频分量 VMD 分解、IWOA 参数优化和 BiLSTM 训练，并提供说明书中的四组模型对比与持续性基线。模型保存后已重载，实际预测入口与独立指标复算均通过检查。

测试集共 8,849 条日间样本，完整组合模型 MAE 为 1.549 MW、RMSE 为 2.683 MW、R² 为 0.9562。组合模型在本次比较中误差最低，相对基础 BiLSTM 的改善较小。

## 仍需核对的条件

本次使用现有标称 50 MW 电站数据。说明书中的 10 MW 目标站需要提供对应数据和站点信息。IWOA 改进公式在说明书中未指定，本实现选用反向初始化、二次收敛因子和精英高斯扰动，并记录了搜索预算与参数。

数据源时间尚无已核实的时区，正式接入需确认时间含义。当前模型使用真实历史功率与四个环境字段，第三周需确认采集来源、缺失处理、运行环境和后端调用。依赖安装完成后可使用 CPU 推理。

此前 FuXi LoRA 的代码和结果保留在原目录，其独立实验记录可从 [旧模型说明](models/full_fuxi_pv_available_data_v1/README.md)查看。
