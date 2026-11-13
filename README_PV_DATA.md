# 光伏历史数据入口

当前项目入口见 [项目 README](README.md)。旧版独立清洗表、旧样本和早期脚本已归档至 [历史目录](archive/2026-11-09/README.md)，当前数据路径保持不变。

新增公开数据按来源分类存放在 `data/raw/public_datasets/`。香港科大 2021—2023 年屋顶光伏数据位于 `data/raw/public_datasets/hkust_rooftop_pv_2021_2023/`，使用方法和版本见 [香港科大数据说明](data/raw/public_datasets/hkust_rooftop_pv_2021_2023/DATASET_GUIDE_ZH.md)。其原始压缩包、解压数据和下载校验记录独立保存，尚未混入现有训练集。

香港科大清洗结果另存于 `data/processed/hkust_rooftop_pv/`，详见 [清洗字段、样本及使用说明](data/processed/hkust_rooftop_pv/README.md)。主样本使用原生 15 分钟站点；小时站点和逆变器数据独立保留。时区、坐标和海平面气压的限制在说明及质量报告中记录，尚未完成全球天气配对。

全站整理后的数据位于 `data/processed/pv_all_sites_v2/`。

先阅读 [数据使用说明](data/processed/pv_all_sites_v2/README.md)，再选择对应站点的 `datasets/site_*.npz` 和 `datasets/site_*_schema.json`。全站清洗表为 `pv_all_sites_cleaned.csv`，质量报告为 `data_report.json`，校验结果为 `validation.json`。

9 份原始文件已全部处理，去重后形成 8 套站点数据。已有 414,697 个有效训练／验证／测试样本，全部通过独立校验。容量为来源文件标称值；第 1 站副本与第 8 站内容相同但容量标注冲突，处理方式与限制见数据说明。

旧版数据与已训练模型保留供追溯。新数据使用统一日期划分，不要与旧版样本、归一化配置或指标混用。
