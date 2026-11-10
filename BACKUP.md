# 项目备份与恢复

完整项目位于 [music_recommendation/](music_recommendation/)，环境安装与入口见 [项目 README](music_recommendation/README.md)。仓库包含原始数据及许可、固定切分、代码、历史 Notebook、权重、冻结成果，以及新增标签特征对照和 Windows 验收证据。

## 历史恢复点

- 原始完整快照为 `e4c247595e727a1109298278cf356323698bc701`。
- 本轮改进的基线为 `5f0be782e6b50ea83efaf4e9ccbe954effb75b90`，包含独立验收副本、优化模式防护和 Windows UTF-8 修复。
- 原五模型实验在 Linux 上训练和评估。历史环境字段按其所属实验解释，本机验收由独立状态文件记录。

## 新增成果与验收

标签特征审计及六次验证集对照位于 `music_recommendation/outputs/content_feature_audit/`，方法和运行结果见 [特征审计文档](music_recommendation/docs/content_feature_audit.md)。新实验记录固定输入、源码哈希、用户分折、特征、学习曲线、权重、逐用户指标和重放结果。

Windows 验收位于 `music_recommendation/outputs/windows_native_verification/`，执行状态见 [status.json](music_recommendation/outputs/windows_native_verification/status.json)，范围与证据见 [验收文档](music_recommendation/docs/validation_safety.md)。统一入口检查依赖、普通及优化测试、三个原生 Jupyter Notebook，以及受保护成果的前后哈希。

仓库根 `.github/workflows/windows-verification.yml` 提供手动触发入口。云端 Actions 状态与本机实测分别记录。

## 完整性与复现

[BACKUP_MANIFEST.json](BACKUP_MANIFEST.json) 列出交付文件的字节数、SHA-256 和 Git blob SHA，并汇总实际文件数量与验证状态。清单覆盖项目及仓库根交付文件，自身通过 Git 版本追踪。

`.gitattributes` 为源码、根文档与历史日志固定 LF。原始数据及新增冻结实验、验收证据目录以原始字节归档，保留原生平台生成的换行与校验值。克隆后的数据、权重及冻结成果应与清单一致。

完整新实验使用独立工作副本。随包历史原始数据、切分、权重、冻结成果和原 Notebook 受完整性校验保护。原始数据的非商业使用条件见随包许可说明。

虚拟环境、工具缓存、Jupyter 运行时连接文件及私人配置保存在本地运行目录。
