# 私有项目备份

完整项目位于 [`music_recommendation/`](music_recommendation/)。克隆仓库后进入该目录，按 [项目 README](music_recommendation/README.md) 恢复环境；不需要 Last.fm 账号或密钥。仓库根 README 与最初提交保持不变。

当前包含211个项目文件：代码、带历史真实输出的 Notebook、原始数据及许可、固定拆分、模型权重、实验结果、冻结记录、图表，以及验收安全修复和测试证据。逐文件 SHA-256 与 Git blob SHA 见 [BACKUP_MANIFEST.json](BACKUP_MANIFEST.json)。不包含虚拟环境、缓存、凭据、课程附件、报告或 PPT。

## 恢复点与修复

- 原始可用快照：提交 `e4c247595e727a1109298278cf356323698bc701`，原203文件完整保存，可单独恢复。
- 后续修复：Windows验收的测试及每本 Notebook 均使用独立临时副本；关键检查不受 Python -O 优化影响，失败会正确退出。
- 普通模式和 Python -O 模式各29项回归通过。另在隔离副本核对11个最终checkpoint、冻结证据、时间顺序和保存的聚合结果。
- 原始数据、模型权重、历史指标、冻结证据和带输出 Notebook 均未修改；没有再次训练或评分保留测试集。修复说明见 [验收安全记录](music_recommendation/docs/validation_safety.md)。

## 尚未验证

Windows实机、原生Jupyter完整验收仍未完成。项目内 `.github/workflows/` 不在仓库顶层，作为原配置备份保存，未启用或触发。旧Notebook保留历史输出；安全验收入口会从修复后的模板生成新的副本。

HetRec 2011 Last.fm 数据仅限非商业用途，保留原数据许可与来源说明。使用新实验目录进行完整复现，不应覆盖本备份中的冻结成果。
