# Last.fm 艺人推荐

基于 HetRec 2011 Last.fm 2K 的可复现推荐系统项目，包含 Popularity、ItemCF、BPR-MF、NeuMF 和内容双塔的训练、验证、冻结决策及独立测试成果。

- [项目说明与运行方法](music_recommendation/README.md)
- [标签特征审计与验证集对照](music_recommendation/docs/content_feature_audit.md)
- [Windows 验收流程与证据](music_recommendation/docs/validation_safety.md)
- [备份与恢复说明](BACKUP.md)
- [逐文件完整性清单](BACKUP_MANIFEST.json)

Windows 验收区分真实神经模型训练、保存权重的验证集重放，以及最终保存成果的一致性核验。具体版本和运行范围见上述记录。

原始数据的来源及非商业使用条件见 [数据许可说明](music_recommendation/data/raw/hetrec2011-lastfm-2k/readme.txt)。
