# Two-Tower 标签自贡献审计与验证集对照

## 审计结论

当前内容特征由用户标注艺人的事件聚合而成。训练正样本对应用户提供的标签会进入该艺人的全局内容向量，训练样本的内容特征因而含有自身行为贡献。对留出交互，已有过滤规则会去除该用户对该艺人的标签事件。这构成需要量化的训练特征内生性与分布差异。

冻结训练集共有 74,308 个用户–艺人交互，其中 16,578 对含自身获准标签，占 22.31%。经过原算法的 2,000 标签词表筛选与每艺人 top-32 截断，15,239 对的自身标签进入实际输入，占 20.51%。9,662 对含自身独占提供的 top-32 标签，占 13.00%。最终选中的 33,352 个艺人–标签特征中，27,983 个仅由一个用户提供。

这些计数描述直接特征贡献。性能影响使用下文同配置验证集对照衡量。

## 对照协议

全局聚合对照和五折用户 cross-fitting 使用同一个共享双塔模型结构。训练用户排序后，以 NumPy `default_rng(42)` 的置换顺序均衡分配到五折，人数依次为 379、379、378、378、378。

每个接收折在局部词表、TF、DF、IDF、top-32 及归一化之前，排除该折所有用户提供的标签事件。用户的训练正样本、负样本和验证集全目录评分始终使用自己的折特征库。IDF 的文档数采用全部训练可见艺人数，两种策略一致。

静态 `tags.dat` 的全部 11,946 个 numeric tagID 排序后定义统一坐标，另加一个未知标签坐标。各折先完成原算法的局部词表和 top-32 选择，再映射到此固定坐标，保留局部并列排序语义。静态目录只定义坐标身份，各折标签频率与是否使用某个标签仍由其获准事件决定。

两种策略的可训练参数均为 831,424，初始化形状和同 seed 参数逐项一致。固定维度 32、学习率 0.003、seed 42/43/44、Adam 与 `weight_decay=1e-4`、batch size 1,024，每个正样本每轮抽取一个训练可见且该用户训练未见的负样本。最多训练 24 轮，每 4 轮使用验证集宏平均 NDCG@10 选择检查点，最小改善量为 `1e-5`，连续三次验证未改善触发早停。

候选集合及指标复用现有评价入口，采用完整艺人目录、用户训练历史过滤和显式冷目标统计。五个接收折的训练正样本内容覆盖率为 71.50%–73.89%。

## 本次验证结果

2026 年 11 月 10 日在 Windows 11 x64、Python 3.12.13、PyTorch 2.9.1+cpu 上完成六次真实训练。每次均运行 24 轮，并按固定验证规则选择检查点。以下汇总采用三个 seed 的均值与样本标准差。

| 特征策略 | NDCG@10 | Recall@10 | Catalogue coverage@10 |
| --- | --- | --- | --- |
| Global | 0.100167 ± 0.001732 | 0.113751 ± 0.001657 | 0.070327 ± 0.003217 |
| 五折 cross-fitting | 0.097841 ± 0.001326 | 0.111910 ± 0.001074 | 0.067926 ± 0.003563 |

| Seed | Global NDCG@10 | Cross-fitting NDCG@10 | 成对差值 |
| --- | --- | --- | --- |
| 42 | 0.099795 | 0.097920 | −0.001875 |
| 43 | 0.102054 | 0.096477 | −0.005577 |
| 44 | 0.098651 | 0.099125 | +0.000475 |

cross-fitting 的平均 NDCG@10 比同坐标空间的 global 低 0.002326，三个 seed 的差值方向存在变化。该结果量化了本次特征策略的敏感性，结合下文的覆盖与共享参数限制解释。六个检查点的宏指标及逐用户结果均重放通过，最大宏指标绝对误差为 0。

## 解释边界

本次统一静态坐标的模型相对历史 2,000 标签 embedding 增加了参数，随机数消耗和初始化也随之变化。因此主要比较为本次重新训练的 global 与 cross-fitting，历史模型指标仅作历史参照。

cross-fitting 同时减少标签提供者和内容覆盖率，其差异反映完整特征策略的敏感性。global 选用 2,000 个不同标签，五折局部词表合计覆盖 2,385 个不同标签，有效更新的 embedding 行也随策略变化。共享模型参数还可以通过其他折的合法训练样本间接学习标签。严格排除与不变性测试针对当前用户所在折的特征构造结果。

本次实验使用独立副本，仅包含训练集、验证集、静态目录、原始标签事件、两个来源清单及所需源码。初始 17 个文件的哈希与 `test_csv_present=false` 记录于 `workspace_inventory.json`。原实验的五模型比较和最终方案属于此前冻结的实验。

## 使用入口与证据

在 `music_recommendation` 目录执行以下命令。

```powershell
python scripts/audit_content_features.py --output-dir E:\Temp\lastfm-content-audit-new
python scripts/audit_content_features.py --run-validation --output-dir E:\Temp\lastfm-content-validation
python scripts/audit_content_features.py --replay-only --output-dir outputs/content_feature_audit
```

默认入口仅生成特征审计。`--root` 可指定精确输入组成的独立副本。训练入口遇到已存在的训练目录或实验清单时，要求选择新输出目录或使用 `--replay-only`。失败状态写入 `status.json`，显式重放失败写入 `replay.json`。

`outputs/content_feature_audit` 保存以下证据。

- `audit.json`、`bank_audits.json` 与 `user_folds.csv` 记录覆盖率、排除结果和固定折分配。
- `static_tag_namespace.json`、`local_vocabularies.json` 与 `feature_banks.npz` 保存完整特征表示。
- `protocol.json` 记录模型配置、实验边界和 Windows/Python/PyTorch 运行环境。
- `global` 和 `crossfit` 各保存三个 seed 的 NPZ 检查点、训练历史、逐用户指标和结果 JSON。
- `validation_runs.csv`、`validation_summary.csv`、`paired_seed_differences.csv`、`paired_user_differences.csv` 提供汇总与成对差值，差值统一为 cross-fitting 减 global。
- `artifact_manifest.json` 保存训练前捕获且训练后复核的源码哈希、输入哈希与产物哈希。重放检查这些来源并逐项核验检查点特征缓冲，以 `1e-7` 容差验证六次宏指标和逐用户结果，显式拒绝非有限指标。

原 `ContentTwoTower`、默认模型网格与旧 NPZ 使用原有入口。新增实验使用独立 `content_two_tower_audit` family 与 `CrossFittedContentTwoTower`，由同一训练器执行。

## 回归验证

新增 11 项测试覆盖自身折标签增删及重复不变性、其他折供体标签影响、留出及目录外事件过滤、固定坐标映射、同参数初始化、正负样本与全目录评分一致性、梯度、旧模型严格加载、默认仅审计、重复训练一致性、六检查点重放、非有限指标拒绝、来源变化拒绝以及训练失败状态。

```powershell
python -B -m unittest discover -s tests -p test_content_crossfit.py -v
python -B -O -m unittest discover -s tests -p test_content_crossfit.py -v
```
