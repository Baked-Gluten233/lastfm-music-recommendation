# Last.fm 艺人推荐：可复现项目

**项目本身已包含两个基线、三个后续模型、验证选择、冻结决策和一轮独立测试。** 第一阶段含数据审计、训练集 EDA、固定拆分、Popularity 与 ItemCF；第二阶段含三种神经模型的12次真实训练、三种子统计与验证集checkpoint重评分；最终阶段固定主模型为ItemCF，并在训练+验证重拟合后只做一轮预先约定的测试。包内提供代码、数据、Notebook、真实表图、权重和运行说明，不包含正式报告/PPT，不代填成员信息或贡献。

测试宏平均 NDCG@10：Popularity 0.065566，**ItemCF 0.172136**，BPR-MF 0.086360，NeuMF 0.092789，内容双塔0.112317。后三者为固定配置的三种子均值。最终选择在测试前冻结，没有看测试后调参或换模型。详见 `outputs/final/five_model_test_comparison.csv`。

## 任务与数据

为已有听歌记录的 Last.fm 用户推荐未出现在其训练历史中的艺人，输出 Top-10/20 排名。weight > 0 表示观察到的隐式反馈，不是显式评分；未观察到不代表不喜欢。数据为累计播放次数，**没有听歌时间**，所以任务是随机留出下的偏好检索，不是预测下一次听歌。

数据：GroupLens 托管 HetRec 2011 Last.fm 2K，非商业用途。见 `data/raw/hetrec2011-lastfm-2k/readme.txt` 和 `docs/research.md`。附原始压缩包与校验值，避免网络限制影响课程复现。

## Windows 运行（CPU）

建议 Windows 10/11、64 位 Python 3.12，先把整个压缩包解压到普通用户可写目录，例如 `E:\Temp\lastfm-project\music_recommendation`，不要直接在压缩包里打开 Notebook。PowerShell 中进入该目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\run_phase1.py
.\.venv\Scripts\python.exe -m jupyterlab
```

然后打开 `notebooks/01_data_and_baselines.ipynb`，选择本环境的 Python 内核，执行 Restart Kernel and Run All。首格会打印解释器路径和 Python 版本。直接调用虚拟环境解释器，无须修改 PowerShell 执行策略，也不用激活脚本、WSL、CUDA 或 C++ 编译工具。如果 `py -3.12` 不存在，需要先安装官方 Python 3.12 64 位版。

第二阶段再安装官方 CPU-only PyTorch，并运行：

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-phase2.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\replay_phase2.py
```

上述最后一步是从随包真实权重重新评分，核对9个选中配置checkpoint，不重新训练。要从头训练全部12次运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_phase2.py
.\.venv\Scripts\python.exe scripts\summarize_phase2.py
```

打开 `notebooks/02_model_iterations.ipynb` 可阅读模型原理、重新评分与测试。默认 `RETRAIN_ALL=False` 是快速重评分模式；改为 `True` 会重新训练全部网格和种子，不能把两种模式混称。训练耗时取决于CPU，使用4个线程；8 GB左右内存的电脑更稳妥，此建议不是实测最低内存需求。

最终结果查看与核验：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_final.py
```

或打开 `notebooks/03_frozen_final_test.ipynb`。它只核验保存的逐用户结果、聚合、冻结文件与checkpoint哈希，**不再次读取或评分测试集**。已完成目录的 `run_final_once.py` 有二次运行保护，正常查看结果不应清除它。

若要从零独立复现，请在另一个干净副本中按阶段一、阶段二、冻结、最终评估的顺序运行，保留原包作为对照；不得用已看到的测试成绩更改选择。第一次最终评估的明确入口是：

```powershell
.\.venv\Scripts\python.exe scripts\freeze_final_decision.py
.\.venv\Scripts\python.exe scripts\run_final_once.py --execute-frozen-test
```

这些命令适用于尚无 `outputs/final` 记录的首次实验目录，不是本交付已完成目录的日常运行命令。具体冻结规则、执行顺序、固定轮数和完整数值均在 `outputs/final`，旧阶段文档保留其当时状态。

**历史训练与本机验收分别记录。** 原五模型实验在 Linux + Python 3.12.14 上完成。2026 年 11 月 7 日，Windows 11 x64 + Python 3.12.13 + PyTorch 2.9.1+cpu 的实机验收通过普通和优化模式各 29 项测试，并以真实 Jupyter 内核执行三个新生成的隔离 Notebook。阶段一重建验证基线，阶段二重放九个选中 checkpoint 的验证指标，阶段三核验保存的最终成果。随包原 Notebook 的历史输出仍对应原实验运行。2026 年 11 月 10 日，当前源码再次通过普通与优化模式各 50 项测试、三个原生 Notebook 和 170 个受保护文件的完整性核对，跳过数及 Notebook 错误输出均为 0。详见 `outputs/windows_native_verification/status.json` 与 [验收记录](docs/validation_safety.md)。

Windows 验收入口如下。临时副本和缓存显式设在 E 盘的本次运行目录，进程环境设置仅影响当前 PowerShell 会话。

```powershell
$run = 'E:\Temp\lastfm-verification'
New-Item -ItemType Directory -Force -Path "$run\temp", "$run\cache" | Out-Null
$env:TEMP = "$run\temp"
$env:TMP = "$run\temp"
$env:TMPDIR = "$run\temp"
$env:MPLCONFIGDIR = "$run\cache\matplotlib"
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe scripts\verify_windows.py --preflight
.\.venv\Scripts\python.exe scripts\verify_windows.py
```

验收入口执行依赖检查、普通与 Python `-O` 测试和三个原生 Jupyter Notebook，并记录源码摘要、数量、耗时和退出码。测试与每本 Notebook 分别使用项目之外的独立临时副本，Notebook 从源码模板重新生成。验收核对受保护数据、历史输出及原 Notebook 的前后文件集合与哈希，结果写入 `outputs/windows_native_verification`。结果比较采用既有指标误差阈值，运行耗时按实际环境记录。

验收、数据一致性、最终聚合和 Notebook 模板中的重要检查使用显式异常，不依赖可被 `python -O` 删除的 `assert`。模板里的子进程测试失败会抛出异常并使执行非零退出。`tests/test_validation_safety.py` 用合成数据和模拟执行器检查副本隔离、失败传播与优化模式防护；这不是 Windows 或真实 Jupyter 内核验收。原已执行 Notebook 保留历史输出，尚未重新执行；其历史代码不作为新的安全验收入口。手动在原目录运行阶段一/二脚本仍会覆盖派生证据，请用独立副本。

仓库根目录的 `.github/workflows/windows-verification.yml` 提供手动触发的 Windows 验收，工作目录设为 `music_recommendation`。Actions 的触发、执行及通过状态单独记录，本机证据见验收文档。`.gitattributes` 为源码、根文档和历史日志固定 LF。原始数据与新增冻结实验、验收证据目录按原始字节归档，保留原生运行产生的换行及其校验值。

如下载失败，已随包提供原始数据，`download_data.py` 会先校验已有压缩包再解压。不要关闭 TLS 校验或绕过机构网络限制。若需要手动下载，只用 [GroupLens 官方页面](https://grouplens.org/datasets/hetrec-2011/) 的 Last.fm 链接，保存到 `data/raw/hetrec2011-lastfm-2k.zip`，再运行下载脚本。校验不符时脚本会停止。

Linux/macOS 对应命令使用 `.venv/bin/python`，创建环境用 `python3 -m venv .venv`。

## 标签自贡献审计与验证集对照

训练授权的用户标签会进入全局艺人特征。新增审计统计当前用户自身标签在原始事件、词表及 top-32 特征中的覆盖率，并提供五折用户 cross-fitting 对照。每折在词表和 TF-IDF 构造前排除本折用户标签，训练正负样本与验证评分采用同一折特征。

```powershell
# 新实验指定新的输出目录，先生成审计再执行六次训练。
$audit = 'E:\Temp\lastfm-content-audit-new'
.\.venv\Scripts\python.exe scripts\audit_content_features.py --output-dir $audit
.\.venv\Scripts\python.exe scripts\audit_content_features.py --run-validation --output-dir $audit
# 从保存权重核对六次验证指标。
.\.venv\Scripts\python.exe scripts\audit_content_features.py --replay-only --output-dir outputs\content_feature_audit
```

同一固定 tagID 坐标下的 global 与 cross-fitting 各运行种子 42/43/44，使用既有维度、学习率及训练预算。新结果只用于验证集敏感性比较，并记录内容覆盖和参数规模。cross-fitting 同时减少标签提供者，因此指标变化包含多种特征差异。方法、真实结果与重放证据见 [标签特征审计](docs/content_feature_audit.md)。历史冻结成果按原实验来源保留。

## 项目导航

- `notebooks/01_data_and_baselines.ipynb`：带历史进程内逐格真实运行输出和 3 张 EDA 图；新生成副本的 Windows 原生执行证据单独保存。
- `src/musicrec/data.py`：字段审计、固定拆分、索引映射。
- `src/musicrec/models.py`：可解释的热门和 ItemCF 实现。
- `src/musicrec/evaluation.py`：Precision / Recall / NDCG / HR、覆盖率、用户 bootstrap。
- `src/musicrec/pipeline.py`：统一配置、验证集调参、结果落盘，不提供测试评估入口。
- `data/processed/*.csv`：固定 train / validation / test 与 ID 映射。
- `outputs/`：真实审计、拆分哈希、逐用户指标、六组 ItemCF 验证结果、图和运行环境。
- `docs/phase1_findings.md`：阶段发现与尚未支持的结论。
- `docs/experiment_protocol.md`：候选集合、指标、冷启动、最终测试规则。
- `docs/model_plan.md`：阶段一时的历史设计计划；第二阶段实际实现以 `phase2_method.md` 为准。
- `notebooks/02_model_iterations.ipynb`：第二阶段真实权重重评分、学习曲线、比较与代码检查。
- `src/musicrec/{content,neural,phase2}.py`：防泄漏内容构造、三种神经模型、固定训练配置。
- `outputs/phase2/`：全部12次训练日志/结果/权重，选中3种子汇总，checkpoint重评分。
- `docs/phase2_method.md` / `docs/phase2_findings.md`：实际方法、结果、失败分析与边界。
- `notebooks/03_frozen_final_test.ipynb`：已保存的最终结果、预先固定案例和一致性核验。
- `src/musicrec/final_evaluation.py`：固定轮数重拟合与一次性测试流程，无测试后调参。
- `outputs/final/`：冻结决策、时间顺序、11个最终checkpoint、五模型测试表、分组、案例和核验。
- `scripts/verify_windows.py`：Windows 原生 Notebook 验收、优化模式测试及完整性检查。
- `docs/learning_and_contributions.md`：学习检查表、AI 披露与真实贡献记录模板。

## 复现与安全边界

历史实验入口将派生成果写入项目目录，验收使用独立临时副本，标签审计还支持显式指定输出目录。原始数据来自官方公开地址。拆分种子42；模型种子42/43/44。开发脚本重新运行会覆盖派生验证输出，可能使旧冻结证据不再匹配，因此完整新实验应另建干净副本。更改配置不能继续引用旧指标。不要根据测试结果重分数据、改候选集合或换指标；测试封存/解封是研究流程约定，不是加密访问控制。

本项目为 AI 辅助实现。成员必须实际运行、理解、验证、修改并如实记录贡献；不能把 AI 生成部分冒称完全独立原创，更不能虚构未训练模型的成绩。
