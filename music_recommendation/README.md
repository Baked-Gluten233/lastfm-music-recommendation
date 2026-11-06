# Last.fm 艺人推荐：可复现项目

**项目本身已包含两个基线、三个后续模型、验证选择、冻结决策和一轮独立测试。** 第一阶段含数据审计、训练集 EDA、固定拆分、Popularity 与 ItemCF；第二阶段含三种神经模型的12次真实训练、三种子统计与验证集checkpoint重评分；最终阶段固定主模型为ItemCF，并在训练+验证重拟合后只做一轮预先约定的测试。包内提供代码、数据、Notebook、真实表图、权重和运行说明，不包含正式报告/PPT，不代填成员信息或贡献。

测试宏平均 NDCG@10：Popularity 0.065566，**ItemCF 0.172136**，BPR-MF 0.086360，NeuMF 0.092789，内容双塔0.112317。后三者为固定配置的三种子均值。最终选择在测试前冻结，没有看测试后调参或换模型。详见 `outputs/final/five_model_test_comparison.csv`。

## 任务与数据

为已有听歌记录的 Last.fm 用户推荐未出现在其训练历史中的艺人，输出 Top-10/20 排名。weight > 0 表示观察到的隐式反馈，不是显式评分；未观察到不代表不喜欢。数据为累计播放次数，**没有听歌时间**，所以任务是随机留出下的偏好检索，不是预测下一次听歌。

数据：GroupLens 托管 HetRec 2011 Last.fm 2K，非商业用途。见 `data/raw/hetrec2011-lastfm-2k/readme.txt` 和 `docs/research.md`。附原始压缩包与校验值，避免网络限制影响课程复现。

## Windows 运行（CPU）

建议 Windows 10/11、64 位 Python 3.12，先把整个压缩包解压到普通用户可写目录，例如 `D:\course\music_recommendation`，不要直接在压缩包里打开 Notebook。PowerShell 中进入该目录：

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

**验证边界：实际运行是 Linux + Python 3.12.14，不是 Windows 实测。** 已单独校验固定直接依赖存在 Windows CPython 3.12 / 通用 wheel；Windows 目标依赖解析结果另见 `outputs/windows_dependency_resolution.json` 与日志（解析的操作系统条件标记仍可能受 Linux 主机影响）。本环境无法启动需要本地通信的原生 Jupyter 内核，故 Notebook 通过 IPython 进程内按序执行所有真实代码格并捕获输出，不声称原生内核 Restart-and-Run-All 已通过。Windows 最终验收仍须在真实 Windows 执行上述测试和 Notebook。结果应对照 CSV 指标，浮点数允许极小数值误差，运行时间不应比较。

已提供 Windows 验收入口：`.\.venv\Scripts\python.exe scripts\verify_windows.py`，**仍未在真实 Windows 上验证通过**。它只在实际 Windows 执行单元测试及三个原生 Jupyter Notebook。测试和每本 Notebook 分别使用项目目录以外的独立临时副本；Notebook 从 `scripts/create*_notebook.py` 模板重新生成，不执行或覆盖随包已保存输出的原 Notebook。每本 Notebook 都从同一原始快照开始，阶段一重建的派生数据不会污染阶段三的冻结依据。原项目仅新增/更新 `outputs/windows_native_verification` 中的日志、执行结果和状态；不修改原数据、冻结证据或原 Notebook，不再次评分测试集。临时副本运行结束或失败后清理，需要额外磁盘空间。隔离用于避免本项目正常代码的意外覆盖，不是恶意代码安全沙箱。

验收、数据一致性、最终聚合和 Notebook 模板中的重要检查使用显式异常，不依赖可被 `python -O` 删除的 `assert`。模板里的子进程测试失败会抛出异常并使执行非零退出。`tests/test_validation_safety.py` 用合成数据和模拟执行器检查副本隔离、失败传播与优化模式防护；这不是 Windows 或真实 Jupyter 内核验收。原已执行 Notebook 保留历史输出，尚未重新执行；其历史代码不作为新的安全验收入口。手动在原目录运行阶段一/二脚本仍会覆盖派生证据，请用独立副本。

项目的 `.github/workflows/` 保留可选 GitHub Actions 配置；本仓库把完整项目放在 `music_recommendation/` 子目录，因此该配置不在仓库顶层，未启用、未触发，不能因已备份就声称工作流通过。`.gitattributes` 与 JSON 显式 LF 换行用于避免跨平台文本换行破坏冻结哈希。

如下载失败，已随包提供原始数据，`download_data.py` 会先校验已有压缩包再解压。不要关闭 TLS 校验或绕过机构网络限制。若需要手动下载，只用 [GroupLens 官方页面](https://grouplens.org/datasets/hetrec-2011/) 的 Last.fm 链接，保存到 `data/raw/hetrec2011-lastfm-2k.zip`，再运行下载脚本。校验不符时脚本会停止。

Linux/macOS 对应命令使用 `.venv/bin/python`，创建环境用 `python3 -m venv .venv`。

## 项目导航

- `notebooks/01_data_and_baselines.ipynb`：带进程内逐格真实运行输出和 3 张 EDA 图；原生 Jupyter 内核及 Windows 复验待完成。
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
- `scripts/verify_windows.py`：实际Windows原生Notebook验收，当前未执行通过。
- `docs/learning_and_contributions.md`：学习检查表、AI 披露与真实贡献记录模板。

## 复现与安全边界

脚本只写本项目目录，不连接 Last.fm 账户，不使用个人用户名或密钥，不修改用户电脑系统设置。只从官方公开地址取数据。拆分种子42；模型种子42/43/44。开发脚本重新运行会覆盖派生验证输出，可能使旧冻结证据不再匹配，因此完整新实验应另建干净副本。更改配置不能继续引用旧指标。不要根据测试结果重分数据、改候选集合或换指标；测试封存/解封是研究流程约定，不是加密访问控制。

本项目为 AI 辅助实现。成员必须实际运行、理解、验证、修改并如实记录贡献；不能把 AI 生成部分冒称完全独立原创，更不能虚构未训练模型的成绩。
