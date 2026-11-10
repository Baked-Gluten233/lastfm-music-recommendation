# 验收安全修复

最初的验收隔离与 Python -O 防护修复基于原快照 `e4c247595e727a1109298278cf356323698bc701`。下文分别记录历史修复、Windows 历史实测及当前版本的统一验收。

## 历史修复内容

- Windows 验收在项目目录之外创建临时工作副本，测试与每本 Notebook 分别隔离。尤其阶段一不能污染阶段三的冻结依据。
- 从已修复的源码模板生成新的验收 Notebook，原带执行输出的 Notebook 保持不变。结果和状态仅写入 outputs/windows_native_verification。
- 复制只包含运行需要的 src/scripts/tests/data/outputs；排除虚拟环境、缓存、交付目录、Git 数据、历史验收目录及符号链接。
- 数据拆分/矩阵一致性、最终 checkpoint/时间/聚合一致性、重评分数量及模板测试失败处理不再依赖 assert。验证重评分同时拒绝非有限误差。
- 新增9项合成回归测试，没有运行真实训练或最终测试评估，也未读取真实 test.csv。

## 历史 Linux 修复验证

在 Linux Python 3.12.14 上，独立修复副本及应用修复后的项目均执行回归；下列耗时为应用后复跑：

- python -m unittest discover -s tests -v：29项通过（16.915秒）。
- python -O -m unittest discover -s tests -v：29项通过（22.239秒）。
- python scripts/verify_windows.py --preflight：native_windows=false、native_kernel_verified=false、status=not_run。
- python scripts/verify_windows.py：退出码1，拒绝在Linux上冒充Windows验收。

日志和预检查结果见 outputs/validation_safety/。该轮隔离回归使用模拟 NotebookClient 与合成文件，内核检查以优化模式编译验证。随后完成的 Windows 实机验收范围见下节，GitHub Actions 执行状态单独记录。

原203文件快照保留在上一提交中。本次只更改安全检查、模板、测试、文档和忽略规则；原数据、指标、权重、冻结证据和已执行 Notebook 均未改变。历史实验由原代码产生，新增修复没有重新训练或重新评分测试集。普通开发脚本仍会覆盖调用目录的派生结果，完整实验必须使用独立副本。本修复防止协作代码的意外覆盖，不是针对恶意代码的权限沙箱。

## 已保存成果复核

另在 Python -O 的临时副本中校验已有最终成果：11个 checkpoint 哈希、冻结依据、时间顺序、11次运行聚合和五模型均值/标准差均通过。临时副本只复制 outputs，不复制 data；禁止读取输出目录之外的 CSV，禁止调用测试评分函数。未重训、未重新读取测试标签或评分测试集，原输出文件保持不变。证据见 outputs/validation_safety/saved_artifact_checks.log。

## Windows 实机历史验收

2026-11-07 在 Windows 11 x64、Python 3.12.13 和 PyTorch 2.9.1+cpu 上完成验收。验收源码为 `ab570cc` 加显式 UTF-8 读取修复，修复后源码随后保存为 `5f0be78`。普通与 Python `-O` 模式各29项测试通过，三个 Notebook 均在真实 Jupyter 内核执行，`pip check` 通过。历史摘要和日志保留在本机原验收证据目录，当前版本的实测结果以新生成的验收状态为准。

该轮阶段一重算验证集基线，阶段二重放9个选中配置的保存权重，阶段三核验保存的最终聚合、冻结依据和11个 checkpoint。验收没有重训真实神经模型或重新评分测试集。历史 `outputs/phase2/protocol.json` 和 `outputs/final/evaluation_state.json` 继续记录原 Linux 实验环境。

## 当前统一验收入口

在64位 Python 3.12 环境安装两份固定依赖后，运行 `python scripts/verify_windows.py`。`--preflight` 可只读查看环境。正式入口将以下证据写入 `outputs/windows_native_verification/`。

- 校验固定直接依赖版本并执行 `pip check`，保留依赖检查日志。
- 普通与 `-O` 单元测试分别使用独立副本，记录退出码、测试数、跳过数、耗时和完整日志。测试失败、零测试或存在跳过均使验收失败。
- 三个 Notebook 分别从原快照复制运行输入、重新生成并使用真实内核执行。首格核对解释器，随后在内核中封禁最终测试入口 `musicrec.final_evaluation.run_final_once` 和该模块的评分函数别名。合成回归验证这两个调用均会抛出异常。
- 保存运行源码、模板、测试和两份依赖文件的逐文件哈希与摘要。结束时再次核对源码，避免执行途中变更造成证据错配。
- 验收前后核对 `data/`、原始 `notebooks/` 和历史 `outputs/` 的文件集合与 SHA-256。缓存、本次验收目录及独立的 `content_feature_audit` 实验目录排除在历史成果集合之外，文件新增、缺失或修改均使验收失败。

入口失败或中断后仍尝试保存状态、失败日志和完整性结果。Notebook 执行失败时保留已经生成的执行副本及日志。只有各步骤通过、源码一致且冻结成果完整性通过，最终状态才为 `passed`。`native_kernel_verified` 表示三个真实内核均执行通过，测试评分防护的安装情况逐 Notebook 记录。

本机运行前将该进程的 `TEMP`、`TMP`、`TMPDIR` 显式指向本次 `E:\Temp` 隔离目录，并提前创建目录和检查实际临时路径。Python 优先读取 `TMPDIR`，因此只设置前两项仍可能使用继承的旧路径。将可配置缓存指向同次运行的缓存目录，验收状态会记录实际临时目录。执行后的 Notebook 和验收证据属于交付物，原带历史输出 Notebook 保留。

仓库根 `.github/workflows/windows-verification.yml` 提供手动 `workflow_dispatch`，运行目录为 `music_recommendation`，上传验收证据的路径也包含该前缀。Actions 状态依据实际工作流运行记录填写，与本机 Windows 验收分别记录。

## 2026 年 11 月 10 日复验

Windows 11 x64、Python 3.12.13、PyTorch 2.9.1+cpu 环境中，普通测试 50 项、Python `-O` 测试 50 项全部通过，跳过数均为 0。固定依赖检查与 `pip check` 通过，三个重新生成的 Notebook 在真实 Jupyter 内核中完整执行，错误输出为 0。入口总耗时 281.526 秒。

170 个受保护历史文件的前后集合与 SHA-256 一致，运行源码哈希一致。完整机器记录见 [status.json](../outputs/windows_native_verification/status.json)，人工可读摘要与运行限制见 [Windows 验收结果](../outputs/windows_native_verification/ACCEPTANCE.md)。

初轮合成检查发现继承的 `TMPDIR` 指向 D 盘。统一 `TEMP`、`TMP`、`TMPDIR` 后，相关合成检查重新通过，正式验收确认实际临时目录位于本次 E 盘目录。入口已加入临时目录可用性及回退检测。

本轮另行完成标签特征的六次真实验证训练，范围与结果见 [标签特征审计](content_feature_audit.md)。GitHub Actions 尚无本轮云端运行记录。
