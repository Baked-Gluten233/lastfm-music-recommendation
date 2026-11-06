# 验收安全修复

本次在原可用快照之后单独提交验收隔离与 Python -O 防护修复，并补强忽略规则。原快照提交为 e4c247595e727a1109298278cf356323698bc701，可作为历史结果恢复点。

## 修复内容

- Windows 验收在项目目录之外创建临时工作副本，测试与每本 Notebook 分别隔离。尤其阶段一不能污染阶段三的冻结依据。
- 从已修复的源码模板生成新的验收 Notebook，原带执行输出的 Notebook 保持不变。结果和状态仅写入 outputs/windows_native_verification。
- 复制只包含运行需要的 src/scripts/tests/data/outputs；排除虚拟环境、缓存、交付目录、Git 数据、历史验收目录及符号链接。
- 数据拆分/矩阵一致性、最终 checkpoint/时间/聚合一致性、重评分数量及模板测试失败处理不再依赖 assert。验证重评分同时拒绝非有限误差。
- 新增9项合成回归测试，没有运行真实训练或最终测试评估，也未读取真实 test.csv。

## 验证结果

在 Linux Python 3.12.14 上，独立修复副本及应用修复后的项目均执行回归；下列耗时为应用后复跑：

- python -m unittest discover -s tests -v：29项通过（16.915秒）。
- python -O -m unittest discover -s tests -v：29项通过（22.239秒）。
- python scripts/verify_windows.py --preflight：native_windows=false、native_kernel_verified=false、status=not_run。
- python scripts/verify_windows.py：退出码1，拒绝在Linux上冒充Windows验收。

日志和预检查结果见 outputs/validation_safety/。隔离回归使用模拟 NotebookClient 与合成文件，内核检查以优化模式编译验证；未运行真实 Jupyter 内核。Windows 原生、完整 Notebook 验收和 GitHub Actions 仍待实际运行，不能声明通过。

原203文件快照保留在上一提交中。本次只更改安全检查、模板、测试、文档和忽略规则；原数据、指标、权重、冻结证据和已执行 Notebook 均未改变。历史实验由原代码产生，新增修复没有重新训练或重新评分测试集。普通开发脚本仍会覆盖调用目录的派生结果，完整实验必须使用独立副本。本修复防止协作代码的意外覆盖，不是针对恶意代码的权限沙箱。

## 已保存成果复核

另在 Python -O 的临时副本中校验已有最终成果：11个 checkpoint 哈希、冻结依据、时间顺序、11次运行聚合和五模型均值/标准差均通过。临时副本只复制 outputs，不复制 data；禁止读取输出目录之外的 CSV，禁止调用测试评分函数。未重训、未重新读取测试标签或评分测试集，原输出文件保持不变。证据见 outputs/validation_safety/saved_artifact_checks.log。
