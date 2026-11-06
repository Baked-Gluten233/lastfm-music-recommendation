# 实施前研究与选择记录

研究日期：2026-11-06。工作区最初仅有课程说明与 lecture 0–6 Markdown，没有已有工程或代码。目标是小组能理解的 CPU、Windows 可移植艺人 Top-K 原型，不是商用推荐服务。阶段一预算为一个普通 CPU 进程；不需要 GPU、账户、私有数据、线上部署。评估范围是已知用户随机留出，不是下一首歌曲或实时服务。

## 四类证据与实际检索

1. 官方事实/数据：打开 GroupLens HetRec 页面与 README；网页 README 解码失败后读取同官方 ZIP 内 README，成功。下载官方压缩包并 SHA256 固定。核实非商业使用、artist 级计数、标签时间含义。页面 https://grouplens.org/datasets/hetrec-2011/ ，数据 https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip 。README Version 1.0（2011-05）；托管文件 HTTP Last-Modified 为 2019-12-03，不把托管日期当采样日期。
2. 可复用实现：检索 `site.github.com microsoft recommenders python windows item similarity implicit`；核读 https://github.com/benfred/implicit 和 https://github.com/recommenders-team/recommenders/blob/main/SETUP.md 。implicit 是 MIT、公开开发的专业协同过滤库，有 Windows wheel 说明；当前具体版本在本阶段未安装复现。Recommenders 官方说明以 Linux 验证为主，Windows 指南可能滞后，不等于它不支持 Windows。两者适合参考，但整包引入不利于本课程透明度和依赖控制。
3. 可复用资产：采用官方 Last.fm 2K；查看 SciPy 官方 PyPI 发行/平台说明 https://pypi.org/project/scipy/1.14.1/ （此页只是历史兼容依据，不作为本项目锁定版本）。实际工作区已有 NumPy 2.3.5 / SciPy 1.17.0 / pandas 2.2.3 / matplotlib 3.10.8；随后按这组版本校验 Windows CPython 3.12 wheel，保存下载/解析日志。官方包与标准稀疏操作是复用部分，算法组织和评估自行实现，不复制仓库源代码。
4. 近期实际经验：检索 `implicit recommenders Windows install scipy 2025 2026 issue`，找到 SciPy 维护者/打包贡献者第一手故障 https://github.com/scipy/scipy/issues/23527 （2025-08-29 发起，结果显示 2026-05 有后续评论，页面当前 Closed）。环境为 conda-forge Windows，涉及非 AVX512 CPU 与部分测试挂起。观察期不是本项目运行期，不能推断 pip wheel 或用户电脑有同问题。故障在本项目是否可复现未知；未将社区摘要当兼容保证。未找到两个独立、近期且与完全相同锁定环境匹配的使用报告，作为证据缺口。

中文/Gitee 查询 `site.gitee.com LastFM 推荐系统 ItemCF` 得到 https://gitee.com/crazy-xy/recommend_system/blob/master/README.md ，正文读取 405，许可证/维护状况未核实，不复用。GitHub 主来源已支持概念与候选筛选，停止继续扩大检索。用户提供的 ChatGPT 分享页在当前读取方式中仅显示空壳，未据此虚构实现；采用已确认的 Last.fm 艺人推荐与五模型路线。

## 候选与决策

- **adopt** 官方 HetRec 数据及标准 NumPy/SciPy/pandas/Matplotlib。小数据规模、非商业课程许可与 CPU 路径满足任务。明确已存在采样偏差，无法恢复听歌序列。
- **build** 透明的二值 popularity、cosine ItemCF、固定拆分与全目录指标。预计代码量少且学生能逐步手算，便于查泄漏和公平比较。
- **defer** implicit / Recommenders 框架和 PyTorch 深度模型。不是认定质量低，而是阶段一无需其额外复杂度；后续神经模型实施前核实 CPU/Windows 官方支持及原论文。
- **exclude** 无法核许可证或来源的 Gitee/教程实现，不复制代码。没有购买服务或运行收费 API，成本仅本地 CPU 时间与数据下载。

验证计划：原始数据 SHA256、解压内容与 ZIP 比对；固定拆分互斥/完整/输入顺序不变；手算指标/余弦/收缩/方向/候选掩码；真实端到端运行；Notebook Restart-and-Run-All；Windows wheel 与目标依赖解析。只有在真实 Windows 运行后才能写“Windows 测试通过”。

## 学术来源

- Last.fm 网站与 Cantador, Brusilovsky, Kuflik (2011), *2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems*, RecSys 2011；引用形式见原始 README。
- BPR: https://arxiv.org/abs/1205.2618 。NeuMF/NCF: https://arxiv.org/abs/1708.05031 。两者用于后续设计，当前没有报告其运行结果。
- 课程附件：《期末项目要求与指南》《数据集获取与选择指南》、recommender_systems_lectures_0-6_markdown_images。未将教程/课程示例成绩复制到本项目。

## 第二阶段增补（2026-11-06）

实施前核读 PyTorch 官方当前安装页及历史版本页，确认 2.9.1 提供 Linux/Windows CPU wheel： https://pytorch.org/get-started/previous-versions/ 。选择官方 CPU 索引、固定2.9.1、普通 eager 模式，无CUDA与编译器依赖。实际Linux CPU安装成功，Windows `torch-2.9.1+cpu-cp312-cp312-win_amd64.whl` 也已验证存在；下载wheel不是Windows实机验证。

模型依据核读BPR与NCF原论文（上列arXiv链接），只按核心机制自行实现。GitHub检索并查看 https://github.com/guoyang9/BPR-pytorch ，未复制代码，许可证和精确版本适配未深入验证，故不作为可直接复用组件。Gitee查询 `site.gitee.com "NeuMF" "pytorch"` 未返回可核验的匹配源码，记录缺口，不改用未知镜像。资产采用官方torch包，数据和划分不变。

近期社区检索 `site.discuss.pytorch.org CPU embedding training slow threads 2026 2025`，核读 https://discuss.pytorch.org/t/performance-collapse-under-high-concurrency-cpu-inference-thread-oversubscription/224189 。2025-12-15作者以16-vCPU Docker并发推理描述线程过量问题；作者自述操作者，独立性/具体版本不明，非Windows或本项目 workload。这只是限制线程并实际计时的动机，不能据此承诺通用最优线程数。旧Windows慢速帖子属于旧版本证据，不据此否定当前版本。没有找到两个独立、完全匹配本环境的近期经验，明确保留证据缺口。

决策：adopt 官方PyTorch自动微分和CPU wheel，build 三个小型透明模型，保留同协议全目录评价；不采用GPU生产框架、无授权来源代码或额外数据。用18项代码/泄漏/梯度/训练复现检查、12次真实验证实验和9个checkpoint重新评分验证结果，而非用论文成绩代替本项目数据。当前验证结果不预设神经模型优于ItemCF。
