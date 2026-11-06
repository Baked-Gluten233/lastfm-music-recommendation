"""Regenerate the phase-1 source notebook (outputs supplied by real execution)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

def main():
    cells = [
        nbf.v4.new_markdown_cell("""# Last.fm 艺人推荐：数据理解与两个基线

**阶段一可复现 Notebook，非完整提交。** 后续 BPR-MF / NeuMF / 内容双塔尚未训练。所有展示数字由本 Notebook 实际运行产生；测试集不评价。AI 辅助编写，成员必须理解、运行并如实记录贡献。

此处保留阶段一的学习过程和当时状态；第二阶段已另行完成三个模型，见 02_model_iterations.ipynb，不应把本页“尚未训练”理解为最新项目状态。

交付验证方式：IPython 进程内按序执行全部代码格并捕获真实输出。当前环境无法启动需本地通信的原生 Jupyter 内核，因此原生 Restart-and-Run-All 与 Windows 实机运行尚未验证。

研究问题：在累计听歌记录的随机留出协议下，能否为已有用户找回未见过的艺人偏好？不声称预测下一次收听。"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import sys
import json
import subprocess
ROOT = Path.cwd().resolve()
if not (ROOT / 'src' / 'musicrec').is_dir():
    ROOT = ROOT.parent
if not (ROOT / 'src' / 'musicrec').is_dir(): raise RuntimeError('请从项目根目录或 notebooks 目录启动')
sys.path.insert(0, str(ROOT / 'src'))
print('Python:', sys.version)
print('Interpreter:', sys.executable)
print('Project:', ROOT.name)
import pandas as pd
from IPython.display import display, Image
from musicrec.pipeline import run_phase1
"""),
        nbf.v4.new_markdown_cell("""## 1. 数据来源与真实性

数据为 GroupLens 托管的 HetRec 2011 Last.fm 2K，非商业用途，见原始 readme。下格核验官方压缩包并解压；已有文件无需联网下载。固定 SHA256 不匹配时停止，不静默换数据。标签时间不是听歌时间。

原始记录含 userID、artistID、weight；每对只有累计播放次数。本阶段模型把所有正次数转为二值反馈，先防止高播放量主导相似度；这项预处理不是模型迭代。"""),
        nbf.v4.new_code_cell("""download = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'download_data.py')],
                          cwd=ROOT, check=True, capture_output=True, text=True)
print(download.stdout)
"""),
        nbf.v4.new_markdown_cell("""## 2. 完整实验：审计、拆分、训练、验证

固定 seed=42；degree >=5 用户验证/测试各留 floor(n/10)，至少一条，其余训练。degree <5 只用于训练。六组 ItemCF 均使用相同训练输入与全目录候选，选择主指标为验证宏平均 NDCG@10。

Popularity 用不同训练听众数；ItemCF 用二值共现余弦×收缩，每个源艺人保留 top-N，再对用户历史求和。源代码在 src/musicrec 下，不是黑箱网络服务。代码会检查划分互斥和完整性，但不会计算测试指标。"""),
        nbf.v4.new_code_cell("experiment = run_phase1(ROOT, seed=42)"),
        nbf.v4.new_code_cell("""audit = experiment['audit']
display(pd.Series({k:audit[k] for k in ['users','catalogue_artists','interactions','sparsity',
    'duplicate_pairs','nonpositive_weights','users_with_50_artists','artists_with_one_listener',
    'tag_events','tag_event_unknown_artists']}, name='observed'))
display(pd.Series(experiment['manifest'], name='protocol'))
"""),
        nbf.v4.new_markdown_cell("""## 3. 三张训练集 EDA 图

1. 活跃度：训练历史高度集中，原始用户很多最多/恰好 50 位艺人，因此不能假装覆盖完整终身听歌历史。低活跃用户另外分析。
2. 艺人长尾：许多艺人只有极少共同听众，直接余弦可能不稳定，收缩是 ItemCF 内部合理候选；不能根据长尾本身保证它更好。
3. 累计次数：分布跨度大；本阶段使用二值反馈，以用户覆盖而非重度听众次数衡量热门。未来若试 log(1+weight)，只能算模型内部优化。

所有图只用 train 生成，不把测试表现用于设计。"""),
        nbf.v4.new_code_cell("""for name in ['01_user_activity.png','02_artist_long_tail.png','03_listening_counts.png']:
    display(Image(filename=str(ROOT / 'outputs' / 'figures' / name)))
"""),
        nbf.v4.new_markdown_cell("""## 4. 统一指标和验证选择

对每位验证用户，从完整艺人目录去掉其训练历史；不抽样负例，也不利用测试答案过滤候选。正反馈缺失项是未知，离线评价不是线上满意度证明。冷艺人保留在分母中。

Precision@K=命中数/K；Recall@K=命中数/用户验证正例数；NDCG 按 log2(rank+1) 折扣并用理想排名归一化。先算每人，再等权平均。完整公式与最终测试方案见 docs/experiment_protocol.md。"""),
        nbf.v4.new_code_cell("""columns = ['model','ndcg@10','recall@10','precision@10','hit_rate@10','catalogue_coverage@10','ndcg@20']
display(experiment['results'][columns].round(6))
print('验证选择的 ItemCF 基线:', experiment['selection'])
print('未选择最终模型；测试指标未计算。')
"""),
        nbf.v4.new_markdown_cell("""## 5. 分组与不确定性

含冷目标用户并不意味着所有目标都冷，下表是用户组整体指标，不是“冷艺人召回率”。分组差异是相关观察，不是因果结论。

bootstrap 在同一验证拆分的用户上重采样，无法覆盖调参赢家偏差、不同拆分或模型种子波动。"""),
        nbf.v4.new_code_cell("""display(pd.read_csv(ROOT / 'outputs' / 'validation_user_groups.csv').round(6))
display(pd.DataFrame({k:v for k,v in experiment['uncertainty'].items() if isinstance(v,dict)}).T)
print(experiment['uncertainty']['caution'])
"""),
        nbf.v4.new_markdown_cell("""## 6. 独立协议测试

小样本手算验证余弦、收缩、稀疏方向、固定拆分、指标、冷目标与已见过滤。单元测试不等于完整模型正确性证明，但能发现常见实现错误。"""),
        nbf.v4.new_code_cell("""tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=ROOT, capture_output=True, text=True)
print(tests.stdout + tests.stderr)
tests.check_returncode()
print('Windows 实际运行:', experiment['runtime']['windows_execution_verified'])
"""),
        nbf.v4.new_markdown_cell("""## 7. 当前能说什么，下一步做什么

目前只有两个基线及 ItemCF 内部调参。验证结果支持在该随机留出与候选协议下继续以个性化 CF 为参照，不代表测试或线上提升。收缩配置的覆盖率变化值得一起报告，不能只选好看的准确率。

接下来以真实实验实现 BPR-MF、NeuMF、内容双塔并比较。标签特征必须仅由允许的训练信息构造，目录外标签先隔离。最终模型和超参冻结后才执行统一测试。没有真实结果的单元格不得用“预计提升”补齐。

**待成员理解/完成：**解释指标、复跑 Windows、分析低活跃用户、核查标签编码与覆盖、填写真实分工和 AI 使用。参见 docs/model_plan.md 与 docs/learning_and_contributions.md。

来源：[GroupLens HetRec](https://grouplens.org/datasets/hetrec-2011/)，原始 README 与 docs/research.md。"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
                         "language_info":{"name":"python","version":"3.12"}}
    path = ROOT / 'notebooks' / '01_data_and_baselines.ipynb'
    nbf.write(notebook, path)
    print(path)

if __name__ == '__main__':
    main()
