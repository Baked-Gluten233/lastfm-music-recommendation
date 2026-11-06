from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]

def main():
    cells=[
      nbf.v4.new_markdown_cell('''# 三个后续模型：真实验证与失败分析

阶段二已实现 BPR-MF、NeuMF、内容双塔。Notebook 默认**重新评分随包真实训练权重**，不会冒称重新训练；设 RETRAIN_ALL=True 可重新训练全部网格与重复种子。训练代码与原始结果均随包提供。

此页保留阶段二开发时的记录。随后已另行冻结并完成独立测试，最新结果见03_frozen_final_test.ipynb；本页的测试封存表述是历史开发状态。

测试集仍封存。交付执行方式是 IPython 进程内顺序运行真实代码格，原生 Jupyter 内核与 Windows 实机运行未验证。AI 辅助实现和解释，人工学习/验证/贡献必须如实补充。'''),
      nbf.v4.new_code_cell('''from pathlib import Path
import sys,json
import pandas as pd
import torch
from IPython.display import display,Image
ROOT=Path.cwd().resolve()
if not (ROOT/'src/musicrec').is_dir(): ROOT=ROOT.parent
assert (ROOT/'src/musicrec').is_dir()
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from musicrec.phase2 import run_phase2
from replay_phase2 import replay
from summarize_phase2 import summarize
print('Python:',sys.version)
print('PyTorch:',torch.__version__)
print('CPU only; no test evaluation')'''),
      nbf.v4.new_markdown_cell('''## 1. 全部训练与快速复核是两个明确模式

默认读取已经真实完成的12次训练记录：每族2个学习率、seed42调参，选中学习率再用43/44训练。后面会从权重重新计算9次选中配置的验证指标。想从头复现训练，把下格改为True；会覆盖派生结果，耗时取决于CPU。'''),
      nbf.v4.new_code_cell('''RETRAIN_ALL=False
if RETRAIN_ALL:
    runs=run_phase2(ROOT,repeat_seeds=True,max_epochs=24,every=4)
else:
    assert (ROOT/'outputs/phase2/status.json').is_file(), '没有已完成结果，请设 RETRAIN_ALL=True'
    runs=pd.read_csv(ROOT/'outputs/phase2/all_runs.csv')
    print('此格没有重新训练，读取已保存的真实训练记录。下一节会独立重新评分。')
display(runs[['family','lr','seed','selected_epoch','ndcg@10','recall@10','training_seconds','scoring_seconds']].round(6))'''),
      nbf.v4.new_markdown_cell('''## 2. 为什么是三个实质不同的模型

- BPR-MF：用户和艺人低秩向量的点积，直接优化正例分数高于采样负例。
- NeuMF：分开的 GMF 与 MLP embedding；前者逐维相乘，后者非线性变换拼接向量，最后融合输出 logit。不是仅给 MF 换损失。
- 内容双塔：独立用户编码与标签内容编码，以归一化向量点积检索。艺人塔没有 artistID embedding，而是训练允许标签的稀疏线性映射加非线性。

损失、学习率、正则、epoch与选择规则见实际配置。改变损失/特征与结构同时发生，所以不能把任何差异归因于某一个因素。'''),
      nbf.v4.new_code_cell('''protocol=json.loads((ROOT/'outputs/phase2/protocol.json').read_text())
display(pd.Series(protocol,name='recorded_protocol'))
content=json.loads((ROOT/'outputs/phase2/content_audit.json').read_text())
display(pd.Series(content,name='training_only_content_audit'))'''),
      nbf.v4.new_markdown_cell('''## 3. 防泄漏与内容局限

只有训练 user–artist 对的标签可进入特征；词表、df、IDF全由训练数据确定。用数值tagID，不猜测文本编码。训练目标不通过用户历史池化进入用户塔，因为用户塔只读取ID。

约三分之一目录有合法内容；缺失内容艺人共享unknown特征，不能互相区分。所有验证冷艺人缺内容，因此此模型不能宣称解决新艺人冷启动。标签日期不是听歌时间，本项目没有时序预测结论。'''),
      nbf.v4.new_markdown_cell('''## 4. 五模型的同协议比较

后续模型只汇总选中学习率的seed42/43/44，每个种子都用验证选择停止点。标准差不是独立测试置信区间，也不是固定epoch的纯初始化方差。基线是确定性算法，显示一个数值。'''),
      nbf.v4.new_code_cell('''comparison=summarize(ROOT)
display(comparison[['model','ndcg@10','ndcg@10_seed_sd','recall@10','catalogue_coverage@10','seeds']].round(6))
for name in ['04_learning_curves.png','05_validation_comparison.png']:
    display(Image(filename=str(ROOT/'outputs/figures'/name)))'''),
      nbf.v4.new_markdown_cell('''## 5. 从真实权重重新评分，而非只读表

下格重新构建训练特征、读取安全NPZ参数、按全部验证用户与全目录候选重新排名。核对9个选中配置checkpoint的NDCG、Recall、Precision和覆盖率，误差超过1e−7则报错。没有读取test.csv。'''),
      nbf.v4.new_code_cell('''replay_checks=replay(ROOT)
display(replay_checks)
assert replay_checks.passed.all()'''),
      nbf.v4.new_markdown_cell('''## 6. 分组与成本

有冷目标用户的分组指标不是“只对冷艺人计算的召回率”。低活跃组样本少，不扩大解释。CPU训练耗时、完整评分耗时和验证搜索耗时是不同概念；这里记录的Linux秒数不是Windows性能承诺。'''),
      nbf.v4.new_code_cell('''groups=pd.read_csv(ROOT/'outputs/phase2/selected_user_groups.csv')
display(groups.groupby(['model','group'])[['ndcg@10','recall@10']].mean().round(6))
display(pd.read_csv(ROOT/'outputs/phase2/selected_seed_summary.csv')[['model','parameters','mean_training_seconds','mean_scoring_seconds','mean_search_seconds']].round(3))
print('测试指标仍未计算。')'''),
      nbf.v4.new_markdown_cell('''## 7. 检查实现，而不是只信曲线

测试覆盖训练对标签过滤、词表门槛、unknown内容、两塔可分解性、NeuMF独立分支梯度、BPR公式、负例采样、冷分数、checkpoint同种子复现及阶段一全部指标协议。'''),
      nbf.v4.new_code_cell('''import subprocess
test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
print(test.stdout+test.stderr)
assert test.returncode==0'''),
      nbf.v4.new_markdown_cell('''## 8. 如何写真实的迭代结论

把“提出假设”和“证据支持”分开：低秩/非线性/内容表示有理论动机，不代表一定超过强邻域基线。当前有限预算下如ItemCF仍领先，应如实保留它，而不是为了最后一个模型好看而改测试规则。

改进方向应先诊断训练目标与Top-K的差距、正则和负采样的敏感性、内容覆盖及过拟合，不从一次失败断言模型家族无效。进一步调参属于内部优化，不新增模型计数。

正式报告/PPT和独立测试尚未完成。先冻结验证决策及最终训练轮数规则，才执行一次统一测试。成员还需Windows复跑、理解方法、填写真实贡献与AI披露。详见 docs/phase2_findings.md、phase2_method.md。'''),
    ]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    p=ROOT/'notebooks/02_model_iterations.ipynb';nbf.write(notebook,p);print(p)

if __name__=='__main__':main()
