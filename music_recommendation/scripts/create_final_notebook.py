from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]

def main():
    cells=[
        nbf.v4.new_markdown_cell('''# 冻结决策与独立测试结果

本Notebook查看并核验已经完成的最终一次评估，**不再次训练、不重新读取测试标签、不重新给测试集评分**。所有输出来自真实保存结果及可重复的聚合检查。

主模型在看测试前固定为ItemCF（100邻居，shrinkage=10）。三种神经模型是预先指定的对照，不根据测试挑种子或换赢家。首次实验/干净复现实验由 scripts/freeze_final_decision.py 与 scripts/run_final_once.py 驱动；已完成目录有防止意外再次评估的状态保护。

历史随包输出来自IPython进程内逐格执行。本模板生成新的隔离验收副本，Windows原生Jupyter内核的执行结果与环境见 outputs/windows_native_verification/status.json。AI辅助使用和真实人工贡献需如实披露。'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import sys,json,subprocess
import pandas as pd
from IPython.display import display,Image
ROOT=Path.cwd().resolve()
if not (ROOT/'src/musicrec').is_dir():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from summarize_final import verify_and_summarize
OUT=ROOT/'outputs/final'
decision=json.loads((OUT/'frozen_decision.json').read_text(encoding="utf-8"))
state=json.loads((OUT/'evaluation_state.json').read_text(encoding="utf-8"))
print('主模型:',decision['selected_primary_model'])
display(pd.DataFrame(decision['neural_models']))
display(pd.Series(state,name='persisted_execution_state'))'''),
        nbf.v4.new_markdown_cell('''## 1. 冻结与时间顺序

训练轮数在测试前确定为既有3种子验证所选epoch的中位数：BPR4、NeuMF24、内容12。这一具体规则是在冻结时确定，不假称第一阶段已给出这些数字。

所有模型用相同83,571条 train+validation 已知历史重拟合。神经模型固定42/43/44三个种子，不早停、不选种子。热度、warm/cold与标签词表/IDF均按已知历史重算，测试仍不参与特征。

核验包括冻结证据哈希、11个checkpoint均先于测试阶段保存、固定轮数、逐用户指标与聚合、三种子均值/样本SD。顺序由代码和持久化记录佐证，不是系统级访问审计。'''),
        nbf.v4.new_code_cell('''comparison,groups=verify_and_summarize(ROOT)
display(comparison[['model','ndcg@10','ndcg@10_seed_sd','recall@10','precision@10','catalogue_coverage@10','frozen_primary']].round(6))
display(pd.Series(json.loads((OUT/'verification.json').read_text(encoding="utf-8"))))'''),
        nbf.v4.new_markdown_cell('''## 2. 同一独立测试协议

候选为全部17,632艺人目录减去用户 train+validation 历史。1,877测试用户、9,263目标；冷目标不删除。NDCG/Recall/Precision逐用户计算再宏平均。

误差棒是3个固定训练预算种子的样本标准差，不是置信区间；基线为确定性算法。验证期和测试期的训练信息量不同，因此不能把测试比验证更高直接当模型“又改进了一次”。'''),
        nbf.v4.new_code_cell('''for name in ['06_final_test_comparison.png','07_test_accuracy_coverage.png']:
    display(Image(filename=str(ROOT/'outputs/figures'/name)))
display(pd.Series(json.loads((OUT/'primary_paired_user_bootstrap.json').read_text(encoding="utf-8"))))'''),
        nbf.v4.new_markdown_cell('''## 3. 分组与内容覆盖

“含冷目标用户”组会同时含warm目标，不等于只对冷艺人算的召回率。低活跃组只有25位，避免扩大解释。标签时间不等于听歌时间，整个任务仍是随机偏好恢复，不是未来下一次听歌预测。'''),
        nbf.v4.new_code_cell('''display(groups.round(6))
display(pd.Series(json.loads((OUT/'refit_content_audit.json').read_text(encoding="utf-8")),name='known_history_content'))'''),
        nbf.v4.new_markdown_cell('''## 4. 预先固定的案例，不挑最好看的

用户2、3、4是在看测试前固定的前三个验证用户ID。下表展示真实Top-10以及是否命中测试；无命中的案例也保留，不能据此推断用户不喜欢推荐，只能说未命中这份有限观察记录。解释来自ItemCF中贡献最大的已知艺人邻居。'''),
        nbf.v4.new_code_cell('''cases=json.loads((OUT/'representative_cases.json').read_text(encoding="utf-8"))
for case in cases:
    print('匿名数据用户',case['userID'])
    print('已知历史示例:',[x['name'] for x in case['known_history_examples']])
    print('测试目标:',[x['name'] for x in case['test_artists']])
    display(pd.DataFrame(case['recommendations'])[['rank','artistID','name','score','test_hit']])'''),
        nbf.v4.new_markdown_cell('''## 5. 代码检查与诚实结论

最终选择来自验证，测试用于一次性报告。ItemCF在这份小规模数据、固定搜索预算和全目录协议下表现最好，不代表神经网络一般无用。内容双塔覆盖更广，但准确率较低；所有新物品缺少独立内容，不能宣称解决冷启动。

Windows实机与原生Jupyter内核的已执行范围见独立验收记录。线上效果、多随机拆分与更大调参空间仍需另外研究。AI生成/辅助部分和成员实际人工工作必须清楚区分。'''),
        nbf.v4.new_code_cell('''tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
print(tests.stdout+tests.stderr)
tests.check_returncode()
print('这里只运行代码/小型合成数据测试，没有再次评价保留测试集。')'''),
    ]
    n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (ipykernel)','name':'python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
    p=ROOT/'notebooks/03_frozen_final_test.ipynb';nbf.write(n,p);print(p)

if __name__=='__main__':main()
