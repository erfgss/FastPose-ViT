"""Offline reporting; never changes model selection."""
from pathlib import Path
import json,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=root/'results';sel=json.loads((data/'selection.json').read_text());runs=json.loads((data/'validation_runs.json').read_text());stress=json.loads((data/'stress_results.json').read_text());cross=json.loads((data/'cross_domain.json').read_text());latency=json.loads((data/'latency.json').read_text())
def table(rows,fields):
 return '\n'.join(['| '+' | '.join(title for key,title in fields)+' |','|'+'|'.join(['---']*len(fields))+'|']+['| '+' | '.join(f'{r[k]:.7f}' if isinstance(r[k],float) else str(r[k]) for k,title in fields)+' |' for r in rows])
for name,rows in [('validation_runs',runs),('validation_summary',sel['summary']),('stress_results',stress),('cross_domain',cross),('latency',latency)]:
 with (data/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
boot=[];rng=np.random.default_rng(20261107)
for condition in ['clean','dark_0.5','bright_1.5','bbox_shift_10pct','bbox_shift_20pct']:
 base=np.load(data/f'{condition}_baseline_errors.npy');b=base[:,1]+base[:,2]
 for model in ['full_seed123','selected_seed123']:
  a=np.load(data/f'{condition}_{model}_errors.npy');diff=(a[:,1]+a[:,2])-b
  samples=diff[rng.integers(0,len(diff),size=(3000,len(diff)))].mean(1);lo,hi=np.quantile(samples,[.025,.975])
  boot.append({'condition':condition,'model':model,'mean_score_difference':float(diff.mean()),'ci95_lower':float(lo),'ci95_upper':float(hi),'translation_p95_m':float(np.quantile(a[:,0],.95)),'rotation_p95_deg':float(np.quantile(a[:,2]*180/np.pi,.95))})
(data/'paired_bootstrap.json').write_text(json.dumps(boot,indent=2))
labels={'full':'Full','no_geometry':'No geometry','no_gates':'No gates','disentangled':'Split translation loss','no_weight_decay':'No weight decay'}
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
s=sel['summary'];base=sel['baseline_validation']['speed_score'];x=np.arange(len(s));delta=[100*(r['mean_speed']/base-1) for r in s];err=[100*r['std_speed']/base for r in s]
axs[0,0].bar(x,delta,yerr=err,capsize=4,color='#3378aa');axs[0,0].axhline(0,color='black',linewidth=.8);axs[0,0].set_xticks(x,[labels[r['config']] for r in s],rotation=20,ha='right');axs[0,0].set_ylabel('Validation score change vs baseline (%)');axs[0,0].set_title('Three seeds: mean +/- sample SD (lower is better)')
conditions=list(dict.fromkeys(r['condition'] for r in stress))
for name in ['baseline','full_seed123','selected_seed123']:
 rows=[r for r in stress if r['model']==name];axs[0,1].plot(range(len(rows)),[r['speed_score'] for r in rows],marker='o',label=name)
axs[0,1].set_xticks(range(len(conditions)),conditions,rotation=20,ha='right');axs[0,1].set_yscale('log');axs[0,1].set_ylabel('SPEED score (log scale)');axs[0,1].set_title('SPEED robustness: all 1,200 holdout images');axs[0,1].legend(fontsize=8)
for j,name in enumerate(['baseline','full_seed123','selected_seed123']):
 rows=[r for r in cross if r['model']==name and r['domain']!='all'];axs[1,0].bar(np.arange(2)+j*.25,[r['speed_score'] for r in rows],width=.24,label=name)
axs[1,0].set_xticks(np.arange(2)+.25,['Lightbox (6,740)','Sunlamp (2,791)']);axs[1,0].set_ylabel('Unthresholded SPEED score');axs[1,0].set_title('SPEED -> SPEED+ real, zero adaptation');axs[1,0].legend(fontsize=8)
axs[1,1].bar([r['model'] for r in latency],[r['median_ms'] for r in latency],color='#d58d39');axs[1,1].set_ylabel('Median milliseconds');axs[1,1].set_title('NPU FP32 network only, batch 1');axs[1,1].tick_params(axis='x',rotation=15)
fig.suptitle('Literature-guided FastPose-ViT experiments',fontsize=17);fig.savefig(root/'experiment_results.png',dpi=160);plt.close(fig)
winner=sel['selected_config'];full=next(r for r in s if r['config']=='full');best=next(r for r in s if r['config']==winner)
text=f'''# FastPose-ViT 文献驱动实验报告

已完成 5 配置 × 3 随机种子 × 8 轮 = 15 次冻结分支训练，以及5种SPEED测试条件、SPEED+两个真实域和NPU延迟测试。验证集选择配置为 **{winner}**，代表模型种子预先固定123。配置/种子/测试条件均在测试前写入 protocol.json。

## 验证集结果（越低越好）

原始基线验证 SPEED：{base:.9f}。每次选择 epoch0–8 的最优验证权重；表中标准差是3个随机种子间的样本标准差，不能当作严格显著性检验。

{table(s,[('config','配置'),('mean_speed','平均SPEED'),('std_speed','种子标准差'),('change_vs_baseline_percent','相对基线变化%')])}

完整双 Token 平均相对变化 {full['change_vs_baseline_percent']:.3f}%；选定配置平均相对变化 {best['change_vs_baseline_percent']:.3f}%。不要把所有收益归因于几何或门控，应分别比较 full 与对应消融。三个种子只提供初步稳定性证据。冻结骨干/回归头保留原基线能力，因此整体接近原基线也不自动证明新增模块有效。

## SPEED 内部测试与鲁棒性

每个条件1200张，标注框。brightness只作图像强度变换；bbox_shift为实际裁剪框向右平移，并同步更新几何条件及相对标签。selected_seed123是验证选定配置；若配置恰为full，两条结果将相同，不代表独立实验。

{table(stress,[('condition','条件'),('model','模型'),('translation_m','平移m'),('rotation_deg','旋转度'),('speed_score','SPEED')])}

逐图配对bootstrap使用3000次重采样，差值定义为新模型减基线，负值较好；区间只描述固定模型/固定样本分布的不确定性，不包含训练种子和历史测试集选择偏差。

{table(boot,[('condition','条件'),('model','模型'),('mean_score_difference','SPEED差值'),('ci95_lower','95%下界'),('ci95_upper','95%上界')])}

## SPEED→SPEED+ 跨域（没有域适应训练）

使用SPEED训练的模型直接评估SPEED+全部9531张真实图像，使用该数据集相机内参。报告与SPEED相同的未阈值化相对平移+角误差，用于本实验内比较，不声称等同官方SPEED+带精度阈值评分。没有使用这些真实域标签选择权重。

{table(cross,[('model','模型'),('domain','域'),('n','样本数'),('translation_m','平移m'),('rotation_deg','旋转度'),('speed_score','未阈值化SPEED')])}

## NPU网络延迟

输入已在设备上，FP32，batch1，20次预热，100次逐次同步计时；不含检测器、读图、裁剪、坐标后处理。共用机器的单次计时可能受调度影响。

{table(latency,[('model','模型'),('median_ms','中位ms'),('p95_ms','p95 ms')])}

## 文献对应关系与复现边界

见 [实验计划](EXPERIMENT_PLAN.md) 的逐篇页码对应，以及 [来源清单](sources.json)。FastPose-ViT 的权重衰减、框质量与域差异实验启发本轮对照；GDRNPP启发分离中心/深度监督，未复现其CAD点匹配或密集几何；UDA论文和光照论文启发独立跨域/强度压力测试，未实现UDA或物理光照仿真。tesi.pdf的讨论用于同时检查计算代价。

本轮各组仅8轮、冻结骨干、无增强，不能替代论文完整收敛训练。基线权重训练覆盖范围未知，可能与内部留出集重叠；测试集此前已观察，因此所有结果属于探索性证据。不能据此宣称论文级创新或对官方盲测的提升。

## 文件

服务器根目录：`/data/FastPose-ViT/experiments/literature_suite_20261107/run02`。
完整选定权重：`selected_model.pth`；15组最佳增量权重：`配置_seed编号/best_decoder.pt`（依赖原baseline，哈希在文件中）。训练日志为同级 `run02.log`；每组history.json记录逐轮结果。保存了脚本快照与固定protocol。
本地 [验证汇总CSV](results/validation_summary.csv)、[全部运行CSV](results/validation_runs.csv)、[压力测试CSV](results/stress_results.csv)、[跨域CSV](results/cross_domain.csv)、[延迟CSV](results/latency.csv)。
'''
if (data/'weight_decay_diagnosis.json').exists():
 diagnosis=json.loads((data/'weight_decay_diagnosis.json').read_text())
 text+='\n## 权重衰减数值检查\n\nLR=5e-5、WD=1e-4 时，FP32 的衰减乘数 1−LR×WD 舍入为1.0。三个种子的 full 与 no_weight_decay 最佳解码器参数逐元素完全相同。因此这里不能推断“权重衰减无用”，只能说这档设置没有产生有效衰减。WD=0.01 的乘数为0.999999523，已补充三种子验证对照；补充实验不修改已锁定的测试模型。\n'
supp_path=data/'supplement/supplement_summary.json'
if supp_path.exists():
 supp=json.loads(supp_path.read_text());text+='\n'+table(supp['summary'],[('config','补充配置'),('mean_speed','平均验证SPEED'),('std_speed','种子标准差'),('change_vs_baseline_percent','相对基线变化%')])+'\n'
 text=text.replace('已完成 5 配置 × 3 随机种子 × 8 轮 = 15 次冻结分支训练','已完成主实验 5 配置 × 3 随机种子 × 8 轮，以及3次有效权重衰减补充对照，共18次冻结分支训练')
clean_base=next(r for r in stress if r['condition']=='clean' and r['model']=='baseline')
clean_selected=next(r for r in stress if r['condition']=='clean' and r['model']=='selected_seed123')
gain=100*(1-clean_selected['speed_score']/clean_base['speed_score'])
interpretation=f'''## 结果解读

- 几何条件的平均收益很小：完整模型比去掉几何条件的验证SPEED低约0.092%；门控对应差异约0.025%。样本量和训练预算不足以宣称稳定创新。
- 分离监督在验证平均上略优，但相对full差异小于种子标准差；不要过度解释这项排序。固定seed123的干净内部测试相对基线改善约{gain:.3f}%，主要来自平移，旋转基本不变。
- 亮度×1.5和框偏移20%时，新增分支反而略差；并未建立普遍鲁棒性提升。框偏移20%使整体误差达到干净条件约5.7倍。
- SPEED+真实域平均旋转误差约65.36°，三个模型几乎相同。冻结SPEED骨干、仅做微小残差校准，不能解决这个域差异。后续更有价值的方向是单独以SPEED+ synthetic训练并设置真实域独立验证协议，以及检查光照增强/框扰动训练，而不是直接增加Token数量。
- 新模型NPU网络推理约4.45ms，基线约3.27ms，约增加36%。目前精度收益与计算开销的交换并不理想。
- 小角度FP32 acos计算对数值误差敏感；本轮缓存批次与此前batch8评估存在约1e-6量级分数差异。应只在同一评估协议内比较微小提升；不能将历史分数最后几位的变化归因于学习收益。

'''
text=text.replace('## 验证集结果（越低越好）',interpretation+'## 验证集结果（越低越好）')
text+='\n## 权重恢复与复现\n\n重新运行主实验：`PYTHONPATH=. FASTPOSE_DEVICE=npu:0 OMP_NUM_THREADS=4 .venv-npu/bin/python -u scripts/run_literature_suite.py --output_dir experiments/literature_suite_new --epochs 8`（必须新建输出目录）。补充对照入口为 `scripts/run_weight_decay_control.py`，参数相同。\n\n恢复任意增量权重：`PYTHONPATH=. .venv-npu/bin/python scripts/load_literature_checkpoint.py experiments/literature_suite_20261107/run02/full_seed123/best_decoder.pt --output experiments/full_seed123_export.pth`。脚本先验证原基线SHA256，拒绝覆盖已有输出。完整权重可通过 `FastPoseViT.from_checkpoint` 加载，再用 `forward_pose` 传入图像、框与相机内参。\n'
(root/'REPORT.md').write_text(text,encoding='utf8');print('REPORT_WRITTEN',winner)
