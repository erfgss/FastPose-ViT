"""Paper-style figures from recorded runs only; no interpolated/made-up experiments."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results';OUT=ROOT/'ablation_figures';OUT.mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf8'))
selection=load(DATA/'selection.json');baseline=selection['baseline_validation']
runs=load(DATA/'validation_runs.json')+load(DATA/'supplement/validation_runs.json')
specs=[('B0','full','Full residual dual Token','yes','yes','Aligned',1e-4),('B1','no_geometry','B0: without geometry','no','yes','Aligned',1e-4),('B2','no_gates','B0: without task gates','yes','no','Aligned',1e-4),('C0','disentangled','B0: split translation loss','yes','yes','Split',1e-4),('D0','no_weight_decay','B0: weight decay = 0','yes','yes','Aligned',0),('D1','weight_decay_0p01','B0: weight decay = 0.01','yes','yes','Aligned',.01)]
rows=[dict(row='A0',config='baseline',method='Original FastPose-ViT',geometry='-',gates='-',loss='-',wd='-',n_seeds=1,translation_cm=baseline['translation_m']*100,translation_cm_sd=None,rotation_deg=baseline['rotation_deg'],rotation_deg_sd=None,score_x1000=baseline['speed_score']*1000,score_x1000_sd=None,change_percent=0)]
curves={}
for row,cfg,method,geo,gate,loss,wd in specs:
 group=[r for r in runs if r['config']==cfg];assert len(group)==3
 rec=dict(row=row,config=cfg,method=method,geometry=geo,gates=gate,loss=loss,wd=wd,n_seeds=3)
 for key,src,scale in [('translation_cm','translation_m',100),('rotation_deg','rotation_deg',1),('score_x1000','speed_score',1000)]:
  values=np.array([r[src]*scale for r in group]);rec[key]=values.mean();rec[key+'_sd']=values.std(ddof=1)
 rec['change_percent']=100*(rec['score_x1000']/1000/baseline['speed_score']-1);rows.append(rec)
 histories=[]
 for r in group:
  folder=DATA/'supplement' if cfg=='weight_decay_0p01' else DATA
  h=load(folder/r['name']/'history.json');assert len(h)==8
  assert np.isclose(min([baseline['speed_score']]+[x['speed_score'] for x in h]),r['speed_score'],rtol=0,atol=1e-12)
  histories.append(h)
 curves[cfg]=histories
def write_csv(name,records):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
write_csv('validation_ablation.csv',rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
colors={'full':'#0072B2','no_geometry':'#D55E00','no_gates':'#CC79A7','disentangled':'#009E73','no_weight_decay':'#777777','weight_decay_0p01':'#E69F00'}
short={'full':'B0 Full','no_geometry':'B1 No geometry','no_gates':'B2 No gates','disentangled':'C0 Split loss','no_weight_decay':'D0 WD=0','weight_decay_0p01':'D1 WD=0.01'}
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def curve(ax,key,scale=1,configs=None):
 for cfg in configs or list(curves):
  y=np.array([[h[key] for h in hist] for hist in curves[cfg]])*scale
  mu=y.mean(0);sd=y.std(0,ddof=1);x=np.arange(1,9)
  ax.plot(x,mu,label=short[cfg],color=colors[cfg],marker='o',ms=3,lw=1.4,ls='--' if cfg=='no_weight_decay' else '-')
  ax.fill_between(x,mu-sd,mu+sd,color=colors[cfg],alpha=.10)
 ax.set_xticks(range(1,9));ax.grid(alpha=.20);ax.set_xlabel('Training epoch')
def fmt(row,key,digits):
 val=f'{row[key]:.{digits}f}'
 return val if row.get(key+'_sd') is None else val+f" ± {row[key+'_sd']:.{digits}f}"

fig=plt.figure(figsize=(18,8.0));grid=GridSpec(2,2,figure=fig,width_ratios=[1,2.8],wspace=.20,hspace=.37)
ax=fig.add_subplot(grid[0,0]);curve(ax,'speed_score',1000,configs=['full','no_geometry','no_gates','disentangled']);ax.axhline(baseline['speed_score']*1000,color='black',ls=':',lw=1,label='A0 Baseline');ax.set_ylabel('Validation SPEED score × 10³ ↓');ax.set_title('(a) Convergence: mean ± SD, 3 seeds');ax.legend(fontsize=8,loc='upper right')
ax=fig.add_subplot(grid[1,0]);
for cfg in ['full','no_geometry','no_gates','disentangled']:
 y=np.array([[100*(h['speed_score']/baseline['speed_score']-1) for h in hist] for hist in curves[cfg]])
 mu=y.mean(0);sd=y.std(0,ddof=1);ax.plot(range(1,9),mu,color=colors[cfg],marker='o',ms=3,label=short[cfg]);ax.fill_between(range(1,9),mu-sd,mu+sd,color=colors[cfg],alpha=.10)
ax.axhline(0,color='black',ls=':',lw=1);ax.set_xticks(range(1,9));ax.set_xlabel('Training epoch');ax.set_ylabel('Score change vs A0 (%) ↓');ax.grid(alpha=.2);ax.set_title('(b) Relative validation improvement')
ax=fig.add_subplot(grid[:,1]);ax.axis('off');ax.set_title('(c) Validation ablation: best checkpoint per seed',loc='left',fontsize=14,pad=25)
headers=['Row','Method / single change','Geom.','Gates','Loss','WD','Translation\n(cm) ↓','Rotation\n(deg) ↓','SPEED × 10³ ↓','Δ vs A0\n(%) ↓']
cells=[]
for r in rows:
 cells.append([r['row'],r['method'],r['geometry'],r['gates'],r['loss'],str(r['wd']),fmt(r,'translation_cm',4),fmt(r,'rotation_deg',5),fmt(r,'score_x1000',6),f"{r['change_percent']:+.3f}"])
tbl=ax.table(cellText=cells,colLabels=headers,colWidths=[.045,.255,.05,.05,.06,.055,.125,.125,.15,.07],bbox=[-.01,.30,1.04,.62],cellLoc='center')
tbl.auto_set_font_size(False);tbl.set_fontsize(8.2)
for (r,c),cell in tbl.get_celld().items():
 cell.set_edgecolor('#b9b9b9');cell.set_linewidth(.35)
 if r==0:cell.set_facecolor('#e7eef5');cell.set_text_props(weight='bold');cell.set_height(cell.get_height()*1.3)
 elif c==1:cell.set_text_props(ha='left')
 if r==5:cell.set_facecolor('#eaf4ef')
for col,key in [(6,'translation_cm'),(7,'rotation_deg'),(8,'score_x1000')]:
 best=min(range(len(rows)),key=lambda i:rows[i][key]);tbl[(best+1,col)].set_text_props(weight='bold')
notes=('A0: fixed pretrained baseline (no seed SD). B0–D1: mean ± sample SD over seeds 123 / 456 / 789.\n'
       'Same 9,600 training / 1,200 validation images; 8 epochs; frozen backbone + original pose head.\n'
       'Aligned: decoded relative translation + rotation chordal loss. Split: crop-space xy/z L1 + same rotation loss.\n'
       'D1: supplementary control, excluded from the already-locked main model selection.\n'
       'D0 exactly matches B0: LR × WD = 5e-9 rounds away in FP32; this is not evidence that WD never matters.\n'
       'Curves show actual epoch metrics, not running minima. This is not a Token-count parameter sweep.\n'
       'Small differences are exploratory; inherited baseline training coverage is unknown.')
ax.text(0,.25,notes,va='top',fontsize=9,linespacing=1.65,transform=ax.transAxes)
fig.suptitle('FastPose-ViT: geometry-conditioned residual dual-Token ablation',fontsize=17,y=.99)
save(fig,'ablation_overview')

# Separate vector figures for manuscript layout.
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,key,scale,title in zip(axs,['speed_score','translation_m','rotation_deg'],[1000,100,1],['Validation SPEED × 10³ ↓','Translation error (cm) ↓','Rotation error (deg) ↓']):
 curve(ax,key,scale);ax.set_ylabel(title)
axs[0].legend(fontsize=8);fig.suptitle('Validation trajectories: three seeds, mean ± sample SD');save(fig,'validation_curves')

# Accuracy-at-threshold table derives only from the three models with saved test errors.
acc=[]
for name,label in [('baseline','A0 Baseline'),('full_seed123','B0 Full, seed123'),('selected_seed123','C0 Split loss, seed123')]:
 x=np.load(DATA/f'clean_{name}_errors.npy');assert x.shape==(1200,3)
 t=x[:,0]<.02;r=x[:,2]<(2*np.pi/180)
 acc.append({'model':label,'n':len(x),'translation_cm':float(x[:,0].mean(dtype=np.float64)*100),'rotation_deg':float(x[:,2].mean(dtype=np.float64)*180/np.pi),'speed_score':float((x[:,1].astype(np.float64)+x[:,2]).mean()),'2deg_2cm_percent':float((t&r).mean()*100),'2deg_percent':float(r.mean()*100),'2cm_percent':float(t.mean()*100)})
write_csv('test_threshold_accuracy.csv',acc)
fig,ax=plt.subplots(figsize=(13,2.8));ax.axis('off')
cells=[[r['model'],str(r['n']),f"{r['translation_cm']:.4f}",f"{r['rotation_deg']:.5f}",f"{r['speed_score']:.8f}",f"{r['2deg_2cm_percent']:.2f}",f"{r['2deg_percent']:.2f}",f"{r['2cm_percent']:.2f}"] for r in acc]
tbl=ax.table(cellText=cells,colLabels=['Model','N','ET (cm) ↓','ER (deg) ↓','SPEED ↓','2° & 2 cm (%) ↑','2° (%) ↑','2 cm (%) ↑'],colWidths=[.23,.05,.11,.11,.13,.15,.11,.11],bbox=[0,.22,1,.65],cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(10)
for (r,c),cell in tbl.get_celld().items():
 cell.set_edgecolor('#aaaaaa');cell.set_linewidth(.4)
 if r==0:cell.set_facecolor('#e7eef5');cell.set_text_props(weight='bold')
ax.set_title('SPEED internal holdout: threshold accuracy (fixed seed 123)',fontsize=14);ax.text(0,.06,'1,200 images; annotated boxes; same cached FP32 protocol. Strict thresholds (<). ADD(-S) was not evaluated.',fontsize=9,transform=ax.transAxes)
save(fig,'test_accuracy_table')

stress=load(DATA/'stress_results.json');fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for name,label,color in [('baseline','A0 Baseline','#444444'),('full_seed123','B0 Full','#0072B2'),('selected_seed123','C0 Split loss','#009E73')]:
 for ax,conds,xx in [(axs[0],['clean','bbox_shift_10pct','bbox_shift_20pct'],[0,10,20]),(axs[1],['dark_0.5','clean','bright_1.5'],[.5,1,1.5])]:
  yy=[next(r['speed_score'] for r in stress if r['model']==name and r['condition']==c) for c in conds]
  ax.plot(xx,yy,marker='o',lw=1.6,label=label,color=color,ls='--' if name=='baseline' else '-')
for ax in axs:ax.set_ylabel('SPEED score ↓');ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_xlabel('Rightward bbox shift (% of bbox width)');axs[0].set_xticks([0,10,20]);axs[0].set_title('Crop/geometry robustness')
axs[1].set_xlabel('Image brightness multiplier');axs[1].set_xticks([.5,1,1.5]);axs[1].set_title('Photometric robustness')
fig.suptitle('Fixed-model sensitivity tests: all 1,200 holdout images');save(fig,'robustness_curves')

def markdown_table(headers,cells):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(row)+' |' for row in cells])
md='# 双 Token 消融实验表与曲线\n\n'
md+='主表使用验证集、3个随机种子的最佳轮次结果；测试表只列实际有逐图预测的3个模型，不能把它当作全部结构的测试消融。所有误差指标越低越好，阈值准确率越高越好。\n\n'
md+='![消融总览](ablation_overview.png)\n\n'
md+=markdown_table(['编号','配置','平移cm（均值±SD）','旋转°（均值±SD）','SPEED×1000（均值±SD）','相对基线变化%'],[[r['row'],r['method'],fmt(r,'translation_cm',4),fmt(r,'rotation_deg',5),fmt(r,'score_x1000',6),f"{r['change_percent']:+.3f}"] for r in rows])
md+='\n\nA0为固定基线，无跨种子标准差。B0–D1各3种子、每次8轮。C0按主实验验证均值选出；D1是后补数值对照，不参与已锁定模型的选择。B0和D0参数完全相同是当前LR×WD在FP32下舍入造成，不是一般性结论。\n\n'
md+='## 测试阈值准确率\n\n![测试准确率表](test_accuracy_table.png)\n\n'
md+=markdown_table(['模型','2°且2cm准确率%','2°准确率%','2cm准确率%'],[[r['model'],f"{r['2deg_2cm_percent']:.2f}",f"{r['2deg_percent']:.2f}",f"{r['2cm_percent']:.2f}"] for r in acc])
md+='\n\n阈值采用严格小于。没有CAD点匹配评估结果，因此不添加ADD(-S)列；也不把不同量纲的误差混成MEAN。\n\n## 曲线与使用说明\n\n- validation_curves：横轴是真实训练轮次，线为3种子平均，阴影为样本标准差；曲线未平滑，未改成累计最优。\n- robustness_curves：框偏移0/10/20%与亮度0.5/1/1.5，是固定模型的鲁棒性曲线，不是Token数量扫描。相邻测点间连线仅用于阅读。\n- ablation_overview：左侧两幅曲线与右侧主表组合，接近参考文献的版式；指标保持本项目实际口径。\n- PNG为高分辨率图片，SVG为可编辑矢量图；CSV保存精确数值；validation_ablation.tex可插入LaTeX（需booktabs）。\n\n原始数据：../results/validation_runs.json、../results/supplement/validation_runs.json、各组history.json及clean_*_errors.npy。\n\n本轮属于小预算冻结分支消融，非从头训练或完整收敛结果。小幅差异不能单独证明创新；已有基线的训练覆盖范围未知，内部测试集也不是官方盲测。\n'
(OUT/'README.md').write_text(md,encoding='utf8')
tex=['% Requires \\usepackage{booktabs}','\\begin{tabular}{llrrr}','\\toprule','Row & Method & $E_T$ (cm) $\\downarrow$ & $E_R$ (deg) $\\downarrow$ & SPEED $\\times 10^3$ $\\downarrow$ \\\\','\\midrule']
for r in rows:tex.append(' & '.join([r['row'],r['method'],fmt(r,'translation_cm',4).replace('±',r'$\pm$'),fmt(r,'rotation_deg',5).replace('±',r'$\pm$'),fmt(r,'score_x1000',6).replace('±',r'$\pm$')])+r' \\')
tex+=['\\bottomrule','\\end{tabular}','% A0 is fixed; remaining rows: mean +/- sample SD over three seeds.']
(OUT/'validation_ablation.tex').write_text('\n'.join(tex),encoding='utf8')
print('CREATED',OUT);print(json.dumps(acc,indent=2))
