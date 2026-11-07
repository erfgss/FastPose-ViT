"""Organize existing log access without moving/deleting files or interrupting writers."""
from pathlib import Path
import csv,json,re,os,datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/organized'
OUT.mkdir(parents=True,exist_ok=True)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()

def alias(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 target=os.path.relpath(source,dest.parent)
 if dest.is_symlink():
  assert dest.resolve()==source.resolve(),f'Existing link conflicts: {dest}'
 elif dest.exists():raise RuntimeError(f'Refusing to overwrite: {dest}')
 else:dest.symlink_to(target,target_is_directory=source.is_dir())
 assert dest.exists(),f'Broken alias: {dest}'

records=[]
for folder in ['experiments','mig_docs/working']:
 for source in sorted((ROOT/folder).rglob('*.log')):
  rel=source.relative_to(ROOT)
  if '.ipynb_checkpoints' in rel.parts:category='99_notebook_snapshots'
  elif 'attempt1_failed' in rel.parts:category='90_failed_attempts'
  elif folder=='experiments':category='01_full_training'
  elif 'speed_validation' in rel.parts:category='02_speed_baseline'
  elif 'dual_token' in rel.parts:category='03_dual_token_smoke'
  elif 'speedplus_preparation' in rel.parts:category='05_dataset_preparation'
  else:category='04_npu_validation'
  dest=OUT/category/rel
  alias(source,dest)
  records.append({'category':category,'source':str(source),'entry':str(dest),
                  'bytes_at_snapshot':source.stat().st_size,
                  'mtime_utc':datetime.datetime.fromtimestamp(source.stat().st_mtime,datetime.timezone.utc).isoformat()})
(OUT/'inventory.json').write_text(json.dumps({'snapshot_utc':now,'mode':'relative symbolic links; source files unchanged','logs':records},indent=2))
with (OUT/'inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

number=r'([-+0-9.eE]+)'
pattern=re.compile(
 r'(\d{4}-\d{2}-\d{2} [\d:,]+) - INFO - Epoch (\d+)/(\d+) [^\n]*\n'
 r'Train Loss: '+number+r' \(Translation: '+number+r', Rotation: '+number+r'\)\n'
 r'Val Loss: '+number+r' \(Relative Translation: '+number+r', Rotation \(Geodesic\): '+number+r'\)\n'
 r'Val Metrics: Trans: '+number+r', Rot: '+number+r', Rel Trans: '+number+r', SPEED Score: '+number+r'\n'
 r'Learning Rate: '+number)
rows=[];runs=[]
for run in sorted((ROOT/'experiments').glob('geometry_dual_token_full_*')):
 if not (run/'run.json').exists():continue
 summary={'run':run.name,'status':(run/'status.txt').read_text().strip(),'phases':{}}
 for phase,subdir,checkpoint in [('pretrain','pretrain_logs','pretrained.pth'),('finetune','finetune_logs','finetuned.pth')]:
  phase_rows=[]
  for source in sorted((run/subdir).glob('*.log')):
   for match in pattern.finditer(source.read_text(errors='replace')):
    v=match.groups();row={'run':run.name,'phase':phase,'timestamp_as_logged':v[0],
      'epoch':int(v[1]),'total_epochs':int(v[2]),'train_loss':float(v[3]),
      'train_translation_loss':float(v[4]),'train_rotation_loss':float(v[5]),
      'val_loss':float(v[6]),'val_relative_translation_loss':float(v[7]),
      'val_rotation_loss':float(v[8]),'translation_error_m':float(v[9]),
      'rotation_error_rad':float(v[10]),'relative_translation_error':float(v[11]),
      'speed_score':float(v[12]),'learning_rate':float(v[13]),'source':str(source)}
    phase_rows.append(row);rows.append(row)
  if phase_rows:
   best=min(phase_rows,key=lambda x:x['val_loss']);last=max(phase_rows,key=lambda x:x['epoch'])
   summary['phases'][phase]={'completed_epochs':last['epoch'],'planned_epochs':last['total_epochs'],
      'best_epoch_from_rounded_logs':best['epoch'],'best_val_loss_from_log':best['val_loss'],
      'latest_train_loss':last['train_loss'],'latest_val_loss':last['val_loss'],
      'best_checkpoint':str(run/checkpoint),'checkpoint_exists':(run/checkpoint).is_file()}
 runs.append(summary)
if rows:
 with (OUT/'training_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'training_summary.json').write_text(json.dumps({'snapshot_utc':now,'runs':runs},indent=2))
if runs:
 current=ROOT/'experiments'/runs[-1]['run'];link=OUT/'current_training'
 if link.is_symlink() and link.resolve()!=current:link.unlink()
 alias(current,link)

labels={'01_full_training':'完整预训练与联合训练（实时日志）','02_speed_baseline':'SPEED 基线模型与检测器评估',
'03_dual_token_smoke':'双 Token 结构短测与命令行验证','04_npu_validation':'NPU 迁移、数值一致性与清理回归',
'05_dataset_preparation':'SPEED+ 解压、格式整理与加载验证','90_failed_attempts':'失败启动记录，保留用于排错','99_notebook_snapshots':'Jupyter 自动生成的日志快照'}
text=['# FastPose-ViT 日志索引','',f'整理时间（UTC）：{now}','',
'日志按用途建立相对软链接，未搬移、删除或截断原文件，也未修改正在运行的训练。分类入口实时指向原日志；CSV/JSON 汇总是本次整理时的快照。','',
'## 快速入口','',
'- [当前训练目录](current_training/)：查看 pretrain.log、finetune.log、status.txt 和最优权重。',
'- [逐轮训练指标 CSV](training_metrics.csv)：预训练与联合训练的损失、SPEED 分数、学习率。',
'- [训练摘要 JSON](training_summary.json)：已完成轮数、历史最优验证损失与权重路径。',
'- [日志清单 CSV](inventory.csv)：每份日志的原路径、分类路径、大小与修改时间。','',
'## 分类目录','', '| 目录 | 用途 | 日志数 |','|---|---|---:|']
for category,label in labels.items():
 count=sum(r['category']==category for r in records)
 entry = f"[{category}/]({category}/)" if count else f"{category} (empty)"
 text.append(f"| {entry} | {label} | {count} |")
text+=['','## 当前训练快照','']
for run in runs:
 text += [f"实验：`{run['run']}`，状态：`{run['status']}`。",'']
 for phase,s in run['phases'].items():
  text.append(f"- {phase}：已完成 {s['completed_epochs']}/{s['planned_epochs']} 轮；日志中最优验证损失 {s['best_val_loss_from_log']:.4f}；权重 `{s['best_checkpoint']}`。")
text+=['','指标来自日志显示值，存在舍入；若不同轮次显示相同最优值，以训练程序实际保存的权重为准。未完成轮次不计入 CSV。','',
'## 查看与刷新','', '```bash','cd /data/FastPose-ViT',
'cat logs/organized/current_training/status.txt',
 'tail -f logs/organized/current_training/finetune.log',
'python scripts/organize_logs.py','```','',
'重复运行整理脚本会刷新清单和指标，不会启动或中断训练。保留 stdout 日志（含进度条）和程序生成的简洁日志；两者不是完全相同的副本。']
(OUT/'README.md').write_text('\n'.join(text)+'\n')
print(json.dumps({'output':str(OUT),'log_count':len(records),'total_log_bytes':sum(r['bytes_at_snapshot'] for r in records),'metric_rows':len(rows),'runs':runs},indent=2))
