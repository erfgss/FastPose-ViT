"""Frozen-baseline ablations: validation selection only, followed by locked evaluation."""
import argparse, copy, csv, hashlib, json, random, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models import FastPoseViT
from src.datasets import SPEEDDataset
from src.device import get_device, synchronize
from src.geometry_decoder import encode_bbox_geometry, forward_pose
from src.utils import get_rotation_matrix, rotation_matrix_to_quaternion, get_absolute_orientation, bbox_relative_translation_to_translation
from src.metrics import translation_metric, relative_translation_metric, rotation_metric

p=argparse.ArgumentParser();p.add_argument('--output_dir',required=True);p.add_argument('--epochs',type=int,default=8)
args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
dev=get_device();assert dev.type=='npu';torch.set_num_threads(4)
opts=SimpleNamespace(no_pixel_augmentation=True,no_spatial_augmentation=True,domain_gap_pixel_augmentation=False,no_rotation_compensation=False)
seeds=[123,456,789]
configs=[dict(name='full',geometry=True,gates=True,loss='aligned',wd=1e-4),dict(name='no_geometry',geometry=False,gates=True,loss='aligned',wd=1e-4),dict(name='no_gates',geometry=True,gates=False,loss='aligned',wd=1e-4),dict(name='disentangled',geometry=True,gates=True,loss='disentangled',wd=1e-4),dict(name='no_weight_decay',geometry=True,gates=True,loss='aligned',wd=0.)]
base_path=Path('weights/vit_b_16.pth');base_sha=hashlib.file_digest(base_path.open('rb'),'sha256').hexdigest()
protocol={'configs':configs,'seeds':seeds,'epochs':args.epochs,'batch_size':64,'lr':5e-5,'eta_min':5e-6,'precision':'FP32','augmentation':'none; frozen cached features','selection':'minimum validation SPEED per seed; select configuration by mean best validation across all seeds; representative seed fixed to123','baseline_sha256':base_sha,'stress_conditions':['clean','dark_0.5','bright_1.5','bbox_shift_10pct','bbox_shift_20pct'],'cross_domain':'SPEED trained -> full SPEED+ real test, no target adaptation','test_selection':False}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def make_model(cfg,seed):
 seed_all(seed)
 m=FastPoseViT('vit_b_16_no_weights',1,768,3,6,merge_outputs=True,no_mlp=True,pose_decoder='geometry_dual_residual',decoder_interaction=cfg['gates']).to(dev)
 m.initialize_residual_from_baseline(str(base_path))
 # Matched shared initialization across architecture ablations.
 seed_all(seed+10000)
 canonical=FastPoseViT('vit_b_16_no_weights',1,768,3,6,merge_outputs=True,no_mlp=True,pose_decoder='geometry_dual_residual').decoder.state_dict()
 shared={k:v for k,v in canonical.items() if k in m.decoder.state_dict()}
 m.decoder.load_state_dict(shared,strict=True)
 if not cfg['geometry']:
  for param in m.decoder.geometry_encoder.parameters():param.data.zero_();param.requires_grad_(False)
 m.eval();return m
def dataset(split,plus=False,shift=0):
 root='/data/datasets/SPEED_PLUS_FIXED' if plus else '/data/datasets/SPEED_FIXED'
 ds=SPEEDDataset(root,split=split,rotation_format='matrix',bbox_json_path=root+('/speed_plus_bbox_annotations.json' if plus else '/speed_bbox_annotations.json'),args=opts,dataset_name='SPEED_PLUS' if plus else 'SPEED')
 if shift:
  # Shift actual crop AND geometry together; original images/annotation files unchanged.
  ds.bbox_data=copy.deepcopy(ds.bbox_data)
  for item in ds.data:
   fn=item['filename'];b=ds._process_bbox(fn);dx=(b[2]-b[0])*shift
   dx=min(max(dx,-b[0]),ds.camera.width-b[2]);ds.bbox_data[fn]=[b[0]+dx,b[1],b[2]+dx,b[3]]
 return ds
def cache_dataset(m,ds,brightness=1.):
 pieces={k:[] for k in ['tokens','geometry','translation','rotation','boxes']}
 mean=torch.tensor([.485,.456,.406],device=dev)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=dev)[None,:,None,None]
 with torch.no_grad():
  for x,t,r,b in DataLoader(ds,batch_size=32,num_workers=2,shuffle=False):
   x=x.to(dev)
   if brightness!=1:x=(((x*std+mean)*brightness).clamp(0,1)-mean)/std
   values=[m.backbone.forward_all_features(x),encode_bbox_geometry(b,ds.camera,dev),t.to(dev),r.to(dev),b.to(dev)]
   for key,value in zip(pieces,values):pieces[key].append(value)
 return {k:torch.cat(v) for k,v in pieces.items()}
def per_sample(y,d,sl,camera):
 pt=bbox_relative_translation_to_translation(y[:,:3],d['boxes'][sl],camera);gt=bbox_relative_translation_to_translation(d['translation'][sl],d['boxes'][sl],camera)
 pq=get_absolute_orientation(pt,rotation_matrix_to_quaternion(get_rotation_matrix(y[:,3:])));gq=get_absolute_orientation(gt,rotation_matrix_to_quaternion(d['rotation'][sl]))
 return torch.stack([translation_metric(pt,gt),relative_translation_metric(pt,gt),rotation_metric(pq,gq)],1)
def measure(m,d,camera,baseline=False):
 m.eval();rows=[]
 with torch.no_grad():
  for i in range(0,len(d['tokens']),64):
   sl=slice(i,i+64);tok=d['tokens'][sl]
   y=m.output(tok[:,:m.backbone.nb_class_tokens].flatten(1)) if baseline else m.forward_residual_features(tok,d['geometry'][sl])
   rows.append(per_sample(y,d,sl,camera).cpu())
 values=torch.cat(rows).numpy();assert np.isfinite(values).all()
 t,rel,r=values.mean(0,dtype=np.float64)
 return {'n':len(values),'translation_m':float(t),'rotation_deg':float(r*180/np.pi),'rotation_rad':float(r),'relative_translation':float(rel),'speed_score':float(rel+r)},values
def loss_fn(y,d,idx,camera,kind):
 target=d['translation'][idx]
 if kind=='aligned':
  pt=bbox_relative_translation_to_translation(y[:,:3],d['boxes'][idx],camera);gt=bbox_relative_translation_to_translation(target,d['boxes'][idx],camera)
  lt=torch.linalg.vector_norm(pt-gt,dim=1)/torch.linalg.vector_norm(gt,dim=1)
 else:
  # GDRN-inspired center/depth L1; not its CAD point-matching rotation loss.
  lt=(y[:,:2]-target[:,:2]).abs().sum(1)+(y[:,2]-target[:,2]).abs()
 R=get_rotation_matrix(y[:,3:]);lr=torch.sqrt(((R-d['rotation'][idx])**2).sum((1,2))/2+1e-8)
 return (lt+lr).mean()
anchor=make_model(configs[0],123);val_ds=dataset('val');camera=val_ds.camera
print("CACHE_START",flush=True)
val=cache_dataset(anchor,val_ds);print("VAL_CACHED",flush=True);train=cache_dataset(anchor,dataset('train'))
baseline,_=measure(anchor,val,camera,True);print('CACHE_READY BASELINE',json.dumps(baseline),flush=True)
results=[]
for cfg in configs:
 for seed in seeds:
  name=f"{cfg['name']}_seed{seed}";run=out/name;run.mkdir();m=make_model(cfg,seed)
  init,_=measure(m,val,camera);assert abs(init['speed_score']-baseline['speed_score'])<1e-9
  best=init;best_epoch=0;history=[]
  def save_delta():
   torch.save({'decoder':{k:v.detach().cpu() for k,v in m.decoder.state_dict().items()},'model_config':m.model_config,'experiment_config':cfg,'seed':seed,'epoch':best_epoch,'baseline_sha256':base_sha},run/'best_decoder.pt')
  save_delta();seed_all(seed+20000)
  opt=torch.optim.AdamW([v for v in m.decoder.parameters() if v.requires_grad],lr=5e-5,weight_decay=cfg['wd']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=5e-6)
  for epoch in range(1,args.epochs+1):
   m.train();perm=torch.randperm(len(train['tokens']),device=dev);total=0.;start=time.monotonic()
   for i in range(0,len(perm),64):
    idx=perm[i:i+64];opt.zero_grad(set_to_none=True);y=m.forward_residual_features(train['tokens'][idx],train['geometry'][idx]);loss=loss_fn(y,train,idx,camera,cfg['loss'])
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(m.decoder.parameters(),1.,error_if_nonfinite=True);opt.step();total+=loss.item()*len(idx)
   sched.step();met,_=measure(m,val,camera);row={'epoch':epoch,'train_loss':total/len(perm),'seconds':time.monotonic()-start,**met};history.append(row)
   if met['speed_score']<best['speed_score']:best=met;best_epoch=epoch;save_delta()
   (run/'history.json').write_text(json.dumps(history,indent=2));print(name,json.dumps(row),'BEST',best_epoch,flush=True)
  record={'name':name,'config':cfg['name'],'seed':seed,'best_epoch':best_epoch,**best};results.append(record);(out/'validation_runs.json').write_text(json.dumps(results,indent=2));del m,opt,sched
summary=[]
for cfg in configs:
 vals=np.array([r['speed_score'] for r in results if r['config']==cfg['name']]);summary.append({'config':cfg['name'],'mean_speed':float(vals.mean()),'std_speed':float(vals.std(ddof=1)),'change_vs_baseline_percent':float((vals.mean()/baseline['speed_score']-1)*100)})
winner=min(summary,key=lambda x:x['mean_speed'])['config']
selection={'baseline_validation':baseline,'summary':summary,'selected_config':winner,'representative_seed':123,'selection_completed_before_test':True};(out/'selection.json').write_text(json.dumps(selection,indent=2));print('LOCKED_SELECTION',json.dumps(selection),flush=True)
del train,val
def load_run(name):
 ck=torch.load(out/name/'best_decoder.pt',map_location='cpu',weights_only=True);assert ck['baseline_sha256']==base_sha
 m=make_model(ck['experiment_config'],ck['seed']);m.decoder.load_state_dict(ck['decoder'],strict=True);return m.eval()
selected=load_run(winner+'_seed123');selected.save(str(out/'selected_model.pth'))
full=load_run('full_seed123');models={'baseline':anchor,'full_seed123':full,'selected_seed123':selected}
eval_results=[]
for condition,brightness,shift in [('clean',1.,0),('dark_0.5',.5,0),('bright_1.5',1.5,0),('bbox_shift_10pct',1.,.1),('bbox_shift_20pct',1.,.2)]:
 ds=dataset('test',shift=shift);d=cache_dataset(anchor,ds,brightness)
 for name,m in models.items():
  metrics,rows=measure(m,d,ds.camera,name=='baseline');np.save(out/f'{condition}_{name}_errors.npy',rows)
  record={'condition':condition,'model':name,**metrics};eval_results.append(record);print('STRESS',json.dumps(record),flush=True)
 del d
(out/'stress_results.json').write_text(json.dumps(eval_results,indent=2))
# Independent real-domain evaluation, streamed batches to limit feature memory.
ds=dataset('test',plus=True);domain_rows={k:[] for k in models};domains=[]
for item in ds.data:domains.append('lightbox' if 'lightbox' in item['filename'] else 'sunlamp')
with torch.no_grad():
 for x,t,r,b in DataLoader(ds,batch_size=32,num_workers=2,shuffle=False):
  tok=anchor.backbone.forward_all_features(x.to(dev));geo=encode_bbox_geometry(b,ds.camera,dev);d={'translation':t.to(dev),'rotation':r.to(dev),'boxes':b.to(dev)}
  for name,m in models.items():
   y=anchor.output(tok[:,:1].flatten(1)) if name=='baseline' else m.forward_residual_features(tok,geo)
   domain_rows[name].append(per_sample(y,d,slice(None),ds.camera).cpu().numpy())
cross=[]
for name,parts in domain_rows.items():
 rows=np.concatenate(parts);assert np.isfinite(rows).all();np.save(out/f'speedplus_{name}_errors.npy',rows)
 for domain in ['all','lightbox','sunlamp']:
  mask=np.ones(len(rows),bool) if domain=='all' else np.array(domains)==domain;t,rel,r=rows[mask].mean(0,dtype=np.float64)
  cross.append({'model':name,'domain':domain,'n':int(mask.sum()),'translation_m':float(t),'rotation_deg':float(r*180/np.pi),'speed_score':float(rel+r)})
(out/'cross_domain.json').write_text(json.dumps(cross,indent=2));print('CROSS_DOMAIN',json.dumps(cross),flush=True)
# Network-only latency, NPU synchronized, data already resident, not end-to-end.
timings=[];x=torch.randn(1,3,224,224,device=dev);geo=torch.tensor([[0.,0.,.1,.1]],device=dev)
with torch.no_grad():
 for name,m in models.items():
  fn=(lambda:anchor.output(anchor.backbone(x))) if name=='baseline' else (lambda:m(x,geo))
  for _ in range(20):fn()
  synchronize(dev);ms=[]
  for _ in range(100):
   start=time.perf_counter();fn();synchronize(dev);ms.append((time.perf_counter()-start)*1000)
  timings.append({'model':name,'batch':1,'warmup':20,'iterations':100,'median_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95))})
(out/'latency.json').write_text(json.dumps(timings,indent=2));(out/'status.txt').write_text('completed\n');print('COMPLETED',flush=True)
