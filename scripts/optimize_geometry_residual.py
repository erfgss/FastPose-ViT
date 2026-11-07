"""Validation-selected residual dual-token optimization. Never opens test.json."""
import argparse,copy,gc,json,math,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models import FastPoseViT
from src.datasets import SPEEDDataset
from src.device import get_device,synchronize
from src.geometry_decoder import encode_bbox_geometry
from src.utils import get_rotation_matrix,rotation_matrix_to_quaternion,get_absolute_orientation,bbox_relative_translation_to_translation
from src.metrics import translation_metric,relative_translation_metric,rotation_metric

p=argparse.ArgumentParser();p.add_argument('--output_dir',required=True);p.add_argument('--epochs',type=int,default=8);p.add_argument('--lr',type=float,default=0.00005);args=p.parse_args()
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.set_num_threads(4)
dev=get_device();assert dev.type=='npu'
opts=SimpleNamespace(no_pixel_augmentation=True,no_spatial_augmentation=True,domain_gap_pixel_augmentation=False,no_rotation_compensation=False)
model=FastPoseViT('vit_b_16_no_weights',1,768,3,6,merge_outputs=True,no_mlp=True,pose_decoder='geometry_dual_residual').to(dev)
model.initialize_residual_from_baseline('weights/vit_b_16.pth');model.eval()
cache={};start=time.monotonic()
for split in ['val','train']:
 ds=SPEEDDataset('/data/datasets/SPEED_FIXED',split=split,rotation_format='matrix',bbox_json_path='/data/datasets/SPEED_FIXED/speed_bbox_annotations.json',args=opts)
 loader=DataLoader(ds,batch_size=32,num_workers=2,shuffle=False)
 parts={k:[] for k in ['tokens','geometry','translation','rotation','boxes']}
 with torch.no_grad():
  for j,(x,t,r,b) in enumerate(loader):
   x=x.to(dev);tokens=model.backbone.forward_all_features(x)
   if j==0 and split=='val':
    base=model.output(model.backbone(x));fresh=model(x,encode_bbox_geometry(b,ds.camera,dev))
    torch.testing.assert_close(base,fresh,rtol=0,atol=0)
    print('INITIAL_BASELINE_EXACT_MATCH',flush=True)
   parts['tokens'].append(tokens);parts['geometry'].append(encode_bbox_geometry(b,ds.camera,dev))
   parts['translation'].append(t.to(dev));parts['rotation'].append(r.to(dev));parts['boxes'].append(b.to(dev))
   if j%50==0:print('CACHE',split,j*32,'/',len(ds),flush=True)
 cache[split]={k:torch.cat(v) for k,v in parts.items()};camera=ds.camera
 del parts,loader,ds
 print('CACHED',split,len(cache[split]['tokens']),flush=True)
print('CACHE_SECONDS',time.monotonic()-start,flush=True)

def metrics():
 model.eval();d=cache['val'];ts=[];rs=[];rels=[]
 with torch.no_grad():
  for i in range(0,len(d['tokens']),64):
   sl=slice(i,i+64);y=model.forward_residual_features(d['tokens'][sl],d['geometry'][sl]);b=d['boxes'][sl]
   pt=bbox_relative_translation_to_translation(y[:,:3],b,camera);gt=bbox_relative_translation_to_translation(d['translation'][sl],b,camera)
   pq=get_absolute_orientation(pt,rotation_matrix_to_quaternion(get_rotation_matrix(y[:,3:])))
   gq=get_absolute_orientation(gt,rotation_matrix_to_quaternion(d['rotation'][sl]))
   ts.append(translation_metric(pt,gt));rels.append(relative_translation_metric(pt,gt));rs.append(rotation_metric(pq,gq))
 vals=[torch.cat(a).mean().item() for a in [ts,rels,rs]]
 return {'translation_m':vals[0],'relative_translation':vals[1],'rotation_rad':vals[2],'speed_score':vals[1]+vals[2]}

def loss_fn(y,d,idx):
 pt=bbox_relative_translation_to_translation(y[:,:3],d['boxes'][idx],camera)
 gt=bbox_relative_translation_to_translation(d['translation'][idx],d['boxes'][idx],camera)
 lt=torch.linalg.vector_norm(pt-gt,dim=1)/torch.linalg.vector_norm(gt,dim=1)
 # Chordal distance of valid rotations is scale-invariant and smooth near zero.
 R=get_rotation_matrix(y[:,3:].float());target=d['rotation'][idx]
 lr=torch.sqrt(((R-target)**2).sum((1,2))/2+1e-8)
 return (lt+lr).mean()

initial=metrics();best=initial['speed_score'];best_epoch=0
model.save(str(out/'best.pth'))
print('VALIDATION_BASELINE',json.dumps(initial),flush=True)
optimizer=torch.optim.AdamW(model.decoder.parameters(),lr=args.lr,weight_decay=0.0001)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr/10)
history=[];train=cache['train'];n=len(train['tokens'])
# Initialization + validation are the only model-selection inputs. Test split is untouched.
for epoch in range(1,args.epochs+1):
 model.train();perm=torch.randperm(n,device=dev);total=0.
 for i in range(0,n,64):
  idx=perm[i:i+64];optimizer.zero_grad(set_to_none=True)
  y=model.forward_residual_features(train['tokens'][idx],train['geometry'][idx]);loss=loss_fn(y,train,idx)
  if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
  total+=loss.item()*len(idx)
 scheduler.step();val=metrics()
 row={'epoch':epoch,'train_loss':total/n,**val};history.append(row)
 if val['speed_score']<best:
  best=val['speed_score'];best_epoch=epoch;model.save(str(out/'best.pth'))
 (out/'history.json').write_text(json.dumps(history,indent=2))
 print('EPOCH',json.dumps(row),'BEST',best_epoch,best,flush=True)
result={'baseline_validation':initial,'best_epoch':best_epoch,'best_speed_score':best,'validation_relative_change_percent':100*(best/initial['speed_score']-1),'epochs':args.epochs,'history':history,'protocol':'FP32 frozen baseline features; train=9600, validation=1200; no test access; no image augmentation for frozen-feature calibration; bounded residual; geometry and gated dual queries retained'}
(out/'result.json').write_text(json.dumps(result,indent=2));(out/'status.txt').write_text('completed\n')
print('RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
