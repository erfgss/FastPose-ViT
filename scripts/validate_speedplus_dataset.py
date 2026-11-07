from pathlib import Path
import json,time,random,math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Subset
from src.datasets import SPEEDDataset
from src.camera import Camera
from src.geometry_decoder import encode_bbox_geometry,forward_pose
from src.models import FastPoseViT
from src.utils import translation_to_bbox_relative_translation,bbox_relative_translation_to_translation
root=Path('/data/datasets/SPEED_PLUS_FIXED');raw=Path('/data/datasets/speedplusv2')
torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.manual_seed(42)
count=0;tick=time.monotonic()
for split in ['train','val','test']:
 rows=json.loads((root/(split+'.json')).read_text())
 for row in rows:
  p=root/'images'/split/row['filename']
  with Image.open(p) as im:
   assert im.size==(1920,1200),(p,im.size)
   im.verify()
  count+=1
  if time.monotonic()-tick>15:print('IMAGE_HEADERS_VERIFIED',count,flush=True);tick=time.monotonic()
print('ALL_IMAGE_HEADERS_VERIFIED',count,flush=True)
report={'image_headers_verified':count,'splits':{},'test_domains':{'lightbox':6740,'sunlamp':2791}}
model=FastPoseViT.from_checkpoint('experiments/geometry_dual_token_full_20261107_022724_UTC/pretrained.pth').cpu().eval()
for split,indices in [('train',[0,100,1000,47965]),('val',[0,100,1000,11993]),('test',[0,6739,6740,9530])]:
 d=SPEEDDataset(str(root),split=split,rotation_format='matrix',bbox_json_path=str(root/'speed_plus_bbox_annotations.json'),dataset_name='SPEED_PLUS')
 x,t,r,b=next(iter(DataLoader(Subset(d,indices),batch_size=4,num_workers=0)))
 assert x.shape==(4,3,224,224) and r.shape==(4,3,3)
 assert all(torch.isfinite(z).all() for z in [x,t,r,b])
 g=encode_bbox_geometry(b,d.camera)
 actual=bbox_relative_translation_to_translation(t,b,d.camera)
 restored=translation_to_bbox_relative_translation(actual,b,d.camera)
 torch.testing.assert_close(t,restored,atol=1e-5,rtol=1e-5)
 with torch.no_grad():pred=forward_pose(model,x,b,d.camera)
 assert pred.shape==(4,9) and torch.isfinite(pred).all()
 report['splits'][split]={'samples':len(d),'sampled_indices':indices,'image_shape':list(x.shape),'prediction_shape':list(pred.shape),'geometry_finite':bool(torch.isfinite(g).all()),'fx':d.camera.fx,'fy':d.camera.fy}
 print('DATALOADER_AND_CPU_FORWARD',split,'passed',flush=True)
assert Camera().fx==0.0176/5.86e-6
assert abs(d.camera.fx-2988.5795163815555)<1e-9
sources=json.loads((root/'image_sources.json').read_text());name=next(iter(sources));source=raw/sources[name]
assert (root/'images/train'/name).stat().st_ino==source.stat().st_ino
report['hardlink_verified']=True
report['camera_mode']='dataset-local SPEED+ intrinsics; original images retained, no undistortion applied'
(root/'loader_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
