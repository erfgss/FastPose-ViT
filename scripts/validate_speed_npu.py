import torch,json
from src.device import get_device,make_grad_scaler,synchronize
from src.models import FastPoseViT
from src.datasets import SPEEDDataset
from src.utils import rotation_matrix_to_6d
from src.losses import RelativeTranslationLoss,RelativeRotation6DFrobeniusLoss
from torch.utils.data import DataLoader,Subset
torch.set_num_threads(4);torch.manual_seed(42)
dataset=SPEEDDataset('/data/datasets/SPEED_FIXED',split='test',rotation_format='matrix',bbox_json_path='assets/speed_bbox_annotations.json')
loader=DataLoader(Subset(dataset,range(32)),batch_size=4)
model=FastPoseViT('vit_b_16_no_weights',1,768,3,6,merge_outputs=True,no_mlp=True)
model.load_from_pretrained('weights/vit_b_16.pth');model.eval()
batches=list(loader)
with torch.no_grad():cpu=torch.cat([model(batch[0]) for batch in batches])
dev=get_device();model.to(dev)
with torch.no_grad():
 npu=torch.cat([model(batch[0].to(dev)).cpu() for batch in batches])
 with torch.autocast(dev.type,dtype=torch.float16):amp=torch.cat([model(batch[0].to(dev)).float().cpu() for batch in batches])
assert torch.isfinite(npu).all() and torch.isfinite(amp).all()
torch.testing.assert_close(npu,cpu,atol=0.005,rtol=0.005)
result={'samples':32,'fp32_max_abs':(npu-cpu).abs().max().item(),'fp32_mean_abs':(npu-cpu).abs().mean().item(),'fp16_max_abs':(amp-cpu).abs().max().item(),'fp16_mean_abs':(amp-cpu).abs().mean().item()}
# Real labeled training samples, finite gradients and optimizer updates; not a convergence study.
trainset=SPEEDDataset('/data/datasets/SPEED_FIXED',split='train',rotation_format='matrix',bbox_json_path='assets/speed_bbox_annotations.json')
trainloader=DataLoader(Subset(trainset,range(12)),batch_size=4)
opt=torch.optim.AdamW(model.parameters(),lr=1e-6);scaler=make_grad_scaler(dev.type,init_scale=256.)
model.train();losses=[]
for x,t,r,b in trainloader:
 x,t,r=x.to(dev),t.to(dev),r.to(dev);opt.zero_grad()
 with torch.autocast(dev.type,dtype=torch.float16):pred=model(x)
 loss=RelativeTranslationLoss()(pred[:,:3].float(),t)+RelativeRotation6DFrobeniusLoss()(pred[:,3:].float(),rotation_matrix_to_6d(r))
 assert torch.isfinite(loss)
 scaler.scale(loss).backward();scaler.unscale_(opt)
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();losses.append(loss.item())
result['real_train_losses']=losses
synchronize(dev)
print(json.dumps(result),flush=True)
open('mig_docs/working/speed_validation/golden_train.json','w').write(json.dumps(result,indent=2))
