import torch,json
from src.device import get_device
from object_detector.LWDETR.lwdetr_ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from object_detector.LWDETR.lwdetr_models import LWDETRWrapper
torch.manual_seed(17)
d=get_device();xs=[torch.randn(1,2,4,20),torch.rand(1,3,2,2,2,2),torch.rand(1,3,2,2,2)]
outs=[];grads=[]
for device in ['cpu',d]:
    args=[x.to(device).detach().requires_grad_() for x in xs]
    y=ms_deform_attn_core_pytorch(args[0],[(4,4),(2,2)],args[1],args[2]);y.square().mean().backward()
    outs.append(y.detach().cpu());grads.append([a.grad.cpu() for a in args])
torch.testing.assert_close(outs[0],outs[1],atol=1e-4,rtol=1e-4)
for a,b in zip(*grads):torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
print('MSDA forward/backward CPU-NPU passed',flush=True)
m=LWDETRWrapper('small');m.load_from_pretrained('weights/small.pth');m.to(d).train()
x=torch.randn(1,3,640,640,device=d)
y=m(x);loss=y['pred_boxes'].square().mean()+y['pred_logits'].square().mean();loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print('Detector FP32 backward passed; loss',loss.item(),flush=True)
