"""Checkpoint inference, CPU golden comparison and synthetic backward validation."""
import json,time,torch
from src.device import get_device,synchronize,make_grad_scaler
from src.models import FastPoseViT

torch.manual_seed(42)
torch.set_num_threads(4)
device=get_device()
results={}
for size in [224,384]:
    name='vit_b_16' if size==224 else 'vit_b_16_384'
    model=FastPoseViT(name+'_no_weights',1,768,3,6,merge_outputs=True,no_mlp=True)
    model.load_from_pretrained('weights/'+name+'.pth')
    x=torch.randn(1,3,size,size)
    model.eval()
    with torch.no_grad(): reference=model(x)
    model.to(device); x=x.to(device)
    with torch.no_grad():
        output=model(x);synchronize(device)
        assert output.shape==(1,9) and torch.isfinite(output).all()
        err=(output.cpu()-reference).abs()
        torch.testing.assert_close(output.cpu(),reference,atol=0.005,rtol=0.005)
        with torch.autocast(device.type,dtype=torch.float16): amp=model(x)
        assert torch.isfinite(amp).all()
        for _ in range(3):model(x)
        synchronize(device);start=time.perf_counter()
        for _ in range(10):model(x)
        synchronize(device)
        results[name]={'fp32_max_abs':err.max().item(),'fp16_max_abs':(amp.cpu()-reference).abs().max().item(),'fp32_ms':(time.perf_counter()-start)*100,'shape':list(output.shape)}
    if size==224:
        model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-5);scaler=make_grad_scaler(device.type,init_scale=256.)
        with torch.autocast(device.type,dtype=torch.float16): loss=model(x).float().square().mean()
        scaler.scale(loss).backward();scaler.unscale_(opt)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        scaler.step(opt);scaler.update();synchronize(device)
        results[name]['train_loss']=loss.item()
    print(json.dumps(results),flush=True)
    del model,x
    torch.npu.empty_cache()
from object_detector.LWDETR.lwdetr_models import LWDETRWrapper
model=LWDETRWrapper('small');model.load_from_pretrained('weights/small.pth');model.to(device).eval()
x=torch.randn(1,3,640,640,device=device)
with torch.no_grad():
    output=model(x);synchronize(device)
    print('DETECTOR',[(k,list(v.shape)) for k,v in output.items() if isinstance(v,torch.Tensor)],flush=True)
    assert all(torch.isfinite(v).all() for v in output.values() if isinstance(v,torch.Tensor))
results['detector']='finite FP32 checkpoint forward at 640x640'
print('RESULTS',json.dumps(results),flush=True)
open('mig_docs/working/smoke_results.json','w').write(json.dumps(results,indent=2))
