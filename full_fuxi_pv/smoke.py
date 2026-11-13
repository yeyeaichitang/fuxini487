"""Full-graph gradient/optimizer smoke test. NO real PV training or accuracy claim."""
from pathlib import Path
import argparse
import gc
from datetime import datetime,timezone
import hashlib
import json
import sys
import time
import numpy as np
import torch
import xarray as xr

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import FullFuXiPV, temporal_features, state_from_array, ROOT, A_NAME, B_NAME, WQ_NAME,CONTEXT_CHANNELS
from offload import saved_tensors_on_disk


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,parameter in model.fuxi.named_parameters():
        if name in (A_NAME,B_NAME):
            continue
        assert not parameter.requires_grad
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/full_fuxi_pv_gradient_smoke')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'scope':'complete original FuXi forward and PV-head backward to internal LoRA',
            'test_only':True,'training_completed':False,'supervision':'synthetic mechanism test',
            'real_pv_samples_used':0,'passed':False}
    started=time.perf_counter()
    try:
        torch.manual_seed(42)
        torch.set_num_threads(2)
        print('Loading FULL original FuXi graph',flush=True)
        model=FullFuXiPV(ROOT/'fuxi-2.1/fuxi-2.1.pt2',weight_cache=args.output/'weight_cache')
        report['trainable_parameters']=sum(p.numel() for p in model.trainable_parameters())
        report['lora_parameters']=getattr(model.fuxi,A_NAME).numel()+getattr(model.fuxi,B_NAME).numel()
        report['base_parameters']=sum(p.numel() for n,p in model.fuxi.named_parameters() if n not in (A_NAME,B_NAME))
        before_hash=frozen_digest(model)
        with xr.open_dataset(ROOT/'fuxi-2.1/input.nc') as dataset:
            state=state_from_array(dataset['input'].values,model.device)
        del dataset
        gc.collect()
        torch.cuda.empty_cache()
        temporal=temporal_features(datetime(2024,9,29,tzinfo=timezone.utc),model.device)
        # Arbitrary synthetic history and grid location: not a matched training pair.
        history=torch.randn(1,16,5,device=model.device,requires_grad=True)
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3,weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=65536)
        initial=model.adapter_state()
        print('Full graph forward with saved activation disk offload',flush=True)
        torch.cuda.reset_peak_memory_stats()
        with saved_tensors_on_disk(args.output/'activation_cache'):
            prediction=model(state,temporal,history,(350,700))
            with xr.open_dataarray(ROOT/'output/fuxi_full_retry_20261108/001.nc') as reference:
                physical=reference.sel(channel=CONTEXT_CHANNELS).isel(lat=350,lon=700).values
            with xr.open_dataarray(ROOT/'fuxi-2.1/mean.nc') as mean_da, xr.open_dataarray(ROOT/'fuxi-2.1/std.nc') as std_da:
                expected=(physical-mean_da.values[model.channels])/std_da.values[model.channels]
            np.testing.assert_allclose(model.last_context.numpy()[0],expected,rtol=1e-3,atol=1e-3)
            report['zero_lora_context_matches_original_forecast']=True
            target=(prediction.detach()+0.02).clamp(max=0.95)
            loss=torch.nn.functional.mse_loss(prediction,target)
            print(f'Prediction={prediction.item():.8f} diagnostic loss={loss.item():.8f}',flush=True)
            print('Backward through full graph to internal LoRA',flush=True)
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        lora_b=getattr(model.fuxi,B_NAME)
        checks={'nonzero_finite_lora_b_gradient':bool(lora_b.grad is not None and torch.isfinite(lora_b.grad).all() and lora_b.grad.abs().max()>0),
                'all_trainable_gradients_finite':all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.trainable_parameters()),
                'hardware_feature_gradients_nonzero':bool(history.grad is not None and torch.isfinite(history.grad).all() and (history.grad[0,:,:4].abs().sum(0)>0).all())}
        report['lora_b_gradient_norm']=float(lora_b.grad.norm()) if lora_b.grad is not None else None
        report['gradient_loss_scale']=scaler.get_scale()
        report['hardware_feature_gradient_norms']=(history.grad[0,:,:4].norm(dim=0)/scaler.get_scale()).tolist() if history.grad is not None else None
        scaler.step(optimizer)
        scaler.update()
        after=model.adapter_state()
        checks['lora_b_updated']=not torch.equal(initial['B'],after['B'])
        checks['pv_head_updated']=any(not torch.equal(initial['head'][k],v) for k,v in after['head'].items())
        checks['frozen_original_weights_unchanged']=frozen_digest(model)==before_hash
        report['checks']=checks
        report['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        if not all(checks.values()):
            raise RuntimeError('Full graph smoke checks failed')
        torch.save({'format':'full_fuxi_pv_lora_v1','state':after,'rank':4,'alpha':8.0,
                    'training_completed':False,'test_only':True,'supervision':'synthetic mechanism test'},args.output/'test_only_adapter.pt')
        report['passed']=True
        print('FULL GRAPH LoRA + HARDWARE INPUT GRADIENT TEST PASSED (not real PV training)',flush=True)
    except Exception as error:
        report['error_type']=type(error).__name__
        report['error']=str(error)
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
