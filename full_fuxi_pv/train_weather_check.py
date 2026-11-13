"""One real-ERA5 supervised LoRA update through full FuXi, not PV fine-tuning."""
from pathlib import Path
import argparse
from datetime import datetime,timedelta
import json
import sys
import time
import numpy as np
import torch
import xarray as xr

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import FullFuXiPV,ROOT,A_NAME,B_NAME,temporal_features,state_from_array
from data import load_weather
from offload import saved_tensors_on_disk
from smoke import frozen_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weather',type=Path,default=ROOT/'data/weather/era5_20190101')
    parser.add_argument('--output',type=Path,default=ROOT/'output/full_fuxi_real_weather_update')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'scope':'one full FuXi LoRA update supervised by independently downloaded future ERA5 temperature',
            'pv_training_completed':False,'production_model_ready':False,'passed':False,
            'training_pairs':1,'optimizer_steps':0,'held_out_accuracy_evaluated':False}
    started=time.perf_counter()
    try:
        torch.manual_seed(42);torch.set_num_threads(2)
        print('Loading full graph with disk-backed original weights',flush=True)
        model=FullFuXiPV(ROOT/'fuxi-2.1/fuxi-2.1.pt2',weight_cache=args.output/'weight_cache')
        for p in model.head.parameters():
            p.requires_grad_(False)
        initial=model.adapter_state()
        original_hash=frozen_digest(model)
        values,init=load_weather(args.weather/'input.nc')
        state=state_from_array(values,model.device)
        del values
        features=temporal_features(init,model.device)
        channel=model.channels[0]  # t2m
        with xr.open_dataset(args.weather/'t2m_truth.nc') as truth:
            if datetime.fromisoformat(truth.attrs['valid_time'])!=init+timedelta(hours=6):
                raise ValueError('Truth is not at input +6h')
            raw=truth.t2m.values.copy()
        with xr.open_dataarray(ROOT/'fuxi-2.1/mean.nc') as mean, xr.open_dataarray(ROOT/'fuxi-2.1/std.nc') as std:
            target=(raw-mean.values[channel])/(std.values[channel]+1e-8)
        if not np.isfinite(target).all():
            raise ValueError('Nonfinite weather truth')
        target=torch.from_numpy(target).float().to(model.device)
        lat_weights=torch.from_numpy(np.cos(np.deg2rad(np.linspace(90,-90,721))).astype(np.float32)).to(model.device)[:,None]
        def loss_from(output):
            error=(output[0,-1,channel].float()-target).square()
            return (error*lat_weights).sum()/(lat_weights.sum()*1440)
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-4,weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=65536)
        torch.cuda.reset_peak_memory_stats()
        print('Forward on real 2019 ERA5 input, loss against real +6h t2m',flush=True)
        with saved_tensors_on_disk(args.output/'activation_cache'):
            output=model.weather_forward(state,features)
            loss=loss_from(output)
            report['initial_normalized_mse']=float(loss.detach())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite weather loss')
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad=getattr(model.fuxi,B_NAME).grad
        report['lora_b_gradient_norm']=float(grad.norm()) if grad is not None else None
        if grad is None or not torch.isfinite(grad).all() or not grad.abs().max()>0:
            raise ValueError('No finite nonzero LoRA gradient from real weather truth')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.trainable_parameters()):
            raise ValueError('Nonfinite trainable gradient')
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.0)
        scaler.step(optimizer);scaler.update()
        report['optimizer_steps']=1
        del output,loss
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.no_grad():
            output=model.weather_forward(state,features)
            report['final_normalized_mse_same_pair']=float(loss_from(output))
        updated=model.adapter_state()
        report['lora_b_updated']=not torch.equal(initial['B'],updated['B'])
        report['original_weights_unchanged']=frozen_digest(model)==original_hash
        report['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        if not report['lora_b_updated'] or not report['original_weights_unchanged']:
            raise ValueError('LoRA update/frozen-base verification failed')
        # Deliberately not a deployable PV model; original weather inference can
        # load these A/B tensors, but this is only a one-pair training check.
        torch.save({'format':'fuxi2.1_original_graph_lora_v1','target_node':'linear_235',
                    'target_weight':'model.decoder.layers.29.attn.wq.weight','rank':4,'alpha':8.0,
                    'A':updated['A'],'B':updated['B'],'training_completed':False,'test_only':True,
                    'supervision':'observed ERA5 +6h 2m temperature','optimizer_steps_completed':1,
                    'pv_training_completed':False},args.output/'weather_training_check_adapter.pt')
        report['passed']=True
        print(json.dumps(report,indent=2),flush=True)
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error)
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
