"""Small, explicitly provisional full-FuXi training run with measured PV labels.

This train-only run does not bypass the verified train/val/test requirements of
train.py and cannot produce a deployable checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import FullFuXiPV,ROOT,A_NAME,B_NAME,temporal_features,state_from_array
from data import validate_example,load_weather
from offload import saved_tensors_on_disk
from smoke import frozen_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/processed/hkust_rooftop_pv/verification/candidate_pairs.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'output/full_fuxi_real_pv_pilot')
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--head-steps',type=int,default=16)
    parser.add_argument('--full-steps',type=int,default=4)
    parser.add_argument('--resume',type=Path,help='Continue an explicitly test-only PV pilot adapter')
    args=parser.parse_args()
    if min(args.samples,args.full_steps)<1 or args.head_steps<0:
        raise ValueError('Positive sample/full-step counts and nonnegative head steps required')
    rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line]
    if not rows or any(r['split']!='train' or r.get('pair_status')!='provisional_not_approved_for_training' for r in rows):
        raise ValueError('This pilot expects the explicitly provisional train-only manifest')
    rows.sort(key=lambda r:r['target_power_mw']/r['request']['capacity_mw'])
    selected=[];sites=set()
    for index in np.linspace(0,len(rows)-1,min(args.samples,len(rows)),dtype=int):
        choices=sorted(range(len(rows)),key=lambda j:abs(j-int(index)))
        row=next((rows[j] for j in choices if rows[j]['request']['station_id'] not in sites),None)
        if row is None:break
        selected.append(row);sites.add(row['request']['station_id'])
    if not selected:raise ValueError('No actual PV samples')
    weather_keys={(r['weather_path'],r['weather_init_utc'],r['latitude'],r['longitude']) for r in selected}
    if len(weather_keys)!=1:raise ValueError('Pilot context caching requires one weather initialization and grid')
    prepared=[validate_example(row) for row in selected]
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report={'passed':False,'supervision':'observed historical photovoltaic active power',
            'synthetic_labels_used':False,'pilot_training_completed':False,
            'formal_training_completed':False,'production_model_ready':False,
            'held_out_accuracy_evaluated':False,'test_only':True,
            'available_candidate_samples':len(rows),'real_pv_samples_used':len(selected),
            'head_optimizer_steps':0,'full_graph_optimizer_steps':0,
            'assumptions':['UTC+08:00 export-time hypothesis','published campus representative coordinates',
                           'publisher SLP retained unchanged; pressure bias remains unresolved'],
            'full_graph_steps':[]}
    def save_report():
        report['elapsed_seconds']=time.perf_counter()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        torch.manual_seed(42);torch.set_num_threads(2)
        (args.output/'training_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
        report['source_manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        print(f'Loading complete FuXi; {len(selected)} measured PV examples selected',flush=True)
        model=FullFuXiPV(ROOT/'fuxi-2.1/fuxi-2.1.pt2',weight_cache=args.output/'weight_cache')
        report['base_parameters']=sum(p.numel() for n,p in model.fuxi.named_parameters() if n not in (A_NAME,B_NAME))
        report['lora_parameters']=sum(getattr(model.fuxi,n).numel() for n in (A_NAME,B_NAME))
        report['head_parameters']=sum(p.numel() for p in model.head.parameters())
        original_digest=frozen_digest(model)
        if args.resume:
            prior=torch.load(args.resume,map_location='cpu',weights_only=True)
            if prior.get('format')!='full_fuxi_pv_lora_pilot_v1' or not prior.get('test_only'):
                raise ValueError('Only a test-only real PV pilot adapter may be resumed')
            if prior.get('manifest_sha256')!=report['source_manifest_sha256']:
                raise ValueError('Resume manifest differs from the original training data')
            model.load_adapter_state(prior['state'])
            report['resumed_from']=str(args.resume)
        initial=model.adapter_state()
        # Each sample has the same actual weather input and campus grid. Freeze
        # the backbone only during head warm-up; do not cache/detach joint steps.
        values,init=load_weather(selected[0]['weather_path'],prepared[0][2])
        state=state_from_array(values,model.device);del values
        temporal=temporal_features(init,model.device)
        history=torch.from_numpy(np.stack([x[0] for x in prepared])).to(model.device)
        targets=torch.tensor([x[1] for x in prepared],dtype=torch.float32,device=model.device)
        grid=prepared[0][3]
        torch.cuda.reset_peak_memory_stats()
        print('Full FuXi forward for real-weather context and initial measured-power loss',flush=True)
        with torch.no_grad():
            model(state,temporal,history[:1],grid)
            context=model.last_context.to(model.device)
        def head_predictions(ctx):
            return model.head(torch.cat([history.flatten(1),ctx.expand(len(history),-1)],dim=1)).squeeze(-1)
        with torch.no_grad():
            initial_predictions=head_predictions(context)
            report['initial_training_mse_pu']=float((initial_predictions-targets).square().mean())
        for name in (A_NAME,B_NAME):getattr(model.fuxi,name).requires_grad_(False)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-3,weight_decay=0)
        for step in range(args.head_steps):
            optimizer.zero_grad(set_to_none=True)
            loss=(head_predictions(context)-targets).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite observed-power head loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.head.parameters()):
                raise ValueError('Invalid head gradient')
            optimizer.step();report['head_optimizer_steps']+=1
        with torch.no_grad():report['after_head_training_mse_pu']=float((head_predictions(context)-targets).square().mean())
        print(f'Head warm-up finished: MSE {report["initial_training_mse_pu"]:.8g} -> {report["after_head_training_mse_pu"]:.8g}',flush=True)
        for name in (A_NAME,B_NAME):getattr(model.fuxi,name).requires_grad_(True)
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-4,weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=65536)
        # Give daylight observations explicit coverage even if most pilot labels
        # are zero near dawn; do not spend every full-graph step on night labels.
        positives=sorted([i for i,p in enumerate(prepared) if p[1]>0],key=lambda i:prepared[i][1],reverse=True)
        order=positives+[i for i,p in enumerate(prepared) if p[1]==0]
        for step in range(args.full_steps):
            index=order[step%len(order)]
            optimizer.zero_grad(set_to_none=True)
            print(f'FULL GRAPH PV TRAINING step {step+1}/{args.full_steps}; station={selected[index]["request"]["station_id"]}, observed target={float(targets[index]):.8g} pu',flush=True)
            with saved_tensors_on_disk(args.output/'activation_cache'):
                prediction=model(state,temporal,history[index:index+1],grid)
                loss=(prediction-targets[index]).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite real PV full-graph loss')
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.trainable_parameters()):
                raise ValueError('Missing/nonfinite joint training gradients')
            gradient=getattr(model.fuxi,B_NAME).grad
            if not gradient.abs().max()>0:raise ValueError('Observed PV loss did not reach internal FuXi LoRA')
            record={'step':step+1,'station_id':selected[index]['request']['station_id'],
                    'target_timestamp':selected[index]['target_timestamp'],
                    'target_power_mw':selected[index]['target_power_mw'],
                    'loss_mse_pu':float(loss.detach()),'lora_b_gradient_norm':float(gradient.norm())}
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.)
            scaler.step(optimizer);scaler.update()
            report['full_graph_optimizer_steps']+=1
            report['full_graph_steps'].append(record)
            del prediction,loss
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            save_report()
            print(json.dumps(record),flush=True)
        with torch.no_grad():
            model(state,temporal,history[:1],grid)
            final_predictions=head_predictions(model.last_context.to(model.device))
            report['final_training_mse_pu_same_samples']=float((final_predictions-targets).square().mean())
        final=model.adapter_state()
        report['lora_b_updated']=not torch.equal(initial['B'],final['B'])
        report['lora_a_updated']=not torch.equal(initial['A'],final['A'])
        report['pv_head_updated']=any(not torch.equal(initial['head'][k],v) for k,v in final['head'].items())
        report['original_frozen_weights_unchanged']=frozen_digest(model)==original_digest
        report['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        if not all(report[k] for k in ['lora_b_updated','pv_head_updated','original_frozen_weights_unchanged']):
            raise ValueError('Parameter update or frozen-weight verification failed')
        checkpoint={'format':'full_fuxi_pv_lora_pilot_v1','state':final,'rank':4,'alpha':8.,
                    'training_completed':False,'test_only':True,'pilot_training_completed':True,
                    'supervision':report['supervision'],'full_graph_optimizer_steps':args.full_steps,
                    'schema_by_site':{r['request']['station_id']:r['schema'] for r in selected},
                    'assumptions':report['assumptions'],'manifest_sha256':report['source_manifest_sha256']}
        torch.save(checkpoint,args.output/'pv_supervised_pilot_adapter.pt')
        details=[{'station_id':r['request']['station_id'],'timestamp':r['target_timestamp'],
                  'actual_power_mw':r['target_power_mw'],
                  'initial_prediction_mw':float(initial_predictions[i])*r['request']['capacity_mw'],
                  'final_prediction_mw':float(final_predictions[i])*r['request']['capacity_mw'],
                  'evaluation_split':'training_replay_not_held_out'} for i,r in enumerate(selected)]
        (args.output/'training_replay_predictions.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        report.update(passed=True,pilot_training_completed=True)
        print(json.dumps(report,indent=2),flush=True)
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error)
        raise
    finally:save_report()


if __name__=='__main__':main()
