"""Supervised PV training THROUGH the complete original FuXi graph.

Requires a validated paired manifest; never substitutes synthetic/mismatched data.
"""
from pathlib import Path
import argparse
from contextlib import nullcontext
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


def metrics(actual,predicted,capacity):
    actual,predicted,capacity=map(lambda a:np.asarray(a,dtype=float),(actual,predicted,capacity))
    error=predicted-actual
    nonzero=actual>0.01*capacity
    return {'samples':len(actual),'mae_mw':float(np.abs(error).mean()),
            'rmse_mw':float(np.sqrt(np.mean(error**2))),
            'mape_percent_above_1pct_capacity':float(np.mean(np.abs(error[nonzero]/actual[nonzero]))*100) if nonzero.any() else None,
            'mape_eligible_samples':int(nonzero.sum()),
            'r2':float(1-np.sum(error**2)/np.sum((actual-actual.mean())**2)) if np.var(actual)>0 else None}


def predict_row(model,row,training,cache,scaler=None):
    x,y,as_of,grid=validate_example(row)
    values,init=load_weather(row['weather_path'],as_of)
    state=state_from_array(values,model.device)
    del values
    if init.isoformat()!=row['weather_init_utc']:
        raise ValueError('Weather init metadata mismatch')
    history=torch.from_numpy(x[None]).to(model.device)
    with saved_tensors_on_disk(cache) if training else torch.no_grad():
        prediction=model(state,temporal_features(init,model.device),history,grid)
        if training:
            loss=(prediction-y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            if scaler is None:
                raise ValueError('FP16 full graph requires gradient scaling')
            scaler.scale(loss).backward()
    return float(prediction.detach()[0])*row['request']['capacity_mw']


def evaluate(model,rows,cache):
    actual,predicted,capacity=[],[],[]
    details=[]
    for row in rows:
        prediction=predict_row(model,row,False,cache)
        actual.append(row['target_power_mw']);predicted.append(prediction);capacity.append(row['request']['capacity_mw'])
        details.append({'station_id':row['request']['station_id'],'timestamp':row['target_timestamp'],
                        'actual_power_mw':row['target_power_mw'],'predicted_power_mw':prediction})
    return metrics(actual,predicted,capacity),details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed/full_fuxi_pv_aligned')
    parser.add_argument('--output',type=Path,default=ROOT/'models/full_fuxi_pv_run1')
    parser.add_argument('--head-epochs',type=int,default=1)
    parser.add_argument('--lora-epochs',type=int,default=1)
    parser.add_argument('--train-limit',type=int,default=8,help='0 uses all paired training samples')
    parser.add_argument('--eval-limit',type=int,default=8,help='0 uses all paired validation/test samples')
    args=parser.parse_args()
    readiness=json.loads((args.data/'readiness.json').read_text(encoding='utf-8'))
    if not readiness.get('ready'):
        raise SystemExit('REAL PV TRAINING BLOCKED: '+ '; '.join(readiness['problems']))
    if min(args.head_epochs,args.lora_epochs)<1 or min(args.train_limit,args.eval_limit)<0:
        raise ValueError('Invalid training arguments')
    if args.output.exists():
        raise ValueError('Output exists; use a new output')
    rows=[json.loads(line) for line in (args.data/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line]
    for row in rows:
        validate_example(row)
    splits={split:[r for r in rows if r['split']==split] for split in ('train','val','test')}
    if not all(splits.values()):
        raise ValueError('All three real-data splits required')
    # Verify that raw history/target periods do not overlap for a station.
    from contract import timestamp
    for site in {r['request']['station_id'] for r in rows}:
        latest=None
        for split in ('train','val','test'):
            station_rows=[r for r in splits[split] if r['request']['station_id']==site]
            if not station_rows:
                continue
            first=min(timestamp(r['request']['history'][0]['timestamp']) for r in station_rows)
            last=max(timestamp(r['target_timestamp']) for r in station_rows)
            if latest is not None and first<=latest:
                raise ValueError('Cross-split raw-time overlap')
            latest=last
    available={key:len(value) for key,value in splits.items()}
    for split in splits:
        limit=args.train_limit if split=='train' else args.eval_limit
        if limit and len(splits[split])>limit:
            ids=np.linspace(0,len(splits[split])-1,limit,dtype=int)
            splits[split]=[splits[split][i] for i in ids]
    args.output.mkdir(parents=True)
    torch.manual_seed(42);torch.set_num_threads(2)
    model=FullFuXiPV(ROOT/'fuxi-2.1/fuxi-2.1.pt2',weight_cache=args.output/'weight_cache')
    log=[]
    stages={}
    for stage,epochs in [('head',args.head_epochs),('lora',args.lora_epochs)]:
        for p in model.head.parameters():
            p.requires_grad_(stage=='head')
        for name in (A_NAME,B_NAME):
            getattr(model.fuxi,name).requires_grad_(stage=='lora')
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3 if stage=='head' else 1e-4,weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=65536)
        initial=model.adapter_state()
        best=model.adapter_state()
        best_score,_=evaluate(model,splits['val'],args.output/'activation_cache')
        received_gradient=False
        for epoch in range(epochs):
            for row in splits['train']:
                optimizer.zero_grad(set_to_none=True)
                prediction=predict_row(model,row,True,args.output/'activation_cache',scaler)
                scaler.unscale_(optimizer)
                for p in model.trainable_parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError('Missing/nonfinite training gradient')
                if stage=='lora':
                    received_gradient|=bool(getattr(model.fuxi,B_NAME).grad.abs().max()>0)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.0)
                scaler.step(optimizer)
                scaler.update()
            score,_=evaluate(model,splits['val'],args.output/'activation_cache')
            log.append({'stage':stage,'epoch':epoch+1,'validation':score})
            (args.output/'training_log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
            if score['mae_mw']<best_score['mae_mw']:
                best,best_score=model.adapter_state(),score
        last=model.adapter_state()
        if stage=='lora':
            if not received_gradient or torch.equal(initial['B'],last['B']):
                raise ValueError('LoRA did not update')
            if any(not torch.equal(initial['head'][k],v) for k,v in last['head'].items()):
                raise ValueError('PV head unexpectedly changed in LoRA-only stage')
            torch.save(last,args.output/'lora_last.pt')
        model.load_adapter_state(best)
        stages[stage]=best
    schema_by_site={r['request']['station_id']:r['schema'] for r in splits['train']}
    output={'format':'full_fuxi_pv_lora_v1','state':model.adapter_state(),'rank':4,'alpha':8.0,
            'training_completed':True,'test_only':False,'supervision':'matched observed station power',
            'schema_by_site':schema_by_site,'model_scope':'complete original FuXi graph with internal layer29 LoRA and PV head',
            'base_model_path':'fuxi-2.1/fuxi-2.1.pt2','manifest_sha256':hashlib.sha256((args.data/'samples.jsonl').read_bytes()).hexdigest()}
    torch.save(output,args.output/'model.pt')
    (args.output/'feature_schema.json').write_text(json.dumps(schema_by_site,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'preprocessing.json').write_text(json.dumps({'schema_by_site':schema_by_site,
        'training_imputation':False,'runtime_power_forward_fill_max_consecutive':2,
        'weather_imputation':False,'history_len':16,'horizon_minutes':15},ensure_ascii=False,indent=2),encoding='utf-8')
    selected=model.adapter_state()
    model.load_adapter_state(stages['head'])
    before,_=evaluate(model,splits['test'],args.output/'activation_cache')
    model.load_adapter_state(selected)
    after,details=evaluate(model,splits['test'],args.output/'activation_cache')
    results={'available_samples':available,'used_samples':{k:len(v) for k,v in splits.items()},
             'before_lora':before,'after_lora':after,
             'lora_selected_by_validation':not torch.equal(stages['head']['B'],selected['B']),
             'hardware_connected':False,'DG_station_accuracy_validated':False}
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (args.output/'predictions.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    sample=splits['test'][0]
    (args.output/'example_input.json').write_text(json.dumps(sample['request'],ensure_ascii=False,indent=2),encoding='utf-8')
    example={'status':'ok','station_id':sample['request']['station_id'],'based_on':sample['request']['as_of'],
             'interval_minutes':15,'model_name':'pv-forecast','model_version':'full-fuxi-pv-'+args.output.name,
             'predictions':[{'timestamp':sample['target_timestamp'],'power_mw':details[0]['predicted_power_mw']}],
             'example_origin':'held-out historical observation, not live hardware'}
    (args.output/'example_output.json').write_text(json.dumps(example,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
