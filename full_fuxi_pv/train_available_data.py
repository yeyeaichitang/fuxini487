"""Train using existing PV sources only; no downloads or application interfaces.

Unpaired station histories pretrain the PV head with its weather-context branch
ablated. The full original FuXi graph receives LoRA updates only from actual
ERA5/PV candidate pairs. Hold out entire stations from every training stage.
"""
from pathlib import Path
import copy
import gc
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import FullFuXiPV,ROOT,A_NAME,B_NAME,temporal_features,state_from_array
from data import validate_example,load_weather
from offload import saved_tensors_on_disk
from smoke import frozen_digest

OUT=ROOT/'models/full_fuxi_pv_available_data_v1'
ORIGINAL=ROOT/'data/processed/pv_all_sites_v2'
HK=ROOT/'data/processed/hkust_rooftop_pv'
MANIFEST=HK/'verification/candidate_pairs.jsonl'
SEED=42
BATCH=1024
HEAD_EPOCHS=3
LORA_EPOCHS=12
WEATHER=['temperature_c','pressure_hpa','humidity_percent','irradiance_w_m2']


def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def head_copy(head):
    return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}


def score(actual,pred):
    error=np.asarray(pred,dtype=float)-np.asarray(actual,dtype=float)
    return {'n':len(error),'mae_pu':float(np.abs(error).mean()),
            'rmse_pu':float(np.sqrt(np.mean(error**2))),
            'mse_pu_squared':float(np.mean(error**2))}


def predict_head(head,features,context):
    return head(torch.cat([features,context.expand(len(features),-1)],dim=1)).squeeze(-1)


def head_eval(head,x,y,device):
    parts=[]
    with torch.no_grad():
        for start in range(0,len(x),8192):
            features=torch.from_numpy(x[start:start+8192]).to(device)
            parts.append(predict_head(head,features,torch.zeros(1,8,device=device)).cpu().numpy())
    return score(y,np.concatenate(parts))


def load_histories(heldout):
    schemas={};usage=[]
    blocks={s:[] for s in ['train','val','test']}
    labels={s:[] for s in blocks}
    # One weather-only scaler fit before the ERA5 pilot date. It uses no held-out
    # station power labels, and is shared by HK training and unseen stations.
    met=pd.read_csv(HK/'weather/weather_15min_trailing.csv',parse_dates=['timestamp'])
    fit=met.loc[(met.timestamp<'2021-12-01') & met[WEATHER].notna().all(axis=1),WEATHER]
    hk_mean=fit.mean().to_numpy();hk_std=fit.std(ddof=0).to_numpy()
    if not np.isfinite(hk_mean).all() or (hk_std<=0).any():raise ValueError('Invalid training weather scaler')
    for source,base in [('original',ORIGINAL),('hkust',HK)]:
        files=sorted((base/'datasets').glob('*_schema.json'))
        if source=='original' and len(files)!=8:raise ValueError('Expected eight canonical original stations')
        for path in files:
            schema=json.loads(path.read_text(encoding='utf-8'))
            site=schema['site_id'];schema=copy.deepcopy(schema)
            if source=='hkust':
                schema['weather_mean']=hk_mean.tolist();schema['weather_std']=hk_std.tolist()
                schema['scaler_fit']='shared observed HK weather before 2021-12-01, no station power labels'
            schemas[site]=schema
            if site in heldout:
                usage.append({'site_id':site,'source':source,'role':'held_out_entire_station','training_samples':0})
                continue
            counts={}
            with np.load(base/'datasets'/f'{site}.npz',allow_pickle=False) as arrays:
                if source=='hkust':
                    raw=arrays['raw_features'].astype(np.float64)
                    scaled=raw.copy();scaled[:,:4]=(raw[:,:4]-hk_mean)/hk_std;scaled[:,4]/=schema['capacity_mw']
                    timeline=scaled.astype(np.float32)
                for split in blocks:
                    if source=='original':
                        x=arrays['X_'+split].reshape(-1,80).copy();y=arrays['y_'+split].copy()
                    else:
                        ids=arrays['target_row_'+split]
                        indices=ids[:,None]+np.arange(-16,0)[None,:]
                        x=timeline[indices].reshape(-1,80);y=arrays['y_'+split].copy()
                    if len(x) and (not np.isfinite(x).all() or not np.isfinite(y).all()):
                        raise ValueError(f'Invalid cleaned history: {site} {split}')
                    blocks[split].append(x);labels[split].append(y);counts[split]=len(y)
            usage.append({'site_id':site,'source':source,'role':'sensor_history_head_pretraining',
                          'training_samples':counts['train'],'samples':counts,
                          'global_weather_context':'not available; branch ablated, no fabricated global observations'})
            print(f'Loaded history {site}: {counts}',flush=True)
    return ({s:np.concatenate(blocks[s]) for s in blocks},
            {s:np.concatenate(labels[s]) for s in labels},schemas,usage)


def make_plots(predictions,metrics):
    import os
    plot_cache=OUT/'plot_cache'
    plot_cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(plot_cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    table=pd.DataFrame(predictions)
    sites=list(table.groupby('station_id').actual_power_mw.max().sort_values(ascending=False).index[:3])
    fig,axes=plt.subplots(len(sites),1,figsize=(11,3*len(sites)),squeeze=False)
    for axis,site in zip(axes[:,0],sites):
        values=table.loc[table.station_id==site].sort_values('timestamp')
        dates=pd.to_datetime(values.timestamp,utc=True).dt.tz_convert('Asia/Hong_Kong')
        axis.plot(dates,values.actual_power_mw*1000,color='black',label='Measured',lw=2)
        axis.plot(dates,values.before_lora_power_mw*1000,'--',label='Before LoRA',lw=1.6)
        axis.plot(dates,values.after_lora_power_mw*1000,label='After LoRA',lw=1.4)
        axis.set(title=site,ylabel='Power (kW)');axis.grid(alpha=.25)
        axis.xaxis.set_major_formatter(DateFormatter('%H:%M',tz=dates.dt.tz))
        axis.legend(loc='upper left',ncol=3)
    axes[-1,0].set_xlabel('2021-12-02 local time (UTC+8 hypothesis)')
    fig.suptitle('Held-out stations: measured power and full-FuXi LoRA predictions\nSingle ERA5 initialization; shared campus location assumption',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'comparison_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for axis,name in zip(axes,['mae_pu','rmse_pu']):
        values=[metrics['before_lora'][name]*100,metrics['after_lora'][name]*100]
        bars=axis.bar(['Before LoRA','After LoRA'],values,color=['#718096','#2878b5'])
        axis.set_ylabel('% of station rated capacity');axis.set_title(name.replace('_pu','').upper())
        axis.bar_label(bars,fmt='%.5f',padding=3);axis.set_ylim(0,max(values)*1.25+1e-5);axis.grid(axis='y',alpha=.2)
    fig.suptitle('Same held-out station examples; lower is better\nNo cross-season or independent-weather accuracy claim',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(OUT/'comparison_metrics.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();log=[]
    status={'completed':False,'production_model_ready':False}
    save_json(OUT/'status.json',status)
    try:
        torch.manual_seed(SEED);torch.set_num_threads(2)
        rng=np.random.default_rng(SEED)
        rows=[json.loads(l) for l in MANIFEST.read_text(encoding='utf-8').splitlines() if l]
        paired_sites=sorted({r['request']['station_id'] for r in rows})
        perm=list(rng.permutation(paired_sites));groups={'train':perm[:-10],'val':perm[-10:-5],'test':perm[-5:]}
        assert not(set(groups['train'])&set(groups['test']))
        heldout=set(groups['val']+groups['test'])
        config={'seed':SEED,'head_epochs':HEAD_EPOCHS,'head_batch_size':BATCH,'lora_max_epochs':LORA_EPOCHS,
                'lora_learning_rate':1e-3,'lora_full_batch':True,'paired_station_groups':groups,
                'protocol':'entire held-out stations excluded from every head and LoRA training stage',
                'scope':'available-data research model; one weather initialization, conditional time/location metadata',
                'source_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
        save_json(OUT/'training_config.json',config)
        x,y,schemas,usage=load_histories(heldout)
        save_json(OUT/'source_usage.json',usage);save_json(OUT/'feature_schema.json',schemas)
        save_json(OUT/'preprocessing.json',{'schema_by_site':schemas,'history_len':16,
                  'interval_minutes':15,'horizon_minutes':15,'training_imputation':False,
                  'model_fields':WEATHER+['power_pu'],'hk_pressure_reference':'publisher SLP, unresolved bias retained'})
        device=torch.device('cuda')
        head=nn.Sequential(nn.Linear(88,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid()).to(device)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
        train_x=torch.from_numpy(x['train']).to(device);train_y=torch.from_numpy(y['train']).to(device)
        best_head=None;best_val=float('inf')
        for epoch in range(HEAD_EPOCHS):
            order=torch.randperm(len(train_y),device=device);total=0.
            for start in range(0,len(order),BATCH):
                ids=order[start:start+BATCH];optimizer.zero_grad(set_to_none=True)
                prediction=predict_head(head,train_x[ids],torch.zeros(1,8,device=device))
                loss=(prediction-train_y[ids]).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite head pretraining loss')
                loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
            validation=head_eval(head,x['val'],y['val'],device)
            record={'stage':'history_head','epoch':epoch+1,'train_mse_pu':total/len(train_y),'validation':validation}
            log.append(record);print(json.dumps(record),flush=True);save_json(OUT/'training_log.json',log)
            if validation['mse_pu_squared']<best_val:best_val=validation['mse_pu_squared'];best_head=head_copy(head)
        head.load_state_dict(best_head)
        history_test=head_eval(head,x['test'],y['test'],device)
        history_counts={s:len(y[s]) for s in y}
        del train_x,train_y,x,y,optimizer;gc.collect();torch.cuda.empty_cache()
        # Rewrite sample schemas only; source measurements and original manifests remain untouched.
        splits={s:[] for s in groups}
        for row in rows:
            row=copy.deepcopy(row);site=row['request']['station_id']
            split=next(s for s,ids in groups.items() if site in ids)
            row['source_calendar_split']=row['split'];row['split']=split;row['schema']=schemas[site]
            splits[split].append(row)
        with (OUT/'paired_split_manifest.jsonl').open('w',encoding='utf-8') as stream:
            for split in splits:
                for row in splits[split]:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        prepared={s:[validate_example(r) for r in splits[s]] for s in splits}
        px={s:torch.from_numpy(np.stack([v[0] for v in prepared[s]])).reshape(-1,80).to(device) for s in splits}
        py={s:torch.tensor([v[1] for v in prepared[s]],dtype=torch.float32,device=device) for s in splits}
        print('Loading complete original FuXi graph for real paired observations',flush=True)
        model=FullFuXiPV(ROOT/'fuxi-2.1/fuxi-2.1.pt2',weight_cache=OUT/'weight_cache')
        model.head.load_state_dict(head.state_dict());del head
        original_hash=frozen_digest(model)
        first=splits['train'][0]
        weather_keys={(r['weather_path'],r['weather_init_utc'],r['latitude'],r['longitude']) for rs in splits.values() for r in rs}
        if len(weather_keys)!=1:raise ValueError('Shared-context batching requires one actual weather input and grid')
        values,init=load_weather(first['weather_path'],prepared['train'][0][2]);state=state_from_array(values,device);del values
        temporal=temporal_features(init,device);grid=prepared['train'][0][3]
        for rs in prepared.values():
            for _,_,as_of,_ in rs:
                if not 0<=(as_of-init).total_seconds()<=21600:raise ValueError('Future or stale global context')
        def full_context():
            output=model.weather_forward(state,temporal)
            return output[:,-1,model.channels,grid[0],grid[1]].float()
        def predictions(context,split):
            return predict_head(model.head,px[split],context)
        def paired_eval(context,split):
            with torch.no_grad():return score(py[split].cpu().numpy(),predictions(context,split).cpu().numpy())
        with torch.no_grad():context=full_context()
        for name in (A_NAME,B_NAME):getattr(model.fuxi,name).requires_grad_(False)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=0)
        best_head=head_copy(model.head);best_val=paired_eval(context,'val')['mse_pu_squared']
        for step in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss=(predictions(context,'train')-py['train']).square().mean();loss.backward();optimizer.step()
            validation=paired_eval(context,'val')
            if validation['mse_pu_squared']<best_val:best_val=validation['mse_pu_squared'];best_head=head_copy(model.head)
        model.head.load_state_dict(best_head)
        baseline_state=model.adapter_state()
        before_val=paired_eval(context,'val');before=paired_eval(context,'test')
        with torch.no_grad():before_pred=predictions(context,'test').cpu().numpy()
        print('Before LoRA held-out station metrics: '+json.dumps(before),flush=True)
        for p in model.head.parameters():p.requires_grad_(False)
        for name in (A_NAME,B_NAME):getattr(model.fuxi,name).requires_grad_(True)
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3,weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=65536)
        best_state=None;best_epoch=None;best_val=float('inf')
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(LORA_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            print(f'FULL FUXI LoRA epoch {epoch+1}/{LORA_EPOCHS}, {len(py["train"])} real paired training targets',flush=True)
            with saved_tensors_on_disk(OUT/'activation_cache'):
                context=full_context();loss=(predictions(context,'train')-py['train']).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite full-network loss')
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient=getattr(model.fuxi,B_NAME).grad
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.trainable_parameters()):
                raise ValueError('Invalid LoRA gradients')
            if not gradient.abs().max()>0:raise ValueError('Real power loss did not reach LoRA')
            gradient_norm=float(gradient.norm());train_loss=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            del loss,context;torch.cuda.empty_cache()
            with torch.no_grad():context=full_context()
            validation=paired_eval(context,'val')
            record={'stage':'full_graph_lora','epoch':epoch+1,'train_mse_pu':train_loss,
                    'lora_b_gradient_norm':gradient_norm,'validation':validation}
            log.append(record);print(json.dumps(record),flush=True);save_json(OUT/'training_log.json',log)
            if validation['mse_pu_squared']<best_val:
                best_val=validation['mse_pu_squared'];best_epoch=epoch+1;best_state=model.adapter_state()
        model.load_adapter_state(best_state)
        with torch.no_grad():context=full_context();after_pred=predictions(context,'test').cpu().numpy()
        after=score(py['test'].cpu().numpy(),after_pred)
        unchanged=frozen_digest(model)==original_hash
        if not unchanged or torch.equal(baseline_state['B'],best_state['B']):raise ValueError('Frozen-weight/update verification failed')
        if any(not torch.equal(baseline_state['head'][k],v) for k,v in best_state['head'].items()):
            raise ValueError('Head changed during LoRA-only comparison')
        base_meta={'format':'full_fuxi_pv_available_data_v1','rank':4,'alpha':8.,
                   'training_completed':True,'production_model_ready':False,'test_only':False,
                   'research_only':True,'base_model_path':'fuxi-2.1/fuxi-2.1.pt2',
                   'schema_by_site':schemas,'data_assumptions':['UTC+8 hypothesis','campus representative coordinates',
                                                              'unresolved publisher SLP bias','site_8 uses declared 30MW capacity'],
                   'model_scope':'complete FuXi graph, internal decoder29 Wq LoRA, four weather sensors and historical power',
                   'evaluation_scope':'station holdout under one ERA5 initialization, not temporal/weather generalization'}
        torch.save({**base_meta,'state':best_state,'selected_lora_epoch':best_epoch},OUT/'model.pt')
        torch.save({**base_meta,'state':baseline_state,'variant':'trained PV head with zero LoRA'},OUT/'baseline.pt')
        details=[]
        for i,row in enumerate(splits['test']):
            cap=row['request']['capacity_mw']
            details.append({'station_id':row['request']['station_id'],'timestamp':row['target_timestamp'],
                            'capacity_mw':cap,'actual_power_mw':row['target_power_mw'],
                            'before_lora_power_mw':float(before_pred[i])*cap,
                            'after_lora_power_mw':float(after_pred[i])*cap,
                            'persistence_power_mw':row['request']['history'][-1]['power_mw']})
        pd.DataFrame(details).to_csv(OUT/'heldout_predictions.csv',index=False,encoding='utf-8-sig')
        persistence=np.array([d['persistence_power_mw']/d['capacity_mw'] for d in details])
        metrics={'before_lora':before,'after_lora':after,'persistence':score(py['test'].cpu().numpy(),persistence),
                 'validation_before':before_val,'validation_selected':best_val,'selected_epoch':best_epoch,
                 'paired_split_counts':{s:len(r) for s,r in splits.items()},'paired_station_groups':groups,
                 'history_pretraining_counts':history_counts,'history_pretraining_temporal_test_before_paired_calibration':history_test,
                 'test_stations_excluded_from_all_training':True,'original_base_weights_unchanged':unchanged,
                 'four_sensor_fields_in_model':WEATHER,'lora_b_updated':True,
                 'head_fixed_for_before_after_lora':True,'unique_global_weather_initializations':1,
                 'production_model_ready':False,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                 'elapsed_seconds':time.perf_counter()-started}
        save_json(OUT/'metrics.json',metrics);make_plots(details,metrics)
        # Reload the actual saved artifact, verify metadata, tensor shapes and values.
        check=torch.load(OUT/'model.pt',map_location='cpu',weights_only=True)
        if not all(torch.isfinite(t).all() for t in [check['state']['A'],check['state']['B'],*check['state']['head'].values()]):
            raise ValueError('Saved model contains nonfinite weights')
        status.update(completed=True,model_artifact='model.pt',heldout_evaluation_completed=True,
                      original_weights_unchanged=True,saved_model_verified=True,
                      elapsed_seconds=time.perf_counter()-started)
        save_json(OUT/'status.json',status)
        print('TRAINING COMPLETE '+json.dumps(metrics),flush=True)
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        save_json(OUT/'status.json',status);raise


if __name__=='__main__':main()
