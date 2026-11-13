"""Verify held-out predictions and reload the trained full-FuXi adapter."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import ROOT,FullFuXiPV,temporal_features,state_from_array
from data import validate_example,load_weather


def main():
    out=ROOT/'models/full_fuxi_pv_available_data_v1'
    config=json.loads((out/'training_config.json').read_text(encoding='utf-8'))
    usage=json.loads((out/'source_usage.json').read_text(encoding='utf-8'))
    metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    rows=[json.loads(l) for l in (out/'paired_split_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=config['paired_station_groups']
    assert all(not(set(groups[a])&set(groups[b])) for a,b in [('train','val'),('train','test'),('val','test')])
    heldout=set(groups['val']+groups['test'])
    assert all(u['training_samples']==0 for u in usage if u['site_id'] in heldout)
    assert len([u for u in usage if u['source']=='original' and u['training_samples']>0])==8
    table=pd.read_csv(out/'heldout_predictions.csv')
    assert len(table)==125 and set(table.station_id)==set(groups['test'])
    assert not table[['station_id','timestamp']].duplicated().any()
    for name,column in [('before_lora','before_lora_power_mw'),('after_lora','after_lora_power_mw')]:
        # Training/evaluation tensors are float32 pu; restore that precision
        # before comparing against the double-precision reduction in score().
        predicted=(table[column]/table.capacity_mw).to_numpy(dtype=np.float32).astype(float)
        actual=(table.actual_power_mw/table.capacity_mw).to_numpy(dtype=np.float32).astype(float)
        error=predicted-actual
        np.testing.assert_allclose(float(np.abs(error).mean()),metrics[name]['mae_pu'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(float(np.sqrt((error**2).mean())),metrics[name]['rmse_pu'],rtol=1e-10,atol=1e-12)
    log=json.loads((out/'training_log.json').read_text(encoding='utf-8'))
    epochs=[r for r in log if r['stage']=='full_graph_lora']
    assert len(epochs)==12 and all(r['lora_b_gradient_norm']>0 for r in epochs)
    assert min(epochs,key=lambda r:r['validation']['mse_pu_squared'])['epoch']==metrics['selected_epoch']
    trained=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    baseline=torch.load(out/'baseline.pt',map_location='cpu',weights_only=True)
    assert not torch.equal(trained['state']['B'],baseline['state']['B'])
    assert all(torch.equal(trained['state']['head'][k],v) for k,v in baseline['state']['head'].items())
    assert all(torch.isfinite(v).all() for v in [trained['state']['A'],trained['state']['B'],*trained['state']['head'].values()])
    # Exercise a saved model on a genuine nonzero-power held-out example.
    test=[r for r in rows if r['split']=='test']
    row=max(test,key=lambda r:r['target_power_mw'])
    x,_,as_of,grid=validate_example(row)
    torch.set_num_threads(2)
    model=FullFuXiPV(ROOT/'fuxi-2.1/fuxi-2.1.pt2',weight_cache=out/'reload_weight_cache')
    model.load_adapter_state(trained['state'])
    values,init=load_weather(row['weather_path'],as_of)
    state=state_from_array(values,model.device);del values
    with torch.no_grad():
        result=float(model(state,temporal_features(init,model.device),torch.from_numpy(x[None]).to(model.device),grid)[0])*row['request']['capacity_mw']
    target=table.loc[(table.station_id==row['request']['station_id'])&(table.timestamp==row['target_timestamp'])].iloc[0]
    np.testing.assert_allclose(result,target.after_lora_power_mw,rtol=1e-5,atol=1e-9)
    with (out/'model.pt').open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    report={'passed':True,'station_holdout_verified':True,'original_stations_used':8,
            'heldout_test_examples':len(table),'full_graph_lora_steps':len(epochs),
            'sensor_fields':metrics['four_sensor_fields_in_model'],
            'metrics_independently_recomputed':True,'fixed_head_comparison_verified':True,
            'saved_model_reloaded_full_graph_forward_passed':True,
            'reload_example_station':row['request']['station_id'],'reload_prediction_mw':result,
            'recorded_prediction_mw':float(target.after_lora_power_mw),
            'model_sha256':digest,'production_model_ready':False}
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
