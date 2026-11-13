"""Offline model inference for the available-data full-FuXi PV checkpoint.

Model-side loading/preprocessing only; no HTTP service or rolling optimization.
"""
from pathlib import Path
from datetime import timedelta
import argparse
import json
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import ROOT,FullFuXiPV,state_from_array,temporal_features
from contract import DataIssue,prepare_history,failure
from data import load_weather,station_grid


def predict_power(request,checkpoint,context,cache_directory):
    """Return the agreed +15-minute power output from a real model forward."""
    version='full-fuxi-pv-available-data-v1'
    try:
        if not isinstance(request,dict):
            raise DataIssue('data_invalid','INVALID_REQUEST','A station-history object is required')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved.get('format')!='full_fuxi_pv_available_data_v1' or not saved.get('training_completed'):
            raise DataIssue('model_error','CHECKPOINT_MISMATCH','A trained available-data PV checkpoint is required')
        site=request.get('station_id')
        if not isinstance(site,str) or site not in saved['schema_by_site']:
            raise DataIssue('station_not_validated','STATION_NOT_SUPPORTED',
                            'Station is absent from the trained schemas; DG labels must not be substituted for source stations')
        schema=saved['schema_by_site'][site]
        x,quality,as_of=prepare_history(request,schema)
        if abs(request['capacity_mw']-schema['capacity_mw'])>1e-6:
            raise DataIssue('station_not_validated','CAPACITY_MISMATCH','Use the actual trained station capacity')
        if not isinstance(context,dict) or context.get('station_id')!=site:
            raise DataIssue('data_insufficient','STATION_CONTEXT_MISSING','Model-side context must identify this station')
        if any(context.get(key) is None for key in ['weather_path','latitude','longitude']):
            raise DataIssue('data_insufficient','WEATHER_CONTEXT_MISSING','Contemporaneous global weather and station location are required')
        grid=station_grid(context['latitude'],context['longitude'])
        weather_path=Path(context['weather_path'])
        if not weather_path.is_absolute():weather_path=ROOT/weather_path
        try:
            values,init=load_weather(weather_path,as_of)
        except (OSError,ValueError) as error:
            raise DataIssue('data_insufficient','WEATHER_CONTEXT_INVALID',str(error)) from error
        torch.set_num_threads(2)
        model=FullFuXiPV(ROOT/saved['base_model_path'],rank=saved['rank'],alpha=saved['alpha'],weight_cache=cache_directory)
        model.load_adapter_state(saved['state'])
        state=state_from_array(values,model.device);del values
        with torch.no_grad():
            pu=float(model(state,temporal_features(init,model.device),torch.from_numpy(x[None]).to(model.device),grid)[0])
        if not np.isfinite(pu):
            raise DataIssue('model_error','NONFINITE_OUTPUT','Prediction is nonfinite')
        return {'status':'ok','station_id':site,'based_on':as_of.isoformat(),
                'interval_minutes':15,'model_name':'pv-forecast','model_version':version,
                'predictions':[{'timestamp':(as_of+timedelta(minutes=15)).isoformat(),
                                'power_mw':float(np.clip(pu,0,1)*request['capacity_mw'])}],
                'data_quality':quality,'weather_context_utc':init.isoformat(),
                'research_only':True,'production_accuracy_validated':False}
    except DataIssue as issue:
        return failure(request,issue,version)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--context',type=Path,required=True,help='Model-side location and global weather input, separate from the agreed external request')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/full_fuxi_pv_available_data_v1/model.pt')
    parser.add_argument('--cache-directory',type=Path,required=True,help='New, writable temporary model weight cache')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=predict_power(json.loads(args.input.read_text(encoding='utf-8')),args.checkpoint,
                         json.loads(args.context.read_text(encoding='utf-8')),args.cache_directory)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    raise SystemExit(0 if result['status']=='ok' else 2)
