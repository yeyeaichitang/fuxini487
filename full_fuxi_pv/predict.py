"""Runtime API for a genuinely PV-trained full-FuXi adapter and hardware history."""
from pathlib import Path
import argparse
import json
import sys
from datetime import timedelta
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import FullFuXiPV,ROOT,state_from_array,temporal_features
from data import load_weather,station_grid
from contract import prepare_history,DataIssue,failure


class PVPredictor:
    def __init__(self,checkpoint,stations_config,cache_directory):
        self.checkpoint=Path(checkpoint)
        self.object=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        if (self.object.get('format')!='full_fuxi_pv_lora_v1'
                or not self.object.get('training_completed') or self.object.get('test_only',True)):
            raise ValueError('A genuinely PV-trained full-graph checkpoint is required; smoke/weather-only checkpoints are rejected')
        self.config=json.loads(Path(stations_config).read_text(encoding='utf-8'))
        self.cache_directory=Path(cache_directory)
        self.model=None
        self.version='full-fuxi-pv-'+self.checkpoint.parent.name

    def predict(self,request):
        try:
            if not isinstance(request,dict):
                raise DataIssue('data_invalid','INVALID_REQUEST','Request must be an object')
            site=request.get('station_id')
            if not isinstance(site,str) or not site:
                raise DataIssue('data_invalid','INVALID_STATION','station_id must be a nonempty string')
            if site not in self.object['schema_by_site']:
                raise DataIssue('station_not_validated','STATION_NOT_VALIDATED','No trained station schema for this station, including unmapped DG stations')
            meta=self.config['stations'].get(site,{})
            if not meta.get('metadata_verified') or any(meta.get(k) is None for k in ('latitude','longitude','timezone')):
                raise DataIssue('data_insufficient','STATION_METADATA_MISSING','Station coordinates and timezone are required')
            schema=self.object['schema_by_site'][site]
            x,quality,as_of=prepare_history(request,schema)
            if abs(request['capacity_mw']-schema['capacity_mw'])>1e-6:
                raise DataIssue('station_not_validated','CAPACITY_MISMATCH','Capacity differs from the training station')
            # The backend can update this catalog with fresh normalized weather.
            # The external request schema remains the agreed station-history JSON.
            choices=[]
            import xarray as xr
            from datetime import datetime,timezone
            for entry in self.config.get('weather_inputs',[]):
                path=ROOT/entry['path']
                with xr.open_dataset(path) as dataset:
                    init=datetime.fromisoformat(str(dataset.time.values[-1].astype('datetime64[s]'))).replace(tzinfo=timezone.utc)
                if timedelta(0)<=as_of-init<=timedelta(hours=6):
                    choices.append((init,path))
            if not choices:
                raise DataIssue('data_insufficient','WEATHER_CONTEXT_MISSING','No contemporaneous full global FuXi input available')
            _,path=max(choices,key=lambda entry:entry[0])
            values,init=load_weather(path,as_of)
            if self.model is None:
                self.model=FullFuXiPV(ROOT/self.object['base_model_path'],rank=self.object['rank'],alpha=self.object['alpha'],weight_cache=self.cache_directory)
                self.model.load_adapter_state(self.object['state'])
            state=state_from_array(values,self.model.device)
            history=torch.from_numpy(x[None]).to(self.model.device)
            with torch.no_grad():
                result=float(self.model(state,temporal_features(init,self.model.device),history,
                                        station_grid(meta['latitude'],meta['longitude']))[0])
            if not np.isfinite(result):
                raise DataIssue('model_error','NONFINITE_OUTPUT','Nonfinite model output')
            return {'status':'ok','station_id':site,'based_on':as_of.isoformat(),'interval_minutes':15,
                    'model_name':'pv-forecast','model_version':self.version,
                    'predictions':[{'timestamp':(as_of+timedelta(minutes=15)).isoformat(),
                                    'power_mw':float(np.clip(result,0,1)*request['capacity_mw'])}],
                    'data_quality':quality,'model_scope':'complete original FuXi graph with internal LoRA and PV head',
                    'weather_context_utc':init.isoformat()}
        except DataIssue as issue:
            return failure(request,issue,self.version)
        except (ValueError,RuntimeError,OSError,KeyError) as error:
            return failure(request,DataIssue('model_error','MODEL_OR_WEATHER_ERROR',str(error)),self.version)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--stations-config',type=Path,default=Path(__file__).with_name('stations.json'))
    parser.add_argument('--cache-directory',type=Path,required=True,help='Fresh writable weight cache directory')
    args=parser.parse_args()
    predictor=PVPredictor(args.checkpoint,args.stations_config,args.cache_directory)
    print(json.dumps(predictor.predict(json.loads(args.input.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))
