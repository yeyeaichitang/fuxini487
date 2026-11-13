"""Build genuinely aligned PV/global-weather manifests. Missing metadata blocks pairing."""
from pathlib import Path
import argparse
from datetime import timezone
import json
import sys
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import prepare_history


def prepare(config_path,output):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    problems=[]
    selected={k:v for k,v in config['stations'].items() if v.get('include_in_training')}
    if not selected:
        problems.append('No stations selected for training')
    for site,meta in selected.items():
        for key in ['latitude','longitude','timezone']:
            if meta.get(key) is None:
                problems.append(f'{site}: missing {key}')
        if meta.get('latitude') is not None and not -90<=meta['latitude']<=90:
            problems.append(f'{site}: invalid latitude')
        if meta.get('longitude') is not None and not -180<=meta['longitude']<=360:
            problems.append(f'{site}: invalid longitude')
        if meta.get('timezone'):
            try:
                ZoneInfo(meta['timezone'])
            except Exception:
                problems.append(f'{site}: invalid timezone')
        if not meta.get('metadata_verified'):
            problems.append(f'{site}: metadata is not verified')
    catalog=config.get('weather_inputs',[])
    if not catalog:
        problems.append('No normalized global weather input files registered')
    weather=[]
    for item in catalog:
        path=ROOT/item['path']
        if not path.exists():
            problems.append(f'Weather input missing: {item["path"]}')
            continue
        with xr.open_dataset(path) as dataset:
            times=pd.DatetimeIndex(dataset.time.values).tz_localize('UTC')
            if dataset['input'].shape!=(2,85,721,1440) or times[1]-times[0]!=pd.Timedelta(hours=6):
                problems.append(f'Weather shape/time mismatch: {item["path"]}')
            weather.append((times[-1],str(path)))
    status={'ready':not problems,'problems':problems,'station_count':len(selected),
            'weather_input_count':len(weather),'real_pv_training_started':False,
            'station_alignment':'unavailable' if problems else 'metadata supplied'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'readiness.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    if problems:
        print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
        return False
    destination=output/'samples.jsonl'
    if destination.exists():
        raise ValueError('Manifest already exists; choose a new output directory')
    data_root=ROOT/config['pv_data_directory']
    weather.sort()
    counts={'train':0,'val':0,'test':0}
    skipped=0
    with destination.open('w',encoding='utf-8') as stream:
        for site,meta in selected.items():
            schema=json.loads((data_root/'datasets'/f'{site}_schema.json').read_text(encoding='utf-8'))
            df=pd.read_csv(data_root/'cleaned'/f'{site}.csv',low_memory=False,parse_dates=['timestamp'])
            dates=pd.DatetimeIndex(df.timestamp).tz_localize(meta['timezone'],ambiguous='NaT',nonexistent='NaT')
            for split in counts:
                with np.load(data_root/'datasets'/f'{site}.npz',allow_pickle=False) as arrays:
                    ids=arrays['target_row_'+split]
                for target in ids:
                    window=dates[target-16:target+1]
                    if window.isna().any():
                        skipped+=1
                        continue
                    as_of=window[-2].to_pydatetime()
                    available=[w for w in weather if pd.Timedelta(0)<=pd.Timestamp(as_of)-w[0]<=pd.Timedelta(hours=6)]
                    if not available:
                        skipped+=1
                        continue
                    init,path=available[-1]
                    history=[]
                    for i in range(target-16,target):
                        row={key:float(df.iloc[i][key]) for key in ['temperature_c','pressure_hpa','humidity_percent','irradiance_w_m2','power_mw']}
                        row['timestamp']=dates[i].isoformat()
                        history.append(row)
                    request={'station_id':site,'capacity_mw':schema['capacity_mw'],'as_of':as_of.isoformat(),
                             'interval_minutes':15,'history':history}
                    prepare_history(request,schema,training=True)
                    row={'request':request,'target_timestamp':dates[target].isoformat(),
                         'target_power_mw':float(df.iloc[target].power_mw),'split':split,
                         'weather_path':path,'weather_init_utc':init.isoformat(),
                         'latitude':meta['latitude'],'longitude':meta['longitude'],
                         'schema':schema,'target_source':'observed historical power, no imputation',
                         'pv_source_sha256':schema['source_sha256']}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    counts[split]+=1
    status.update({'samples':counts,'skipped_no_weather_or_ambiguous_time':skipped,
                   'ready':all(counts.values()),'station_alignment':'paired',
                   'evaluation_scope':'retrospective ERA5 hindcast, not live weather availability'})
    if not all(counts.values()):
        status['problems'].append('Need actual matched samples in train, validation and test periods')
    (output/'readiness.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    return status['ready']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('stations.json'))
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/full_fuxi_pv_aligned')
    args=parser.parse_args()
    raise SystemExit(0 if prepare(args.config,args.output) else 2)
