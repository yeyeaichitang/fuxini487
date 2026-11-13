"""Validate global weather context and genuinely paired station examples."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import numpy as np
import xarray as xr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fuxi-2.1'))
from variables import C85_CHANNEL_NAMES
from contract import prepare_history,timestamp,number


def load_weather(path,as_of=None):
    with xr.open_dataset(path) as dataset:
        data=dataset['input']
        if data.dims!=('time','channel','lat','lon') or data.shape!=(2,85,721,1440):
            raise ValueError('Wrong FuXi global weather dimensions')
        if list(data.channel.values)!=C85_CHANNEL_NAMES:
            raise ValueError('Wrong FuXi channel order')
        np.testing.assert_allclose(data.lat.values,np.linspace(90,-90,721))
        np.testing.assert_allclose(data.lon.values,np.arange(1440)/4)
        if dataset.attrs.get('time_zone')!='UTC' or 'zscore' not in dataset.attrs.get('normalization',''):
            raise ValueError('Weather needs explicit UTC and FuXi normalization provenance')
        times=data.time.values.astype('datetime64[s]')
        if times[1]-times[0]!=np.timedelta64(6,'h'):
            raise ValueError('Weather input frames must be six hours apart')
        init=datetime.fromisoformat(str(times[-1])).replace(tzinfo=timezone.utc)
        if as_of is not None and not timedelta(0)<=as_of-init<=timedelta(hours=6):
            raise ValueError('Global input must not use future weather or be more than six hours old')
        values=data.values.copy()
    return values,init


def station_grid(latitude,longitude):
    latitude,longitude=number(latitude,'latitude'),number(longitude,'longitude')
    if not -90<=latitude<=90 or not -180<=longitude<=360:
        raise ValueError('Station coordinates out of bounds')
    return int(round((90-latitude)*4)),int(round((longitude%360)*4))%1440


def validate_example(row):
    if row['split'] not in ('train','val','test') or row.get('target_source')!='observed historical power, no imputation':
        raise ValueError('Unverified sample split or target provenance')
    request,schema=row['request'],row['schema']
    if request['station_id']!=schema['site_id'] or abs(request['capacity_mw']-schema['capacity_mw'])>1e-6:
        raise ValueError('Station/capacity mismatch')
    x,_,as_of=prepare_history(request,schema,training=True)
    target=number(row['target_power_mw'],'target_power_mw')
    if not 0<=target<=request['capacity_mw']:
        raise ValueError('Invalid target power')
    if timestamp(row['target_timestamp'])!=as_of+timedelta(minutes=15):
        raise ValueError('Target must be the next measured 15-minute observation')
    grid=station_grid(row['latitude'],row['longitude'])
    if not Path(row['weather_path']).is_file():
        raise ValueError('Missing global weather input')
    return x,target/request['capacity_mw'],as_of,grid
