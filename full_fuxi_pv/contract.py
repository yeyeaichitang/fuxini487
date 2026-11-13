"""Station sensor contract shared by full-FuXi training and runtime prediction."""
from datetime import datetime,timedelta
import math
import numpy as np

WEATHER=('temperature_c','pressure_hpa','humidity_percent','irradiance_w_m2')
LIMITS=((-50,70),(300,1100),(0,100),(0,1500))


class DataIssue(ValueError):
    def __init__(self,status,code,message):
        super().__init__(message)
        self.status,self.code=status,code


def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise DataIssue('data_invalid','INVALID_NUMBER',f'{name}: expected a finite number')
    return float(value)


def timestamp(value):
    try:
        result=datetime.fromisoformat(value.replace('Z','+00:00'))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError()
        if result.minute%15 or result.second or result.microsecond:
            raise ValueError()
        return result
    except (ValueError,TypeError,AttributeError):
        raise DataIssue('data_invalid','INVALID_TIMESTAMP','ISO8601 with timezone on a 15-minute boundary required')


def prepare_history(request,schema,training=False):
    if not isinstance(request,dict):
        raise DataIssue('data_invalid','INVALID_REQUEST','Request must be an object')
    if not isinstance(request.get('station_id'),str) or not request['station_id']:
        raise DataIssue('data_invalid','INVALID_STATION','station_id required')
    capacity=number(request.get('capacity_mw'),'capacity_mw')
    if capacity<=0:
        raise DataIssue('data_invalid','INVALID_CAPACITY','capacity_mw must be positive')
    if request.get('interval_minutes')!=15:
        raise DataIssue('data_invalid','INVALID_INTERVAL','interval_minutes must be 15')
    history=request.get('history')
    if not isinstance(history,list) or len(history)!=16:
        raise DataIssue('data_insufficient','INSUFFICIENT_HISTORY','16 measured history records required')
    if not all(isinstance(row,dict) for row in history):
        raise DataIssue('data_invalid','INVALID_HISTORY','Each record must be an object')
    times=[timestamp(row.get('timestamp')) for row in history]
    if any(b-a!=timedelta(minutes=15) for a,b in zip(times,times[1:])):
        raise DataIssue('data_invalid','IRREGULAR_TIMESTAMPS','Consecutive 15-minute records required')
    as_of=timestamp(request.get('as_of'))
    if as_of!=times[-1]:
        raise DataIssue('data_invalid','AS_OF_MISMATCH','as_of must match the last history timestamp')
    mean,std=np.asarray(schema['weather_mean']),np.asarray(schema['weather_std'])
    if mean.shape!=(4,) or std.shape!=(4,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid training normalization schema')
    features=[]
    filled=[]
    previous=None
    last_actual=None
    run=0
    for index,row in enumerate(history):
        weather=[]
        for j,(key,(low,high)) in enumerate(zip(WEATHER,LIMITS)):
            if row.get(key) is None:
                raise DataIssue('data_insufficient','MISSING_WEATHER',f'{key} is missing; weather is never imputed')
            value=number(row[key],key)
            if not low<=value<=high:
                raise DataIssue('data_invalid','WEATHER_OUT_OF_RANGE',f'{key} outside [{low},{high}]')
            weather.append((value-mean[j])/std[j])
        if 'power_mw' not in row:
            raise DataIssue('data_invalid','MISSING_FIELD','power_mw field required; use null for missing measurements')
        if row['power_mw'] is None:
            run+=1
            if training or run>2 or previous is None:
                raise DataIssue('data_insufficient','MISSING_POWER','Cannot fill this power gap')
            power=previous
            filled.append(index)
        else:
            power=number(row['power_mw'],'power_mw')
            if not 0<=power<=capacity:
                raise DataIssue('data_invalid','POWER_OUT_OF_RANGE','Power outside station capacity')
            previous=power
            last_actual=times[index]
            run=0
        features.append(weather+[power/capacity])
    quality={'power_imputed_count':len(filled),'power_imputed_indices':filled,
             'last_actual_power_at':last_actual.isoformat() if last_actual else None,
             'filled_values_are_measurements':False,'input_quality':'power_forward_filled' if filled else 'measured'}
    return np.asarray(features,dtype=np.float32),quality,as_of


def failure(request,issue,version):
    request=request if isinstance(request,dict) else {}
    return {'status':issue.status,'error_code':issue.code,'message':str(issue),
            'station_id':request.get('station_id'),'based_on':request.get('as_of'),
            'interval_minutes':15,'model_name':'pv-forecast','model_version':version,'predictions':[]}
