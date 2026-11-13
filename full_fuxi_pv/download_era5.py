"""Download a small real ERA5 global input pair from Google Research's public mirror.

Default: 2019-01-01 00/06 UTC inputs, plus independent 12 UTC t2m truth.
Uses native Zarr-v2 chunk metadata and numcodecs; no account or secrets required.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'output/weather_dependencies'))
sys.path.insert(0,str(ROOT/'fuxi-2.1'))
from numcodecs import get_codec
from variables import C85_CHANNEL_NAMES,PRESSURE_LEVELS,PL_VARIABLES,NORM_EPSILON

BASE='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
SURFACE={'msl':'mean_sea_level_pressure','t2m':'2m_temperature','d2m':'2m_dewpoint_temperature',
         'sst':'sea_surface_temperature','u10m':'10m_u_component_of_wind','v10m':'10m_v_component_of_wind',
         'u100m':'100m_u_component_of_wind','v100m':'100m_v_component_of_wind',
         'lcc':'low_cloud_cover','mcc':'medium_cloud_cover','hcc':'high_cloud_cover',
         'tcc':'total_cloud_cover','tcw':'total_column_water'}
DIAGNOSTIC=['ssr','ssrd','fdir','ttr','tp']


class Archive:
    def __init__(self,cache):
        self.cache=cache
        self.objects={}
        self.metadata=json.loads(self.fetch('.zmetadata').read_text())['metadata']

    def fetch(self,key):
        path=self.cache/key
        path.parent.mkdir(parents=True,exist_ok=True)
        url=BASE+'/'+key
        if not path.exists():
            part=path.with_name(path.name+'.partial')
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=60) as response,part.open('wb') as output:
                        shutil.copyfileobj(response,output,1024*1024)
                    part.replace(path)
                    break
                except Exception:
                    if attempt==2:
                        raise
                    time.sleep(2)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        self.objects[key]={'url':url,'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
        return path

    def chunk(self,var,indices):
        meta=self.metadata[var+'/.zarray']
        if meta.get('filters') or meta['order']!='C' or meta['zarr_format']!=2:
            raise ValueError('Unsupported source encoding')
        key=var+'/'+'.'.join(map(str,indices))
        encoded=self.fetch(key).read_bytes()
        decoded=get_codec(meta['compressor']).decode(encoded) if meta['compressor'] else encoded
        return np.frombuffer(decoded,dtype=meta['dtype']).reshape(meta['chunks'])

    def coordinate(self,var):
        meta=self.metadata[var+'/.zarray']
        if meta['shape'][0]>meta['chunks'][0]:
            raise ValueError('Coordinate spans multiple chunks')
        return self.chunk(var,[0])[:meta['shape'][0]].copy()

    def time_index(self,when):
        attrs=self.metadata['time/.zattrs']
        if attrs['units']!='hours since 1900-01-01 00:00:00':
            raise ValueError('Unexpected time origin')
        index=int((when-datetime(1900,1,1,tzinfo=timezone.utc)).total_seconds()/3600)
        size=self.metadata['time/.zarray']['chunks'][0]
        actual=self.chunk('time',[index//size])[index%size]
        if actual!=index:
            raise ValueError('Time coordinate verification failed')
        return index

    def field(self,var,index):
        meta=self.metadata[var+'/.zarray']
        if meta['chunks'][0]!=1 or meta['shape'][1:]!=meta['chunks'][1:]:
            raise ValueError('Expected one full global field per time chunk')
        return self.chunk(var,[index]+[0]*(len(meta['shape'])-1))[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',default='2019-01-01T00:00:00+00:00')
    parser.add_argument('--output',type=Path,default=ROOT/'data/weather/era5_20190101')
    args=parser.parse_args()
    start=datetime.fromisoformat(args.start)
    if start.tzinfo is None or start.utcoffset()!=timedelta(0) or start.minute or start.second or start.hour%6:
        raise ValueError('Start must be an explicit UTC six-hour boundary')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'input.nc').exists():
        raise ValueError('Final input already exists; choose a new output')
    archive=Archive(args.output/'source_chunks')
    lat,lon,levels=[archive.coordinate(var) for var in ['latitude','longitude','level']]
    np.testing.assert_allclose(lat,np.linspace(90,-90,721))
    np.testing.assert_allclose(lon,np.arange(1440)/4)
    selected=[list(levels).index(level) for level in PRESSURE_LEVELS]
    with xr.open_dataarray(ROOT/'fuxi-2.1/mean.nc') as array:
        means=array.values.copy()
    with xr.open_dataarray(ROOT/'fuxi-2.1/std.nc') as array:
        stds=array.values.copy()
    # Memory-map the output while downloading one source field at a time.
    scratch=args.output/'normalized_input.f32'
    values=np.memmap(scratch,mode='w+',dtype=np.float32,shape=(2,85,721,1440))
    times=[start,start+timedelta(hours=6)]
    for frame,when in enumerate(times):
        index=archive.time_index(when)
        for short,long_name,_ in PL_VARIABLES:
            print(f'DOWNLOAD {when.isoformat()} {long_name}',flush=True)
            field=archive.field(long_name,index)
            for source_level,level in zip(selected,PRESSURE_LEVELS):
                channel=C85_CHANNEL_NAMES.index(f'{short}{level}')
                raw=field[source_level]*(1000 if short=='q' else 1)
                values[frame,channel]=(raw-means[channel])/(stds[channel]+NORM_EPSILON)
            del field
        winds={}
        for short,long_name in SURFACE.items():
            print(f'DOWNLOAD {when.isoformat()} {long_name}',flush=True)
            field=archive.field(long_name,index)
            channel=C85_CHANNEL_NAMES.index(short)
            values[frame,channel]=(field-means[channel])/(stds[channel]+NORM_EPSILON)
            if short in ['u10m','v10m','u100m','v100m']:
                winds[short]=field.copy()
            del field
        for height in ['10m','100m']:
            speed=np.hypot(winds['u'+height],winds['v'+height])
            channel=C85_CHANNEL_NAMES.index('ws'+height)
            values[frame,channel]=(speed-means[channel])/(stds[channel]+NORM_EPSILON)
        for short in DIAGNOSTIC:
            # Original graph zeros these channels before the backbone. These are
            # explicit ignored placeholders, NEVER fabricated observations/targets.
            values[frame,C85_CHANNEL_NAMES.index(short)]=0
        values.flush()
    for i,name in enumerate(C85_CHANNEL_NAMES):
        if np.isinf(values[:,i]).any() or (name!='sst' and np.isnan(values[:,i]).any()):
            raise ValueError(f'Invalid downloaded/normalized channel {name}')
    time_values=np.array([np.datetime64(t.replace(tzinfo=None),'ns') for t in times])
    attrs={'source':'ECMWF ERA5 final analysis via Google Research ARCO public mirror',
           'source_url':BASE,'normalization':'fuxi-2.1 mean/std zscore',
           'diagnostic_input_placeholders':','.join(DIAGNOSTIC),
           'diagnostic_policy':'zeroed internally by original FuXi; not observed radiation/precipitation',
           'time_zone':'UTC','specific_humidity_conversion':'kg/kg to g/kg, multiply 1000',
           'station_alignment':'not performed; station coordinates/timezones unavailable'}
    dataset=xr.Dataset({'input':(('time','channel','lat','lon'),values)},
                       coords={'time':time_values,'channel':C85_CHANNEL_NAMES,'lat':lat,'lon':lon},attrs=attrs)
    dataset.to_netcdf(args.output/'input.nc',engine='netcdf4')
    dataset.close()
    del dataset,values
    # A real independent future analysis for weather-only gradient validation.
    truth_time=start+timedelta(hours=12)
    print(f'DOWNLOAD {truth_time.isoformat()} independent t2m truth',flush=True)
    truth=archive.field('2m_temperature',archive.time_index(truth_time)).copy()
    xr.Dataset({'t2m':(('lat','lon'),truth,{'units':'K'})},coords={'lat':lat,'lon':lon},
               attrs={'valid_time':truth_time.isoformat(),'source':attrs['source'],'source_url':BASE}).to_netcdf(args.output/'t2m_truth.nc')
    result={'source':attrs['source'],'retrieval_manifest_version':1,'input_times_utc':[t.isoformat() for t in times],
            'truth_time_utc':truth_time.isoformat(),'channel_count':85,'observed_or_derived_prognostic_channels':80,
            'ignored_input_channels':DIAGNOSTIC,'station_alignment_complete':False,
            'source_objects':archive.objects,'source_metadata_attributes':archive.metadata['.zattrs'],
            'input_bytes':(args.output/'input.nc').stat().st_size,
            'input_sha256':hashlib.sha256((args.output/'input.nc').read_bytes()).hexdigest()}
    (args.output/'download_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # Remove only the temporary memmap created by this invocation.
    scratch.unlink()
    print(f'DOWNLOAD COMPLETE: {args.output}',flush=True)


if __name__=='__main__':
    main()
