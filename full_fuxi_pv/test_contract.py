from pathlib import Path
import copy
from datetime import datetime,timedelta,timezone
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import prepare_history,DataIssue
from data import load_weather,station_grid
from prepare import prepare


def fixture():
    start=datetime(2020,1,1,tzinfo=timezone.utc)
    rows=[{'timestamp':(start+timedelta(minutes=15*i)).isoformat(),'temperature_c':20.,
           'pressure_hpa':900.,'humidity_percent':50.,'irradiance_w_m2':100.,'power_mw':2.} for i in range(16)]
    request={'station_id':'site_1_b','capacity_mw':50.,'as_of':rows[-1]['timestamp'],'interval_minutes':15,'history':rows}
    return request,{'weather_mean':[0.,0.,0.,0.],'weather_std':[1.,1.,1.,1.]}


class ContractTests(unittest.TestCase):
    def test_four_hardware_features_are_in_actual_tensor(self):
        request,schema=fixture()
        x,quality,_=prepare_history(request,schema)
        np.testing.assert_allclose(x[-1],[20,900,50,100,0.04])
        self.assertEqual(quality['power_imputed_count'],0)

    def test_short_power_gap_only_runtime_and_never_modifies_truth(self):
        request,schema=fixture()
        request['history'][-1]['power_mw']=None
        request['history'][-2]['power_mw']=None
        original=copy.deepcopy(request)
        x,quality,_=prepare_history(request,schema)
        self.assertEqual(quality['power_imputed_count'],2)
        self.assertEqual(request,original)
        self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(DataIssue):
            prepare_history(request,schema,training=True)

    def test_three_missing_power_or_missing_weather_rejected(self):
        request,schema=fixture()
        for row in request['history'][-3:]:
            row['power_mw']=None
        with self.assertRaises(DataIssue):
            prepare_history(request,schema)
        request,schema=fixture()
        request['history'][-1]['pressure_hpa']=None
        with self.assertRaises(DataIssue):
            prepare_history(request,schema)

    def test_time_without_timezone_and_future_weather_rejected(self):
        request,schema=fixture()
        request['history'][0]['timestamp']='2020-01-01T00:00:00'
        with self.assertRaises(DataIssue):
            prepare_history(request,schema)
        weather=Path(__file__).resolve().parents[1]/'data/weather/era5_20190101/input.nc'
        with self.assertRaises(ValueError):
            load_weather(weather,datetime(2019,1,1,tzinfo=timezone.utc))

    def test_invalid_numbers_and_power_capacity(self):
        request,schema=fixture()
        for field,value in [('temperature_c',True),('humidity_percent',101.),('irradiance_w_m2',float('nan')),('power_mw',51.)]:
            altered=copy.deepcopy(request)
            altered['history'][0][field]=value
            with self.assertRaises(DataIssue):
                prepare_history(altered,schema)

    def test_station_coordinates_and_pairing_block(self):
        self.assertEqual(station_grid(0,0),(360,0))
        self.assertEqual(station_grid(90,-0.25),(0,1439))
        with self.assertRaises(DataIssue):
            station_grid(None,0)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            config={'stations':{'site_1_b':{'include_in_training':True,'latitude':None,'longitude':None,'timezone':None,'metadata_verified':False}},'weather_inputs':[]}
            path=directory/'config.json';path.write_text(json.dumps(config),encoding='utf-8')
            self.assertFalse(prepare(path,directory/'prepared'))
            self.assertFalse((directory/'prepared/samples.jsonl').exists())


if __name__=='__main__':
    unittest.main()
