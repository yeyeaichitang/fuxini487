"""Boundary tests for PV preparation; never modify real data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).with_name('prepare_all_pv_data.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_timestamp_units_missing_power_gaps_and_split_boundaries(self):
        times = pd.DatetimeIndex(np.concatenate([
            pd.date_range('2019-12-31 08:00', periods=64, freq='15min').to_numpy(),
            pd.date_range('2020-01-01', periods=64, freq='15min').to_numpy(),
            pd.date_range('2020-04-01', periods=64, freq='15min').to_numpy(),
        ])).astype('datetime64[us]')
        ramp = np.arange(len(times)) % 20
        frame = pd.DataFrame({'timestamp': times, 'source_row': np.arange(2,len(times)+2),
                              'temperature_c': 10.0+ramp, 'pressure_hpa': 900+ramp,
                              'humidity_percent': 40+ramp, 'irradiance_w_m2': 100+ramp,
                              'power_mw': 5+ramp/10})
        frame.loc[20,'power_mw'] = -0.01
        frame.loc[21,'humidity_percent'] = -99
        frame.loc[22,'temperature_c'] = np.inf
        frame.loc[23,'power_mw'] = np.nan
        frame.loc[70,'power_mw'] = 51
        # Real zero is retained, while negative and missing observations stay invalid.
        frame.loc[24,'power_mw'] = 0
        frame = frame.drop(index=90).reset_index(drop=True)
        source = {'site_id':'test_site', 'source_file':'test.xlsm', 'source_sheet':'pv',
                  'capacity_mw':50,'capacity_status':'test', 'source_sha256':'test',
                  'duplicate_capacity_conflict':False,'raw_rows':len(frame),'data':frame}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for sub in ('quarantine','quality','cleaned','datasets'):
                (out/sub).mkdir()
            cleaned, report = prepare.clean_source(source,out)
            self.assertEqual(len(cleaned),len(frame))
            self.assertEqual(report['negative_power_rows'],1)
            self.assertTrue(np.isnan(cleaned.loc[20,'power_mw']))
            self.assertEqual(cleaned.loc[24,'power_mw'],0)
            self.assertEqual(cleaned.loc[20,'power_mw_raw'],-0.01)
            index = prepare.build_samples(cleaned,source,out,report)
            self.assertEqual(set(index.split),{'train','val','test'})
            train_mean = cleaned.loc[cleaned.split=='train', prepare.WEATHER].mean().to_numpy(np.float32)
            with np.load(out/'datasets/test_site.npz') as data:
                np.testing.assert_allclose(data['weather_mean'],train_mean)
                for split in ('train','val','test'):
                    ids=data['target_row_'+split]
                    self.assertGreater(len(ids),0)
                    for target in ids:
                        window=cleaned.iloc[target-16:target+1]
                        self.assertEqual(set(window.split),{split})
                        self.assertTrue((window.timestamp.diff().dropna()==pd.Timedelta(minutes=15)).all())
                    self.assertTrue(np.isfinite(data['X_'+split]).all())
                    self.assertTrue(np.isfinite(data['y_'+split]).all())

    def test_conflicting_duplicate_and_invalid_timestamps_quarantined(self):
        df=pd.DataFrame({'source_row':[2,3,4,5], 'timestamp':['2019-01-01','2019-01-01','invalid','2019-01-01 00:01:00'],
                         'temperature_c':[10]*4,'pressure_hpa':[900]*4,'humidity_percent':[50]*4,
                         'irradiance_w_m2':[100]*4,'power_mw':[1,2,3,4]})
        source={'site_id':'test','source_file':'test.xlsm','source_sheet':'pv','capacity_mw':50,'data':df}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for sub in ('quarantine','quality','cleaned'):
                (out/sub).mkdir()
            cleaned,report=prepare.clean_source(source,out)
            self.assertEqual(len(cleaned),0)
            self.assertEqual(report['quarantined_timestamp_rows'],4)


if __name__=='__main__':
    unittest.main()
