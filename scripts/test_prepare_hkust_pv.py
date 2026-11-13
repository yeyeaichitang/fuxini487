"""Check causal aggregation, missing data and split boundaries with known values."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from prepare_hkust_pv import WEATHER, target_rows, trailing_weather


class PreparationTests(unittest.TestCase):
    def test_trailing_window_never_reads_future_and_requires_all_minutes(self):
        dates = pd.date_range('2022-01-01', periods=31, freq='min')
        source = pd.DataFrame({'temperature_c': np.arange(31, dtype=float)}, index=dates)
        aggregated, counts = trailing_weather(source)
        self.assertAlmostEqual(aggregated.loc['2022-01-01 00:15', 'temperature_c'], 8)
        self.assertEqual(counts.loc['2022-01-01 00:15', 'temperature_c'], 15)
        source.loc['2022-01-01 00:16', 'temperature_c'] = 10000
        after, _ = trailing_weather(source)
        self.assertEqual(after.loc['2022-01-01 00:15', 'temperature_c'], 8)
        source.loc['2022-01-01 00:01', 'temperature_c'] = np.nan
        missing, _ = trailing_weather(source)
        self.assertTrue(np.isnan(missing.loc['2022-01-01 00:15', 'temperature_c']))

    def frame(self, start='2022-01-01', periods=40):
        a = pd.DataFrame({'timestamp': pd.date_range(start, periods=periods, freq='15min')})
        for col in WEATHER:
            a[col] = 1.0
        a['power_mw'] = 0.5
        return a

    def test_nanosecond_and_microsecond_dates_produce_same_windows(self):
        a = self.frame()
        self.assertEqual(len(target_rows(a, 1)['train']), 24)
        a['timestamp'] = a.timestamp.astype('datetime64[ns]')
        self.assertEqual(len(target_rows(a, 1)['train']), 24)

    def test_wind_direction_wraps_around_north(self):
        dates = pd.date_range('2022-01-01', periods=16, freq='min')
        frame = pd.DataFrame({'wind_direction_deg': [0] + [359]*7 + [1]*8}, index=dates)
        values, _ = trailing_weather(frame)
        direction = values.loc['2022-01-01 00:15', 'wind_direction_deg']
        self.assertTrue(direction < 1 or direction > 359)

    def test_rejects_missing_history_and_invalid_targets(self):
        a = self.frame()
        a.loc[10, 'humidity_percent'] = np.nan
        ids = target_rows(a, 1)['train']
        self.assertTrue(all(not j - 16 <= 10 < j for j in ids))
        a.loc[35, 'power_mw'] = np.nan
        self.assertNotIn(35, target_rows(a, 1)['train'])

    def test_no_window_crosses_split_or_gap(self):
        a = self.frame('2022-06-30 20:00')
        ids = target_rows(a, 1)
        self.assertEqual(len(ids['train']), 0)
        self.assertEqual(int(ids['val'][0]), 32)
        a = self.frame().drop(index=20).reset_index(drop=True)
        ids = target_rows(a, 1)['train']
        self.assertTrue(all(j < 20 or j >= 36 for j in ids))


if __name__ == '__main__':
    unittest.main()
