"""Independently verify saved CSV and NPZ datasets against extracted observations."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['temperature_c', 'pressure_hpa', 'humidity_percent', 'irradiance_w_m2', 'power_mw']


def verify(folder):
    report = json.loads((folder/'data_report.json').read_text(encoding='utf-8'))
    counts = dict(train=0, val=0, test=0)
    checked = []
    for station in report['stations']:
        site = station['site_id']
        raw_file = ROOT/'data/raw'/station['source_file']
        assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == station['source_sha256']
        extracted = pd.read_csv(folder/'extracted'/f'{site}.csv', low_memory=False)
        clean = pd.read_csv(folder/'cleaned'/f'{site}.csv', parse_dates=['timestamp'], low_memory=False)
        rejected = pd.read_csv(folder/'quarantine'/f'{site}_timestamps.csv')
        assert len(extracted) == len(clean) + len(rejected) == station['raw_rows']
        assert not clean.timestamp.isna().any() and clean.timestamp.is_unique
        assert clean.timestamp.is_monotonic_increasing
        assert (clean.timestamp.astype('datetime64[ns]').astype('int64') % (15*60*10**9) == 0).all()
        source = extracted.set_index('source_row').loc[clean.source_row]
        limits = [(-50,70), (300,1100), (0,100), (0,1500), (0,station['capacity_mw'])]
        for field, (low, high) in zip(FIELDS, limits):
            raw = pd.to_numeric(source[field], errors='coerce').to_numpy(dtype=float)
            preserved = pd.to_numeric(clean[field+'_raw'], errors='coerce').to_numpy(dtype=float)
            np.testing.assert_allclose(preserved, raw, equal_nan=True)
            for actual, expected in zip(clean.loc[np.isnan(raw),field+'_raw'], source.loc[np.isnan(raw),field]):
                assert (pd.isna(actual) and pd.isna(expected)) or str(actual) == str(expected)
            valid = np.isfinite(raw) & (raw >= low) & (raw <= high)
            assert np.array_equal(clean[field+'_invalid'], ~valid)
            np.testing.assert_allclose(clean[field], np.where(valid, raw, np.nan), equal_nan=True)
        if station['duplicate_excluded']:
            other = pd.read_csv(folder/'extracted'/f'{station["canonical_site_id"]}.csv', low_memory=False)
            pd.testing.assert_frame_equal(extracted, other)
            assert not (folder/'datasets'/f'{site}.npz').exists()
            continue
        schema = json.loads((folder/'datasets'/f'{site}_schema.json').read_text(encoding='utf-8'))
        capacity = schema['capacity_mw']
        values = clean[FIELDS].to_numpy(dtype=np.float32)
        values[:,4] = (clean.power_mw / capacity).to_numpy(dtype=np.float32)
        ts = clean.timestamp.to_numpy(dtype='datetime64[ns]').astype('int64')
        with np.load(folder/'datasets'/f'{site}.npz', allow_pickle=False) as data:
            mean = clean.loc[clean.timestamp < '2020-01-01', FIELDS[:4]].mean().to_numpy(dtype=np.float32)
            std = clean.loc[clean.timestamp < '2020-01-01', FIELDS[:4]].std(ddof=0).to_numpy(dtype=np.float32)
            np.testing.assert_allclose(data['weather_mean'], mean, rtol=1e-6)
            np.testing.assert_allclose(data['weather_std'], std, rtol=1e-6)
            values[:,:4] = (values[:,:4] - mean) / std
            previous_latest = None
            for split, lower, upper in [('train',None,'2020-01-01'), ('val','2020-01-01','2020-04-01'), ('test','2020-04-01',None)]:
                ids, x, y = data['target_row_'+split], data['X_'+split], data['y_'+split]
                assert len(ids) > 0, f'{site}/{split}: empty training partition'
                assert x.shape == (len(ids),16,5) and y.shape == (len(ids),)
                assert np.isfinite(x).all() and np.isfinite(y).all()
                assert ((y >= 0) & (y <= 1)).all()
                assert np.array_equal(data['time_'+split], ts[ids])
                starts = clean.timestamp.iloc[ids-16]
                ends = clean.timestamp.iloc[ids]
                if lower:
                    assert (starts >= pd.Timestamp(lower)).all()
                if upper:
                    assert (ends < pd.Timestamp(upper)).all()
                if previous_latest is not None:
                    assert starts.min() > previous_latest
                previous_latest = ends.max()
                for first in range(0,len(ids),2048):
                    targets = ids[first:first+2048]
                    rows = targets[:,None] + np.arange(-16,0)
                    np.testing.assert_allclose(x[first:first+2048],values[rows],rtol=2e-5,atol=2e-5)
                    np.testing.assert_allclose(y[first:first+2048], values[targets,4],rtol=1e-6)
                    assert (np.diff(ts[targets[:,None]+np.arange(-16,1)],axis=1) == 15*60*10**9).all()
                # Independently recompute the complete set, so omissions also fail.
                eligible = clean.timestamp.notna()
                if lower:
                    eligible &= clean.timestamp >= pd.Timestamp(lower)
                if upper:
                    eligible &= clean.timestamp < pd.Timestamp(upper)
                block = np.flatnonzero(eligible)
                candidates = block[16:]
                good = []
                observed = clean[FIELDS].notna().all(axis=1).to_numpy()
                target_valid = clean.power_mw.notna().to_numpy()
                for target in candidates:
                    if (observed[target-16:target].all() and target_valid[target]
                            and ts[target]-ts[target-16] == 16*15*60*10**9):
                        good.append(target)
                assert np.array_equal(ids,np.asarray(good,dtype=np.int64))
                assert len(ids) == station['splits'][split]['samples']
                counts[split] += len(ids)
        checked.append(site)
        print(f'VERIFIED {site}',flush=True)
    for split, count in counts.items():
        index = pd.read_csv(folder/f'pv_{split}_samples.csv')
        assert len(index) == count == report['samples'][split]
        assert not index.duplicated(['site_id','target_timestamp']).any()
        assert set(index.site_id) == set(checked)
    merged = pd.read_csv(folder/'pv_all_sites_cleaned.csv', usecols=['site_id','timestamp'])
    assert len(merged) == report['canonical_cleaned_rows']
    assert not merged.duplicated(['site_id','timestamp']).any()
    assert set(merged.site_id) == set(checked)
    result = {'passed':True, 'source_workbooks_verified':report['source_workbooks'],
              'datasets_verified':checked,'samples_verified':counts,
              'checks':['source hashes unchanged','row accounting','raw values preserved',
                        'bounds and invalid flags','duplicate exclusion','train-only normalization',
                        'all sample features and labels reconstructed','all eligible windows present',
                        '15-minute continuity','no split overlap','no NaN or Inf in training arrays']}
    (folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    report['data_preparation_complete'] = True
    report['validation_pending'] = False
    (folder/'data_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed/pv_all_sites_v2')
    verify(parser.parse_args().data)
