"""Streaming city aggregation with content-checked, configuration-aware cache."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .core import digest, clean_signature

BASE = 'https://d37ci6vzurychx.cloudfront.net'
COLUMNS = ['tpep_pickup_datetime', 'PULocationID', 'trip_distance', 'total_amount']


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])))
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with session.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        if not temporary.stat().st_size:
            raise ValueError('Пустой ответ источника')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        session.close()
    return destination


def cache_read(path, signature):
    path = Path(path)
    meta_path = path.with_suffix('.json')
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['signature'] != signature or meta['aggregate_sha256'] != digest(path):
            return None
        table = pd.read_parquet(path)
        if (table.trips_count.sum() != meta['kept_rows'] or table.pickup_hour.duplicated().any()
                or meta['raw_rows'] != sum(meta[k] for k in
                    ['kept_rows', 'rejected_date', 'rejected_values', 'rejected_zone'])):
            return None
        return table, meta
    except (ValueError, KeyError, OSError):
        return None


def aggregate_file(raw_path, year, month, config, zone_ids):
    low = pd.Timestamp(year, month, 1)
    high = low + pd.offsets.MonthBegin(1)
    counts = None
    raw_hours = set()
    audit = dict(year=year, month=month, raw_rows=0, rejected_date=0,
                 rejected_values=0, rejected_zone=0, kept_rows=0)
    with pq.ParquetFile(raw_path) as parquet:
        if not set(COLUMNS).issubset(parquet.schema_arrow.names):
            raise ValueError('Неполная схема исходного Parquet')
        for batch in parquet.iter_batches(batch_size=config.batch_size, columns=COLUMNS):
            frame = batch.to_pandas()
            audit['raw_rows'] += len(frame)
            time = pd.to_datetime(frame.tpep_pickup_datetime, errors='coerce')
            in_month = time.ge(low) & time.lt(high)
            raw_hours.update(time[in_month].dt.floor('h').unique())
            distance = pd.to_numeric(frame.trip_distance, errors='coerce')
            amount = pd.to_numeric(frame.total_amount, errors='coerce')
            finite = np.isfinite(distance) & np.isfinite(amount)
            valid = finite & distance.gt(0) & distance.le(config.max_distance) & amount.ge(0) & amount.le(config.max_amount)
            zones = frame.PULocationID.isin(zone_ids)
            audit['rejected_date'] += int((~in_month).sum())
            audit['rejected_values'] += int((in_month & ~valid).sum())
            audit['rejected_zone'] += int((in_month & valid & ~zones).sum())
            keep = in_month & valid & zones
            grouped = time[keep].dt.floor('h').value_counts(sort=False)
            counts = grouped if counts is None else counts.add(grouped, fill_value=0)
            audit['kept_rows'] += int(keep.sum())
    if counts is None or audit['kept_rows'] == 0:
        raise ValueError('Месяц не содержит допустимых поездок')
    hours = pd.date_range(low, high, freq='h', inclusive='left')
    dst = hours.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').isna()
    missing = hours.difference(pd.DatetimeIndex(list(raw_hours)))
    unexpected = missing.difference(hours[dst])
    if len(unexpected):
        raise ValueError(f'В исходнике нет ни одной записи за часы: {unexpected[:5].tolist()}')
    result = counts.rename_axis('pickup_hour').rename('trips_count').reset_index()
    result['trips_count'] = result.trips_count.astype('int64')
    result = result.sort_values('pickup_hour').reset_index(drop=True)
    assert result.trips_count.sum() == audit['kept_rows']
    return result, audit


def prepare(root, config):
    root = Path(root)
    raw, cache, artifacts = [root / name for name in ['raw', 'cache', 'artifacts']]
    for path in (raw, cache, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    lookup_path = download(f'{BASE}/misc/taxi_zone_lookup.csv', cache / 'taxi_zone_lookup.csv')
    lookup = pd.read_csv(lookup_path)
    zone_ids = set(lookup.loc[lookup.Borough.isin(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']), 'LocationID'])
    signature = clean_signature(config, zone_ids, digest(lookup_path))
    months = [(config.year, month) for month in range(1, 13)]
    if config.include_new_test:
        months.append((config.year + 1, 1))
    frames, audits = [], []
    for year, month in months:
        output = cache / f'city_{year}_{month:02d}_{signature[:12]}.parquet'
        cached = cache_read(output, signature)
        if cached is None:
            url = f'{BASE}/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
            path = download(url, raw / url.rsplit('/', 1)[-1])
            frame, audit = aggregate_file(path, year, month, config, zone_ids)
            audit.update(signature=signature, source_url=url, source_sha256=digest(path))
            temporary = output.with_suffix('.tmp.parquet')
            frame.to_parquet(temporary, index=False)
            temporary.replace(output)
            audit['aggregate_sha256'] = digest(output)
            meta = output.with_suffix('.json')
            tmp_meta = meta.with_suffix('.tmp')
            tmp_meta.write_text(json.dumps(audit, indent=2), encoding='utf-8')
            tmp_meta.replace(meta)
            path.unlink()
        else:
            frame, audit = cached
        frames.append(frame)
        audits.append(audit)
        print(f'{year}-{month:02d}: {audit["kept_rows"]:,} поездок', flush=True)
    sparse = pd.concat(frames, ignore_index=True)
    end = pd.Timestamp(config.year + 1, 2 if config.include_new_test else 1, 1)
    grid = pd.date_range(pd.Timestamp(config.year, 1, 1), end, freq='h', inclusive='left')
    panel = sparse.set_index('pickup_hour').reindex(grid, fill_value=0).rename_axis('pickup_hour').reset_index()
    assert panel.trips_count.sum() == sparse.trips_count.sum()
    bad = grid.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').isna()
    panel.loc[bad, 'trips_count'] = np.nan
    panel.to_parquet(artifacts / 'panel.parquet', index=False)
    pd.DataFrame(audits).to_csv(artifacts / 'data_audit.csv', index=False)
    return panel
