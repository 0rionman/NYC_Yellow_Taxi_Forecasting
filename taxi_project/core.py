"""Shared data contract, causal features, model artifact and monitoring."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Config:
    year: int = 2024
    availability_delay_hours: int = 1
    max_distance: float = 100.0
    max_amount: float = 500.0
    batch_size: int = 200000
    seed: int = 42
    trials: int = 8
    n_estimators: int = 1500
    alpha: float = .1
    include_new_test: bool = True
    business_use: str = 'Не согласовано'
    business_max_mae: float | None = None
    cost_underestimation: float = 3.0
    cost_overestimation: float = 1.0
    selection_metric: str = 'MAE'

    def __post_init__(self):
        if not 1 <= self.availability_delay_hours <= 24:
            raise ValueError('Задержка доступности должна быть от 1 до 24 часов')
        if not 0 < self.alpha < 1:
            raise ValueError('alpha должна лежать между 0 и 1')
        if self.trials < 1 or self.n_estimators < 1:
            raise ValueError('Число испытаний и деревьев должно быть положительным')
        if min(self.cost_underestimation, self.cost_overestimation) <= 0:
            raise ValueError('Сценарные штрафы должны быть положительными')
        if self.selection_metric not in ['MAE', 'BusinessLossPerHour']:
            raise ValueError('Неизвестная метрика выбора')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_signature(config, zone_ids, lookup_sha):
    payload = dict(schema=SCHEMA_VERSION, rule='city-month-finite-v2',
                   max_distance=config.max_distance, max_amount=config.max_amount,
                   zone_ids=sorted(int(z) for z in zone_ids), lookup_sha=lookup_sha)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_panel(frame, allow_nan=False):
    if set(frame.columns) != {'pickup_hour', 'trips_count'}:
        raise ValueError('Ожидаются только pickup_hour и trips_count')
    out = frame.copy()
    out['pickup_hour'] = pd.to_datetime(out.pickup_hour, errors='raise')
    if out.empty or out.pickup_hour.isna().any():
        raise ValueError('Пустые данные или даты')
    if out.pickup_hour.dt.tz is not None:
        raise ValueError('Нужны местные часы Нью-Йорка без timezone')
    if not out.pickup_hour.eq(out.pickup_hour.dt.floor('h')).all():
        raise ValueError('Время должно совпадать с началом часа')
    if out.pickup_hour.duplicated().any():
        raise ValueError('Дублирующиеся часы')
    out['trips_count'] = pd.to_numeric(out.trips_count, errors='raise')
    known = out.trips_count.dropna()
    if not np.isfinite(known).all() or (known < 0).any():
        raise ValueError('Число поездок должно быть конечным и неотрицательным')
    if not allow_nan and out.trips_count.isna().any():
        raise ValueError('Пропуски в числе поездок')
    out = out.sort_values('pickup_hour').reset_index(drop=True)
    expected = pd.date_range(out.pickup_hour.min(), out.pickup_hour.max(), freq='h')
    if len(out) != len(expected):
        raise ValueError('Пропущены часы в сетке')
    return out


def features(frame, config):
    out = validate_panel(frame, allow_nan=True)
    dt = out.pickup_hour.dt
    out['hour'] = dt.hour
    out['weekday'] = dt.dayofweek
    out['is_weekend'] = (dt.dayofweek >= 5).astype(int)
    # Federal observed holidays, defined explicitly; no external calendar dependency.
    dates = USFederalHolidayCalendar().holidays(out.pickup_hour.min().normalize(),
                                               out.pickup_hour.max().normalize())
    out['is_holiday'] = out.pickup_hour.dt.normalize().isin(dates).astype(int)
    for name, period in [('hour', 24), ('weekday', 7)]:
        out[name + '_sin'] = np.sin(2 * np.pi * out[name] / period)
        out[name + '_cos'] = np.cos(2 * np.pi * out[name] / period)
    delay = config.availability_delay_hours
    for lag in sorted({delay, delay + 1, delay + 2, 24, 168}):
        out[f'lag_{lag}'] = out.trips_count.shift(lag)
    for window in (3, 24, 168):
        prior = out.trips_count.shift(delay).rolling(window, min_periods=window)
        out[f'roll_mean_{window}'] = prior.mean()
        out[f'roll_std_{window}'] = prior.std(ddof=0)
    return out


def feature_sets(config):
    calendar = ['hour', 'weekday', 'is_weekend', 'is_holiday',
                'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
    lags = [f'lag_{v}' for v in sorted({config.availability_delay_hours,
            config.availability_delay_hours + 1, config.availability_delay_hours + 2, 24, 168})]
    rolling = [f'roll_{stat}_{w}' for w in (3, 24, 168) for stat in ('mean', 'std')]
    return {'calendar': calendar, 'lags': calendar + lags, 'full': calendar + lags + rolling}


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.shape != prediction.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Некорректные массивы метрик')
    return dict(MAE=float(mean_absolute_error(y, prediction)),
                RMSE=float(np.sqrt(mean_squared_error(y, prediction))),
                WAPE_pct=float(100 * np.abs(y - prediction).sum() / y.sum()) if y.sum() else float('nan'))


def business_metrics(y, prediction, config):
    error = np.asarray(y, float) - np.asarray(prediction, float)
    cost = config.cost_underestimation * np.maximum(0, error) + config.cost_overestimation * np.maximum(0, -error)
    return {'BusinessLoss': float(cost.sum()), 'BusinessLossPerHour': float(cost.mean())}


def radius(y, prediction, method, alpha=.1):
    error = np.abs(np.asarray(y) - np.asarray(prediction))
    scale = np.sqrt(np.maximum(1, prediction)) if method == 'scaled' else np.ones(len(error))
    scores = np.sort(error / scale)
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if len(scores) < 30 or rank > len(scores):
        raise ValueError('Недостаточно данных для калибровки')
    return float(scores[rank - 1])


def bounds(prediction, q, method):
    prediction = np.asarray(prediction)
    scale = np.sqrt(np.maximum(1, prediction)) if method == 'scaled' else np.ones(len(prediction))
    return np.maximum(0, prediction - q * scale), prediction + q * scale


@dataclass
class ForecastModel:
    pipeline: object
    name: str
    columns: list
    config: Config
    interval_method: str
    interval_q: float
    fitted_until: str
    reference: dict
    schema_version: int = SCHEMA_VERSION

    def predict_features(self, table):
        if self.name == 'WeeklyNaive':
            return table.lag_168.to_numpy()
        return np.maximum(0, self.pipeline.predict(table[self.columns]))

    def forecast(self, history, target_time, live=False, now=None):
        h = validate_panel(history)
        target = pd.Timestamp(target_time)
        if target.tzinfo is not None or pd.isna(target) or target != target.floor('h'):
            raise ValueError('Момент прогноза должен быть началом локального часа')
        last = target - pd.Timedelta(hours=self.config.availability_delay_hours)
        if h.pickup_hour.max() != last:
            raise ValueError('История устарела или содержит ещё недоступные часы')
        if live:
            current = pd.Timestamp.now(tz='America/New_York') if now is None else pd.Timestamp(now)
            if current.tzinfo is None:
                raise ValueError('Для live-проверки now должен содержать timezone')
            current = current.tz_convert('America/New_York').tz_localize(None).floor('h')
            if target != current:
                raise ValueError('Прогноз не соответствует текущему часу Нью-Йорка')
        grid = pd.date_range(last - pd.Timedelta(hours=167), target, freq='h')
        if grid.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').isna().any():
            raise ValueError('История затрагивает неоднозначные часы DST')
        tail = h[h.pickup_hour >= grid.min()]
        if len(tail) != 168:
            raise ValueError('Нужны минимум 168 полных часов истории')
        future = pd.DataFrame({'pickup_hour': pd.date_range(last + pd.Timedelta(hours=1), target, freq='h'),
                               'trips_count': np.nan})
        row = features(pd.concat([tail, future], ignore_index=True), self.config).tail(1)
        if row[self.columns].isna().any().any() or pd.isna(row.lag_168.iloc[0]):
            raise ValueError('Недостаточно истории для признаков')
        pred = self.predict_features(row)
        lo, hi = bounds(pred, self.interval_q, self.interval_method)
        return pd.DataFrame({'pickup_hour': [target], 'prediction': pred, 'lower': lo, 'upper': hi})

    def save(self, path):
        joblib.dump(self, path)


def load_model(path):
    # Load only trusted local artifacts, never arbitrary uploaded pickle/joblib files.
    obj = joblib.load(path)
    if not isinstance(obj, ForecastModel) or obj.schema_version != SCHEMA_VERSION:
        raise ValueError('Несовместимая версия артефакта')
    return obj


def reference_distribution(values):
    values = np.asarray(values, float)
    cuts = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    edges = np.r_[-np.inf, cuts[1:-1], np.inf]
    counts = np.histogram(values, bins=edges)[0] + .5
    return {'edges': edges.tolist(), 'probabilities': (counts / counts.sum()).tolist()}


def monitor(values, reference, actual=None, prediction=None, coverage=None, max_mae=None):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Мониторинг требует конечные наблюдения')
    counts = np.histogram(values, bins=reference['edges'])[0] + .5
    observed = counts / counts.sum()
    expected = np.asarray(reference['probabilities'])
    psi = float(np.sum((observed - expected) * np.log(observed / expected)))
    alerts = []
    if psi > .2:
        alerts.append('PSI > 0.2: проверить изменение распределения (технический ориентир, не бизнес-SLA)')
    output = {'psi': psi, 'n': len(values), 'alerts': alerts}
    if actual is not None:
        output.update(metrics(actual, prediction))
        if max_mae is not None and output['MAE'] > max_mae:
            alerts.append('MAE выше согласованного бизнес-порога')
    if coverage is not None:
        output['coverage'] = float(coverage)
        if coverage < .85:
            alerts.append('Покрытие <85% при номинале 90%: проверить интервалы; порог исследовательский')
    return output
