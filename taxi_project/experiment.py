"""Calendar CV, Optuna selection, ablation, interval comparison and final evaluation."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, process_time
import json
import pickle
import threading
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
import psutil
import optuna
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from .core import (ForecastModel, features, feature_sets, metrics, business_metrics,
                   radius, bounds, reference_distribution, monitor, digest)


def scores(y, pred, config):
    return {**metrics(y, pred), **business_metrics(y, pred, config)}


def calendar_folds(data, year):
    for month in [8, 9, 10]:
        start = pd.Timestamp(year, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        train = data[data.pickup_hour < start]
        valid = data[(data.pickup_hour >= start) & (data.pickup_hour < end)]
        if train.empty or valid.empty:
            raise ValueError('Не хватает данных для календарной CV')
        assert train.pickup_hour.max() < valid.pickup_hour.min()
        yield f'{year}-{month:02d}', start, train, valid


def make_pipeline(family, params, columns, config):
    categorical = [c for c in ['hour', 'weekday'] if c in columns]
    numeric = [c for c in columns if c not in categorical]
    if family == 'Ridge':
        prep = ColumnTransformer([('calendar', OneHotEncoder(handle_unknown='ignore'), categorical),
                                  ('numeric', StandardScaler(), numeric)])
        estimator = Ridge(alpha=params['alpha'], solver='lsqr')
    else:
        prep = ColumnTransformer([('numeric', 'passthrough', columns)], remainder='drop')
        prep.set_output(transform='pandas')
        kwargs = dict(objective='regression_l1', learning_rate=.05, n_estimators=config.n_estimators,
                      n_jobs=2, random_state=config.seed, deterministic=True, force_col_wise=True,
                      verbosity=-1)
        if config.selection_metric == 'BusinessLossPerHour':
            kwargs.update(objective='quantile', alpha=config.cost_underestimation /
                          (config.cost_underestimation + config.cost_overestimation))
        kwargs.update(params)
        estimator = lgb.LGBMRegressor(**kwargs)
    return Pipeline([('preprocess', prep), ('model', estimator)])


def measured_fit(pipeline, x, y, **kwargs):
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.02):
            samples.append(process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start, cpu = perf_counter(), process_time()
    try:
        pipeline.fit(x, y, **kwargs)
    finally:
        stop.set()
        thread.join()
    elapsed = perf_counter() - start
    return dict(fit_seconds=elapsed, cpu_seconds=process_time() - cpu,
                process_peak_rss_mb=max(samples + [process.memory_info().rss]) / 2**20,
                additional_rss_mb=max(0, max(samples) - baseline) / 2**20)


def fit_fold(family, params, columns, train, boundary, config):
    # Dedicated internal early-stop period AND separate calibration period.
    calibration_start = boundary - pd.Timedelta(days=14)
    stop_start = boundary - pd.Timedelta(days=28)
    embargo = pd.Timedelta(hours=config.availability_delay_hours - 1)
    fit = train[train.pickup_hour < calibration_start - embargo]
    calibration = train[train.pickup_hour >= calibration_start]
    if min(len(fit), len(calibration)) < 30:
        raise ValueError('Недостаточно данных внутренних окон')
    if family == 'WeeklyNaive':
        return None, fit, calibration, 0, dict(fit_seconds=0, cpu_seconds=0, process_peak_rss_mb=0, additional_rss_mb=0)
    count = 0
    if family == 'LightGBM':
        core = train[train.pickup_hour < stop_start - embargo]
        early = train[(train.pickup_hour >= stop_start) & (train.pickup_hour < calibration_start)]
        initial = make_pipeline(family, params, columns, config)
        initial.named_steps['preprocess'].fit(core[columns])
        x_early = initial.named_steps['preprocess'].transform(early[columns])
        initial.fit(core[columns], core.trips_count,
                    model__eval_set=[(x_early, early.trips_count)],
                    model__callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
        count = max(1, initial.named_steps['model'].best_iteration_)
        params = {**params, 'n_estimators': count}
    pipeline = make_pipeline(family, params, columns, config)
    timing = measured_fit(pipeline, fit[columns], fit.trips_count)
    return pipeline, fit, calibration, count, timing


def pred(pipeline, family, frame, columns):
    if family == 'WeeklyNaive':
        return frame.lag_168.to_numpy()
    return np.maximum(0, pipeline.predict(frame[columns]))


def evaluate_candidate(data, family, params, columns, config, label, capture=False):
    rows, interval_rows, held_predictions = [], [], []
    for fold, boundary, train, valid in calendar_folds(data, config.year):
        start = perf_counter()
        pipeline, fit, calibration, count, timing = fit_fold(family, params, columns, train, boundary, config)
        train_pred = pred(pipeline, family, fit, columns)
        cal_pred = pred(pipeline, family, calibration, columns)
        elapsed = []
        for _ in range(5):
            started = perf_counter()
            valid_pred = pred(pipeline, family, valid, columns)
            elapsed.append(perf_counter() - started)
        row = dict(candidate=label, family=family, fold=fold, n_features=len(columns),
                   n_train=len(fit), n_validation=len(valid), best_iteration=count,
                   train_MAE=metrics(fit.trips_count, train_pred)['MAE'],
                   inference_median_s=float(np.median(elapsed)), inference_p95_s=float(np.quantile(elapsed,.95)),
                   model_bytes=len(pickle.dumps(pipeline)) if pipeline is not None else 0,
                   total_fold_seconds=perf_counter()-start, **timing, **scores(valid.trips_count, valid_pred, config))
        rows.append(row)
        if capture:
            output = valid[['pickup_hour','trips_count']].copy()
            output['prediction'] = valid_pred
            output['fold'] = fold
            held_predictions.append(output)
            for method in ['absolute', 'scaled']:
                q = radius(calibration.trips_count, cal_pred, method, config.alpha)
                lower, upper = bounds(valid_pred, q, method)
                interval_rows.append(dict(candidate=label, fold=fold, method=method,
                    coverage=float(((valid.trips_count >= lower) & (valid.trips_count <= upper)).mean()),
                    mean_width=float(np.mean(upper-lower)), n=len(valid)))
    return pd.DataFrame(rows), pd.DataFrame(interval_rows), held_predictions


def summary(rows):
    return rows.groupby(['candidate','family'], as_index=False).agg(
        MAE_mean=('MAE','mean'), MAE_std=('MAE','std'), RMSE_mean=('RMSE','mean'),
        WAPE_mean=('WAPE_pct','mean'), BusinessLossPerHour_mean=('BusinessLossPerHour','mean'),
        BusinessLossPerHour_std=('BusinessLossPerHour','std'), train_MAE_mean=('train_MAE','mean'),
        fit_seconds_mean=('fit_seconds','mean'), inference_median_s=('inference_median_s','median'),
        peak_rss_mb=('process_peak_rss_mb','max'), model_bytes=('model_bytes','max'),
        n_features=('n_features','first'))


def run_experiment(panel, config, artifacts):
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    sets = feature_sets(config)
    table = features(panel, config).dropna().reset_index(drop=True)
    tune_data = table[table.pickup_hour < pd.Timestamp(config.year,11,1)]
    configs = {'WeeklyNaive': ('WeeklyNaive', {}, sets['full'])}
    all_rows = []
    first, _, _ = evaluate_candidate(tune_data, 'WeeklyNaive', {}, sets['full'], config, 'WeeklyNaive')
    all_rows.append(first)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in ['Ridge','LightGBM']:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.seed))
        def objective(trial):
            if family == 'Ridge':
                params = {'alpha':trial.suggest_float('alpha', .1, 1000, log=True)}
            else:
                params = dict(num_leaves=trial.suggest_int('num_leaves',15,63),
                    min_child_samples=trial.suggest_int('min_child_samples',30,200),
                    reg_lambda=trial.suggest_float('reg_lambda',.1,10,log=True))
            rows, _, _ = evaluate_candidate(tune_data,family,params,sets['full'],config,f'{family}_trial{trial.number}')
            for column in ['MAE','BusinessLossPerHour']:
                trial.set_user_attr(column+'_std', float(rows[column].std()))
            return float(rows[config.selection_metric].mean())
        study.optimize(objective, n_trials=config.trials)
        study.trials_dataframe().to_csv(artifacts/f'optuna_{family}.csv',index=False)
        best = study.best_params
        for subset, columns in sets.items():
            label = f'{family}_{subset}'
            configs[label] = (family,best,columns)
            rows, _, _ = evaluate_candidate(tune_data,family,best,columns,config,label)
            all_rows.append(rows)
            print(f'CV {label}: {config.selection_metric}={rows[config.selection_metric].mean():.3f}', flush=True)
    cv = pd.concat(all_rows,ignore_index=True)
    cv.to_csv(artifacts/'cv_folds.csv',index=False)
    leaderboard = summary(cv)
    metric = config.selection_metric+'_mean'
    best_score = leaderboard[metric].min()
    # Predeclared simplicity rule: within 1% of best, prefer fewer features, lower variability, then latency.
    eligible = leaderboard[leaderboard[metric] <= best_score*1.01 + 1e-12]
    chosen = eligible.sort_values(['n_features',config.selection_metric+'_std','inference_median_s','candidate']).iloc[0].candidate
    leaderboard['selected'] = leaderboard.candidate.eq(chosen)
    leaderboard.to_csv(artifacts/'cv_summary.csv',index=False)
    family, params, columns = configs[chosen]
    _, intervals, oof = evaluate_candidate(tune_data,family,params,columns,config,chosen,capture=True)
    intervals.to_csv(artifacts/'interval_cv.csv',index=False)
    interval_summary = intervals.groupby('method',as_index=False).agg(coverage=('coverage','mean'),mean_width=('mean_width','mean'))
    interval_summary['coverage_gap'] = abs(interval_summary.coverage-(1-config.alpha))
    method = interval_summary.sort_values(['coverage_gap','mean_width','method']).iloc[0].method
    interval_summary.to_csv(artifacts/'interval_cv_summary.csv',index=False)
    (artifacts/'selection.json').write_text(json.dumps({'candidate':chosen,'family':family,
        'params':params,'features':columns,'interval_method':method,'selection_metric':config.selection_metric,
        'note':'Fixed using August–October CV before evaluating December/January'},indent=2),encoding='utf-8')
    pd.concat(oof).to_csv(artifacts/'cv_predictions.csv',index=False)
    fitted_until = pd.Timestamp(config.year,11,1)
    fit = table[table.pickup_hour < fitted_until-pd.Timedelta(hours=config.availability_delay_hours-1)]
    calibration = table[(table.pickup_hour >= fitted_until) & (table.pickup_hour < pd.Timestamp(config.year,12,1))]
    selected_by_family = {'WeeklyNaive':'WeeklyNaive'}
    for f in ['Ridge','LightGBM']:
        subset = leaderboard[leaderboard.family.eq(f)].sort_values(metric)
        selected_by_family[f] = chosen if family == f else subset.iloc[0].candidate
    models, test_rows, predictions = {}, [], []
    for f, label in selected_by_family.items():
        _, parameters, cols = configs[label]
        if f == 'WeeklyNaive':
            pipeline = None
            timing = dict(fit_seconds=0,cpu_seconds=0,process_peak_rss_mb=0,additional_rss_mb=0)
        else:
            if f == 'LightGBM':
                rounds = int(np.median(cv[cv.candidate.eq(label)].best_iteration))
                parameters = {**parameters,'n_estimators':max(1,rounds)}
            pipeline = make_pipeline(f,parameters,cols,config)
            timing = measured_fit(pipeline,fit[cols],fit.trips_count)
        cal_pred = pred(pipeline,f,calibration,cols)
        q = radius(calibration.trips_count,cal_pred,method,config.alpha)
        model = ForecastModel(pipeline,f,cols,config,method,q,str(fitted_until),reference_distribution(fit.trips_count))
        models[label] = model
        for period, begin, end in [('December_seen',pd.Timestamp(config.year,12,1),pd.Timestamp(config.year+1,1,1)),
                                    ('January_new',pd.Timestamp(config.year+1,1,1),pd.Timestamp(config.year+1,2,1))]:
            subset = table[(table.pickup_hour>=begin)&(table.pickup_hour<end)]
            if subset.empty:
                continue
            timings=[]
            for _ in range(5):
                start=perf_counter(); p=model.predict_features(subset); timings.append(perf_counter()-start)
            lower, upper = bounds(p,q,method)
            coverage=float(subset.trips_count.between(lower,upper).mean())
            test_rows.append(dict(period=period,candidate=label,selected=label==chosen,n=len(subset),
                coverage=coverage,mean_width=float(np.mean(upper-lower)),inference_median_s=float(np.median(timings)),
                inference_p95_s=float(np.quantile(timings,.95)),model_bytes=len(pickle.dumps(model)),
                **timing,**scores(subset.trips_count,p,config)))
            if label==chosen:
                output=subset[['pickup_hour','trips_count','hour','weekday','is_holiday']].copy()
                output['prediction'],output['lower'],output['upper']=p,lower,upper
                output['period']=period
                predictions.append(output)
    champion=models[chosen]
    champion.save(artifacts/'model.joblib')
    results=pd.DataFrame(test_rows)
    results.to_csv(artifacts/'evaluation.csv',index=False)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions['abs_error']=abs(predictions.trips_count-predictions.prediction)
    predictions['covered']=predictions.trips_count.between(predictions.lower,predictions.upper)
    predictions.to_csv(artifacts/'predictions.csv',index=False)
    for dimension in ['hour','weekday','is_holiday']:
        predictions.groupby(['period',dimension]).agg(MAE=('abs_error','mean'),coverage=('covered','mean'),n=('covered','size')).to_csv(artifacts/f'errors_{dimension}.csv')
    # Importance belongs to development data; do not use January for feature selection.
    explain_models(models,selected_by_family,calibration,artifacts,config)
    monitors=[]
    for period, group in predictions.groupby('period'):
        monitors.append({'period':period,**monitor(group.trips_count,champion.reference,
            group.trips_count,group.prediction,group.covered.mean(),config.business_max_mae)})
    (artifacts/'monitoring.json').write_text(json.dumps(monitors,ensure_ascii=False,indent=2),encoding='utf-8')
    (artifacts/'config.json').write_text(json.dumps(asdict(config),ensure_ascii=False,indent=2),encoding='utf-8')
    versions={p:version(p) for p in ['numpy','pandas','pyarrow','scikit-learn','lightgbm','optuna','streamlit']}
    versions.update(python=platform.python_version(),cpu_logical=psutil.cpu_count(),model_threads=2,
                    gpu_used=False,model_sha256=digest(artifacts/'model.joblib'))
    (artifacts/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    return table,leaderboard,results,champion


def explain_models(models, selected_by_family, calibration, artifacts, config):
    tree=models[selected_by_family['LightGBM']]
    booster=tree.pipeline.named_steps['model'].booster_
    pd.Series(booster.feature_importance(importance_type='gain'),index=tree.columns,name='gain').to_csv(artifacts/'importance_gain.csv')
    # Second implementation: sklearn permutation importance, diagnostic only.
    subset=calibration.iloc[::max(1,len(calibration)//400)]
    result=permutation_importance(tree.pipeline,subset[tree.columns],subset.trips_count,
                                  scoring='neg_mean_absolute_error',n_repeats=5,random_state=config.seed,n_jobs=1)
    pd.DataFrame({'feature':tree.columns,'mae_increase':result.importances_mean,
                  'std':result.importances_std}).to_csv(artifacts/'importance_permutation.csv',index=False)
    # Blocked joint permutation preserves correlation within lag/rolling groups (diagnostic caveats remain).
    rows=[]
    base=metrics(subset.trips_count,tree.predict_features(subset))['MAE']
    rng=np.random.default_rng(config.seed)
    for group,prefixes in [('lags',('lag_',)),('rolling',('roll_',)),('calendar',('hour','weekday','is_'))]:
        cols=[c for c in tree.columns if c.startswith(prefixes)]
        if not cols:
            continue
        deltas=[]
        for _ in range(5):
            shifted=subset.copy()
            order=np.roll(np.arange(len(subset)),int(rng.integers(24,max(25,len(subset)-24))))
            shifted.loc[:,cols]=subset[cols].to_numpy()[order]
            deltas.append(metrics(subset.trips_count,tree.predict_features(shifted))['MAE']-base)
        rows.append({'group':group,'mae_increase':np.mean(deltas),'std':np.std(deltas)})
    pd.DataFrame(rows).to_csv(artifacts/'importance_groups.csv',index=False)
    ridge=models[selected_by_family['Ridge']]
    names=ridge.pipeline.named_steps['preprocess'].get_feature_names_out()
    pd.DataFrame({'feature':names,'coefficient':ridge.pipeline.named_steps['model'].coef_}).to_csv(
        artifacts/'ridge_coefficients.csv',index=False)
    ridge_perm=permutation_importance(ridge.pipeline,subset[ridge.columns],subset.trips_count,
        scoring='neg_mean_absolute_error',n_repeats=5,random_state=config.seed,n_jobs=1)
    pd.DataFrame({'feature':ridge.columns,'mae_increase':ridge_perm.importances_mean,
                  'std':ridge_perm.importances_std}).to_csv(artifacts/'ridge_permutation.csv',index=False)
