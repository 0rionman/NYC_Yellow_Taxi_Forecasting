"""All fixtures are artificial and isolated; they are never published as model quality."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from taxi_project.core import (Config, features, validate_panel, metrics, business_metrics,
    radius, bounds, clean_signature, digest, ForecastModel, load_model,
    reference_distribution, monitor)
from taxi_project.data import aggregate_file, cache_read
from taxi_project.experiment import calendar_folds, run_experiment, make_pipeline
from taxi_project.reporting import eda, final_report


def sample_history(end='2024-02-01 23:00', periods=300):
    return pd.DataFrame({'pickup_hour':pd.date_range(end=end,periods=periods,freq='h'),
                         'trips_count':np.arange(periods,dtype=float)+100})


def simple_model(config=None):
    cfg=config or Config()
    return ForecastModel(None,'WeeklyNaive',['lag_168'],cfg,'absolute',10.,'2024-01-01',
                         reference_distribution(np.arange(100)))


def test_causal_features_and_exact_lags():
    history=sample_history()
    cfg=Config()
    original=features(history,cfg)
    changed=history.copy()
    changed.loc[200:,'trips_count']+=10000
    mutated=features(changed,cfg)
    pd.testing.assert_series_equal(original.iloc[200].drop('trips_count'),mutated.iloc[200].drop('trips_count'))
    assert original.iloc[200].lag_168==history.iloc[32].trips_count
    assert original.iloc[200].roll_mean_24==history.iloc[176:200].trips_count.mean()


@pytest.mark.parametrize('delay',[1,2,24])
def test_delay_in_training_and_inference(delay):
    history=sample_history()
    cfg=Config(availability_delay_hours=delay)
    table=features(history,cfg)
    assert table.iloc[200][f'lag_{delay}']==history.iloc[200-delay].trips_count
    model=simple_model(cfg)
    target=history.pickup_hour.max()+pd.Timedelta(hours=delay)
    result=model.forecast(history,target)
    expected=history.set_index('pickup_hour').loc[target-pd.Timedelta(hours=168),'trips_count']
    assert result.prediction.iloc[0]==expected


@pytest.mark.parametrize('fault',['gap','duplicate','negative','nan','infinity','offgrid','extra_column'])
def test_invalid_history_rejected(fault):
    frame=sample_history()
    if fault=='gap': frame=frame.drop(100)
    if fault=='duplicate': frame=pd.concat([frame,frame.iloc[[0]]])
    if fault=='negative': frame.loc[10,'trips_count']=-1
    if fault=='nan': frame.loc[10,'trips_count']=np.nan
    if fault=='infinity': frame.loc[10,'trips_count']=np.inf
    if fault=='offgrid': frame.loc[10,'pickup_hour']+=pd.Timedelta(minutes=1)
    if fault=='extra_column': frame['surprise']=0
    with pytest.raises(ValueError): validate_panel(frame)


def test_stale_and_future_data_rejected():
    frame=sample_history()
    model=simple_model()
    with pytest.raises(ValueError,match='устарела'):
        model.forecast(frame,frame.pickup_hour.max()+pd.Timedelta(hours=2))
    with pytest.raises(ValueError,match='устарела'):
        model.forecast(frame,frame.pickup_hour.max())
    with pytest.raises(ValueError,match='текущему'):
        model.forecast(frame,frame.pickup_hour.max()+pd.Timedelta(hours=1),live=True,
                       now=pd.Timestamp('2026-01-01',tz='America/New_York'))


def test_dst_rejected():
    frame=sample_history(end='2024-03-11 23:00')
    with pytest.raises(ValueError,match='DST'):
        simple_model().forecast(frame,'2024-03-12')


def test_business_loss():
    result=business_metrics([10,10],[8,13],Config())
    assert result['BusinessLoss']==9
    assert result['BusinessLossPerHour']==4.5
    assert np.isnan(metrics([0,0],[1,2])['WAPE_pct'])


def test_asymmetric_objective():
    cfg=Config(selection_metric='BusinessLossPerHour',n_estimators=5)
    pipeline=make_pipeline('LightGBM',{'num_leaves':7},['lag_1'],cfg)
    estimator=pipeline.named_steps['model']
    assert estimator.get_params()['objective']=='quantile'
    assert estimator.get_params()['alpha']==.75
    pipeline.fit(pd.DataFrame({'lag_1':np.arange(100)}),np.arange(100))
    assert np.isfinite(pipeline.predict(pd.DataFrame({'lag_1':[5,10]}))).all()


def test_intervals():
    y=np.arange(100)+20
    p=y-2
    q=radius(y,p,'absolute')
    lower,upper=bounds(p,q,'absolute')
    assert q==2 and np.all(upper==y) and np.all(lower>=0)
    with pytest.raises(ValueError): radius([1],[1],'absolute')


def test_roundtrip(tmp_path):
    model=simple_model()
    frame=sample_history()
    target=frame.pickup_hour.max()+pd.Timedelta(hours=1)
    path=tmp_path/'model.joblib'
    model.save(path)
    restored=load_model(path)
    pd.testing.assert_frame_equal(model.forecast(frame,target),restored.forecast(frame,target))
    restored.schema_version=1
    restored.save(path)
    with pytest.raises(ValueError): load_model(path)


def test_cleaning_and_cache(tmp_path):
    cfg=Config(batch_size=101)
    hours=pd.date_range('2024-01-01','2024-02-01',freq='h',inclusive='left')
    raw=pd.DataFrame({'tpep_pickup_datetime':hours,'PULocationID':1,'trip_distance':2.,'total_amount':10.})
    extra=raw.iloc[:5].copy()
    extra.loc[0,'trip_distance']=0
    extra.loc[1,'total_amount']=-1
    extra.loc[2,'PULocationID']=999
    extra.loc[3,'tpep_pickup_datetime']=pd.Timestamp('2008-01-01')
    extra.loc[4,'trip_distance']=150
    path=tmp_path/'raw.parquet'
    pd.concat([raw,extra]).to_parquet(path,index=False)
    table,audit=aggregate_file(path,2024,1,cfg,{1})
    assert audit['kept_rows']==744
    assert audit['rejected_values']==3 and audit['rejected_zone']==1 and audit['rejected_date']==1
    signature=clean_signature(cfg,{1},'lookup')
    aggregate=tmp_path/'cache.parquet'
    table.to_parquet(aggregate,index=False)
    audit.update(signature=signature,aggregate_sha256=digest(aggregate))
    meta=aggregate.with_suffix('.json')
    meta.write_text(json.dumps(audit),encoding='utf-8')
    assert cache_read(aggregate,signature) is not None
    assert cache_read(aggregate,clean_signature(replace(cfg,max_distance=200),{1},'lookup')) is None
    assert cache_read(aggregate,clean_signature(cfg,{1,2},'lookup')) is None
    meta.write_text('broken',encoding='utf-8')
    assert cache_read(aggregate,signature) is None


def test_monitoring_alerts():
    reference=reference_distribution(np.arange(1000))
    output=monitor(np.arange(2000,2100),reference,[10]*100,[0]*100,coverage=.5,max_mae=5)
    assert output['psi']>.2 and len(output['alerts'])==3


@pytest.fixture(scope='session')
def experiment(tmp_path_factory):
    folder=tmp_path_factory.mktemp('synthetic_experiment')
    hours=pd.date_range('2024-01-01','2025-02-01',freq='h',inclusive='left')
    rng=np.random.default_rng(42)
    signal=200+50*np.sin(2*np.pi*hours.hour.to_numpy()/24)
    panel=pd.DataFrame({'pickup_hour':hours,'trips_count':rng.poisson(signal).astype(float)})
    dst=hours.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').isna()
    panel.loc[dst,'trips_count']=np.nan
    panel.to_parquet(folder/'panel.parquet',index=False)
    config=Config(trials=1,n_estimators=12)
    table,leaderboard,results,model=run_experiment(panel,config,folder)
    eda(panel,config,folder)
    final_report(table,leaderboard,results,model,config,folder)
    return folder,table,leaderboard,results,model


def test_full_workflow(experiment):
    folder,table,leaderboard,results,model=experiment
    assert leaderboard.selected.sum()==1
    assert set(results.period)=={'December_seen','January_new'}
    assert results.MAE.notna().all()
    assert (folder/'REPORT.md').exists()
    assert (folder/'importance_permutation.csv').exists()
    cv=pd.read_csv(folder/'cv_folds.csv')
    assert cv.groupby('candidate').fold.nunique().eq(3).all()
    for _,boundary,train,valid in calendar_folds(table,2024):
        assert train.pickup_hour.max()<boundary<=valid.pickup_hour.min()
    history=pd.read_parquet(folder/'panel.parquet').tail(168)
    loaded=load_model(folder/'model.joblib')
    pd.testing.assert_frame_equal(model.forecast(history,'2025-02-01'),loaded.forecast(history,'2025-02-01'))


def test_empty_app(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('TAXI_ARTIFACTS',str(tmp_path))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.info)>0


def test_app_does_not_show_incomplete_run(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    (tmp_path/'run_status.json').write_text(json.dumps({'status':'training'}))
    monkeypatch.setenv('TAXI_ARTIFACTS',str(tmp_path))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception and len(app.warning)>0 and len(app.metric)==0


def test_app_with_results(experiment,monkeypatch):
    from streamlit.testing.v1 import AppTest
    folder,*_=experiment
    monkeypatch.setenv('TAXI_ARTIFACTS',str(folder))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.metric)==4
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.error)==0
