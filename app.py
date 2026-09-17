"""Streamlit interface for verified artifacts, diagnostics and next-hour forecasts."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
from taxi_project.core import load_model, business_metrics, Config

st.set_page_config(page_title='NYC Taxi — спрос на следующий час',page_icon='🚕',layout='wide')
st.title('🚕 Поездки Yellow Taxi по Нью-Йорку')
st.caption('Исследовательская модель • прогноз на один час • пять боро • локальное время Нью-Йорка')
root=Path(os.environ.get('TAXI_ARTIFACTS',str(Path(__file__).parent/'artifacts')))
if (root/'run_status.json').exists():
    try:
        status=json.loads((root/'run_status.json').read_text(encoding='utf-8'))
        if status.get('status')!='complete':
            st.warning('Текущий запуск ещё не завершён. Дождитесь успешного окончания обучения и отчёта.')
            st.stop()
    except (ValueError,OSError):
        st.error('Некорректный статус запуска. Повторите обучение и формирование отчёта.')
        st.stop()
if not (root/'evaluation.csv').exists():
    st.info('Сначала выполните ноутбук Colab и распакуйте его архив результатов. Поместите папку artifacts рядом с app.py.')
    st.markdown('Демонстрационные метрики не подставляются. После обучения здесь появятся сравнение моделей, графики и прогноз.')
    st.stop()

try:
    results=pd.read_csv(root/'evaluation.csv')
    predictions=pd.read_csv(root/'predictions.csv',parse_dates=['pickup_hour'])
    cv=pd.read_csv(root/'cv_summary.csv')
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
except (OSError,ValueError,KeyError) as exc:
    st.error(f'Не удалось прочитать комплект результатов: {exc}')
    st.stop()

period=st.sidebar.selectbox('Период оценки',results.period.unique().tolist())
if period=='December_seen':
    st.sidebar.warning('Декабрь уже изучен в версии 1. Это историческая проверка.')
else:
    st.sidebar.info('Независимость января сохраняется только до изменения модели по его результатам.')
st.sidebar.markdown('### Сценарные штрафы')
under=st.sidebar.number_input('Недооценка одной поездки',min_value=.01,value=float(config['cost_underestimation']))
over=st.sidebar.number_input('Переоценка одной поездки',min_value=.01,value=float(config['cost_overestimation']))
st.sidebar.caption('Значения 3 и 1 — условный сценарий. Это не измеренные финансовые потери. Изменение полей пересчитывает только бизнес-ошибку выбранной модели.')
selected=results[results.period.eq(period)&results.selected.eq(True)].iloc[0]
group=predictions[predictions.period.eq(period)]
scenario=Config(**{**config,'cost_underestimation':under,'cost_overestimation':over})
cost=business_metrics(group.trips_count,group.prediction,scenario)
a,b,c,d=st.columns(4)
a.metric('MAE, поездки/час',f'{selected.MAE:.1f}')
b.metric('WAPE',f'{selected.WAPE_pct:.2f}%')
c.metric('Покрытие интервала',f'{selected.coverage:.1%}')
d.metric('Сценарный штраф / час',f'{cost["BusinessLossPerHour"]:.1f}')
if selected.coverage<.9:
    st.warning('Номинальное покрытие 90% на этом периоде не достигнуто.')
tabs=st.tabs(['Прогнозы и ошибки','Сравнение и признаки','Следующий час','Мониторинг и отчёт'])
with tabs[0]:
    st.subheader('Факт и прогноз')
    days=sorted(group.pickup_hour.dt.date.unique())
    start=st.selectbox('Начало графика',days)
    length=st.slider('Показать дней',1,31,7)
    subset=group[(group.pickup_hour>=pd.Timestamp(start)) &
                 (group.pickup_hour<pd.Timestamp(start)+pd.Timedelta(days=length))]
    chart=subset.set_index('pickup_hour')[['trips_count','prediction','lower','upper']]
    st.line_chart(chart.rename(columns={'trips_count':'Факт','prediction':'Прогноз','lower':'Нижняя граница','upper':'Верхняя граница'}))
    st.caption('Ось X — локальное время; ось Y — поездки за час. Границы — эмпирический прогнозный интервал.')
    st.subheader('Ошибки по часам')
    st.bar_chart(group.groupby('hour').abs_error.mean().rename('MAE, поездки/час'))
    st.dataframe(group.nlargest(15,'abs_error')[['pickup_hour','trips_count','prediction','abs_error']])
    st.download_button('Скачать прогнозы CSV',group.to_csv(index=False).encode('utf-8-sig'),file_name=f'{period}_predictions.csv')
with tabs[1]:
    st.subheader('Три календарных фолда: среднее и разброс')
    st.dataframe(cv)
    st.caption('Эти данные использованы при выборе. Числа нового теста не входят в подбор.')
    st.subheader('Зафиксированные модели: проверка периода')
    st.dataframe(results[results.period.eq(period)])
    st.caption('BusinessLoss в таблице использует штрафы конфигурации обучения; боковая панель меняет только карточку сценария.')
    for name in ['cv_comparison.png','importance.png']:
        if (root/name).exists():
            st.image(str(root/name))
    st.caption('Gain и перестановочная важность не являются причинным влиянием признаков.')
with tabs[2]:
    st.subheader('Прогноз следующего часа')
    st.info('Артефакт загружается только из локальной папки результатов. Не используйте чужие непроверенные model.joblib.')
    st.caption('Очищенный предыдущий час может быть недоступен в реальном времени. Пока задержка не подтверждена источником, это исследовательский сценарий.')
    uploaded=st.file_uploader('История CSV: pickup_hour, trips_count; минимум 168 полных часов',type=['csv'])
    try:
        if uploaded is not None:
            history=pd.read_csv(uploaded,parse_dates=['pickup_hour'])
        elif (root/'panel.parquet').exists():
            history=pd.read_parquet(root/'panel.parquet').tail(168).copy()
        else:
            history=None
        if history is not None:
            default=pd.Timestamp(history.pickup_hour.max())+pd.Timedelta(hours=config['availability_delay_hours'])
            target=st.text_input('Начало целевого часа, местное время',value=str(default))
            live=st.checkbox('Проверить соответствие текущему часу Нью-Йорка',value=False)
            if st.button('Рассчитать прогноз'):
                model=load_model(root/'model.joblib')
                forecast=model.forecast(history,target,live=live)
                st.dataframe(forecast)
                st.download_button('Скачать прогноз',forecast.to_csv(index=False).encode(),file_name='next_hour.csv')
    except (ValueError,KeyError,OSError,TypeError) as exc:
        st.error(f'Прогноз не выполнен: {exc}')
with tabs[3]:
    st.subheader('Диагностика данных и качества')
    if (root/'monitoring.json').exists():
        monitoring=json.loads((root/'monitoring.json').read_text(encoding='utf-8'))
        for item in monitoring:
            if item['period']==period:
                st.json(item)
                for alert in item['alerts']:
                    st.warning(alert)
    st.caption('PSI и пороги покрытия — исследовательская диагностика. Они не заменяют согласованные SLA и регулярный мониторинг сервиса.')
    if (root/'data_audit.csv').exists():
        st.dataframe(pd.read_csv(root/'data_audit.csv'))
    if (root/'REPORT.md').exists():
        report=(root/'REPORT.md').read_text(encoding='utf-8')
        st.download_button('Скачать полный отчёт',report,file_name='REPORT.md')
        st.markdown(report)
