"""Figures and evidence-based Russian reports; no pre-filled experimental results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def eda(panel, config, artifacts):
    artifacts=Path(artifacts)
    train=panel[panel.pickup_hour < pd.Timestamp(config.year,10,1)].copy()
    train['hour']=train.pickup_hour.dt.hour
    train['weekday']=train.pickup_hour.dt.dayofweek
    values=train.trips_count.dropna()
    stats=values.describe(percentiles=[.01,.05,.5,.95,.99])
    stats.to_csv(artifacts/'target_statistics.csv',header=['value'])
    center=values.median()
    mad=(values-center).abs().median()
    outliers=train[abs(train.trips_count-center)>6*max(1,mad)].copy()
    outliers.to_csv(artifacts/'eda_extreme_hours.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    train.set_index('pickup_hour').trips_count.resample('D').sum(min_count=1).plot(ax=axes[0,0])
    axes[0,0].set(title='Train: поездки по дням',xlabel='Дата',ylabel='Поездки за сутки')
    axes[0,1].hist(values,bins=50)
    axes[0,1].set(title='Train: распределение целевой переменной',xlabel='Поездки за час',ylabel='Количество часов')
    train.groupby('hour').trips_count.mean().plot.bar(ax=axes[1,0])
    axes[1,0].set(title='Профиль суток',xlabel='Час Нью-Йорка',ylabel='Среднее поездок/час')
    train.groupby('weekday').trips_count.mean().plot.bar(ax=axes[1,1])
    axes[1,1].set(title='Профиль недели',xlabel='День недели: 0 — понедельник',ylabel='Среднее поездок/час')
    save(fig,artifacts/'eda.png')
    corr={str(lag):float(train.trips_count.autocorr(lag)) for lag in [1,24,168]}
    # Interpolation is only for this spectrum, never for model features/targets.
    spectrum=train.trips_count.interpolate(limit_direction='both')
    freq,power=periodogram(spectrum,detrend='linear')
    period=1/freq[1:]
    keep=(period>=2)&(period<=400)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(period[keep],power[1:][keep])
    ax.axvline(24,color='orange',label='24 часа');ax.axvline(168,color='red',label='168 часов')
    ax.set(title='Train: периодограмма',xlabel='Период, часы',ylabel='Спектральная мощность')
    ax.legend();save(fig,artifacts/'seasonality.png')
    peak=int(train.groupby('hour').trips_count.mean().idxmax())
    trough=int(train.groupby('hour').trips_count.mean().idxmin())
    text=f'''# EDA: выводы по обучающему периоду

Медиана: {center:.2f} поездки/час; 95-й процентиль: {values.quantile(.95):.2f}.
Доля нулей среди известных часов: {values.eq(0).mean():.2%}.
Пропуски: {train.trips_count.isna().sum()} часов; их не заменяем будущими значениями.
Наибольший средний объём в {peak:02d}:00, наименьший — в {trough:02d}:00.
Автокорреляции: 1 час {corr['1']:.3f}; 24 часа {corr['24']:.3f}; 168 часов {corr['168']:.3f}.
Это поддерживает гипотезы суточной и недельной сезонности; полезность признаков оценивается отдельно на CV.

За пределами медианы ±6 MAD найдено {len(outliers)} часов. Это диагностический порог,
не основание удалять высокий реальный спрос. Часы выгружены для проверки причин;
погода и события отсутствуют, поэтому объяснения не приписываем.
Правила очистки исходных поездок проверяются отдельным аудитом; исключённые записи не обязательно ошибочны.
Спектр использует интерполяцию только для визуализации. Годовая сезонность на одном году не доказана.
'''
    (artifacts/'EDA_REPORT.md').write_text(text,encoding='utf-8')


def final_report(table, leaderboard, results, model, config, artifacts):
    artifacts=Path(artifacts)
    if not config.include_new_test:
        for name in ['forecast_January_new.png','errors_January_new.png']:
            (artifacts/name).unlink(missing_ok=True)
    chosen=leaderboard[leaderboard.selected].iloc[0].candidate
    predictions=pd.read_csv(artifacts/'predictions.csv',parse_dates=['pickup_hour'])
    cv=pd.read_csv(artifacts/'cv_folds.csv')
    numeric=table[table.pickup_hour<pd.Timestamp(config.year,10,1)].select_dtypes('number')
    correlation=numeric.corr()
    correlation.to_csv(artifacts/'feature_correlations.csv')
    fig,ax=plt.subplots(figsize=(11,9))
    im=ax.imshow(correlation,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(correlation)),correlation.columns,rotation=90,fontsize=7)
    ax.set_yticks(range(len(correlation)),correlation.columns,fontsize=7)
    ax.set_title('Train: корреляции таргета и признаков');fig.colorbar(im,ax=ax,label='Pearson r')
    save(fig,artifacts/'correlations.png')
    fig,ax=plt.subplots(figsize=(13,5))
    names=leaderboard.candidate.tolist()
    positions=np.arange(len(names))
    ax.bar(positions,leaderboard.MAE_mean,yerr=leaderboard.MAE_std,capsize=4,label='Validation: среднее ± std')
    ax.plot(positions,leaderboard.train_MAE_mean,'o-',color='orange',label='Train MAE')
    ax.set_xticks(positions,names,rotation=25,ha='right')
    ax.set(title='Календарная CV: качество и признаки',xlabel='Модель / набор признаков',ylabel='MAE, поездки/час')
    ax.legend();save(fig,artifacts/'cv_comparison.png')
    for period,group in predictions.groupby('period'):
        zoom=group.head(24*14)
        fig,axes=plt.subplots(2,1,figsize=(14,8))
        axes[0].plot(zoom.pickup_hour,zoom.trips_count,label='Факт')
        axes[0].plot(zoom.pickup_hour,zoom.prediction,label='Прогноз')
        axes[0].fill_between(zoom.pickup_hour,zoom.lower,zoom.upper,alpha=.2,label='Интервал: номинал 90%')
        axes[0].set(title=f'{period}: Нью-Йорк; покрытие месяца {group.covered.mean():.1%}',
                    xlabel='Локальное время Нью-Йорка',ylabel='Поездки за час')
        axes[0].legend()
        axes[1].hist(group.trips_count-group.prediction,bins=50)
        axes[1].set(title='Распределение остатков',xlabel='Факт минус прогноз, поездки',ylabel='Количество часов')
        save(fig,artifacts/f'forecast_{period}.png')
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        group.groupby('hour').abs_error.mean().plot.bar(ax=axes[0])
        axes[0].set(title='Ошибки по часам',xlabel='Час Нью-Йорка',ylabel='MAE, поездки/час')
        group.groupby('weekday').abs_error.mean().plot.bar(ax=axes[1])
        axes[1].set(title='Ошибки по дням недели',xlabel='0 — понедельник',ylabel='MAE, поездки/час')
        save(fig,artifacts/f'errors_{period}.png')
    gain=pd.read_csv(artifacts/'importance_gain.csv',index_col=0).iloc[:,0]
    permutation=pd.read_csv(artifacts/'importance_permutation.csv').set_index('feature')
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    gain.nlargest(12).sort_values().plot.barh(ax=axes[0])
    axes[0].set(title='LightGBM: gain',xlabel='Суммарное уменьшение потерь',ylabel='Признак')
    permutation.mae_increase.nlargest(12).sort_values().plot.barh(ax=axes[1])
    axes[1].set(title='sklearn: перестановочная важность',xlabel='Рост MAE при перестановке',ylabel='Признак')
    save(fig,artifacts/'importance.png')
    paragraphs=[]
    for period,group in results.groupby('period'):
        selected=group[group.selected].iloc[0]
        baseline=group[group.candidate.eq('WeeklyNaive')].iloc[0]
        improvement=100*(1-selected.MAE/baseline.MAE) if baseline.MAE else np.nan
        paragraphs.append(f'''### {period}

MAE выбранной модели {selected.MAE:.2f}; RMSE {selected.RMSE:.2f}; WAPE {selected.WAPE_pct:.2f}%.
Снижение MAE относительно WeeklyNaive {improvement:.2f}%. Отрицательное число означает ухудшение.
Сценарная Business Loss {selected.BusinessLoss:.2f}, на час {selected.BusinessLossPerHour:.2f}.
Покрытие интервала {selected.coverage:.2%}, номинал {1-config.alpha:.0%}; средняя ширина {selected.mean_width:.2f}.
{'Номинальное покрытие на этом периоде не достигнуто.' if selected.coverage < 1-config.alpha else 'Эмпирическое покрытие достигло номинала на этом периоде; это не гарантия будущего покрытия.'}
''')
    best=leaderboard[leaderboard.selected].iloc[0]
    lag_gain=gain.idxmax() if len(gain) else 'нет'
    ablation=[]
    for family in ['Ridge','LightGBM']:
        rows=leaderboard[leaderboard.family.eq(family)].set_index('candidate')
        full=rows.loc[f'{family}_full','MAE_mean']
        for group in ['calendar','lags']:
            value=rows.loc[f'{family}_{group}','MAE_mean']
            ablation.append(f'- {family}: {group} MAE {value:.2f}, полный набор {full:.2f}; '
                            f'разница сокращённый минус полный {value-full:+.2f} поездки/час.')
    ablation_text='\n'.join(ablation)
    report=f'''# NYC Taxi v2 — результаты выполнения

## Задача и бизнес-интерпретация
Суммарное число очищенных поездок Yellow Taxi по пяти боро Нью-Йорка на следующий локальный час.
Назначение: {config.business_use}. Если заказчик ещё не определил действие по прогнозу, готовность бизнес-сценария не подтверждена.
Сценарные штрафы: недооценка {config.cost_underestimation}, переоценка {config.cost_overestimation} за поездку.
Business Loss = сумма положительных недооценок × первый штраф + сумма переоценок × второй штраф.
Конкретная стоимость ошибки зависит от юнит-экономики автопарка; модель демонстрирует возможность
оптимизации под асимметричные штрафы (undersupply penalization).
Если трактовать штрафы как доллары, результат — условный сценарный долларовый ущерб, не измеренные потери.
Нет перевода поездок в число машин без длительности поездок, загрузки и времени доступности автомобилей.

## Протокол
Optuna: {config.trials} испытаний на каждое семейство; внешние месяцы CV — август, сентябрь, октябрь {config.year}.
Внутри каждого фолда последние 14 дней — калибровка; предыдущие 14 — early stopping.
После early stopping модель переобучается на данных до калибровки, внешний месяц не участвует в обучении.
Для каждого семейства проверены календарь, календарь+лаги, полный набор с окнами.
Выбор по {config.selection_metric}: в пределах 1% от лучшего среднего предпочитаем меньше признаков,
затем меньший разброс и время. Это заранее заданное исследовательское правило, не бизнес-SLA.
CV использована для выбора, её метрики оптимистичны относительно независимой проверки.
Финальное обучение — до ноября с учётом задержки, калибровка — ноябрь.
Декабрь {config.year} уже изучен в версии 1; January_new — новый holdout только при первом зафиксированном запуске.
Повторный подбор по January_new лишает его независимости.

## Выбранная модель и признаки
Выбрано {chosen}: {len(model.columns)} признаков.
CV MAE {best.MAE_mean:.2f} ± {best.MAE_std:.2f}; train MAE {best.train_MAE_mean:.2f}.
Разрыв train/validation — повод для анализа сложности, не автоматическое доказательство переобучения.
Таблица cv_summary.csv содержит все группы, времена, размер модели и память.
{ablation_text}
Положительная разница означает преимущество полного набора по MAE на CV, отрицательная — сокращённого.
Если выбран бизнес-критерий, окончательный выбор может отличаться от лучшего MAE.
Наиболее высокий gain у признака {lag_gain}; gain не показывает причинное влияние и знак эффекта.
Permutation importance измеряет изменение MAE; коррелирующие лаги могут делить важность,
а перестановка создаёт нереалистичные сочетания. Групповая циклическая перестановка сохранена отдельно.
Объяснение на ноябре не используется для пересмотра зафиксированного выбора.
Для Ridge отдельно сохранены коэффициенты после предобработки и перестановочная важность.
Для WeeklyNaive интерпретация непосредственная: прогноз равен значению 168 часов назад.

{''.join(paragraphs)}

## Ресурсы и мониторинг
Инференс измерен 5 раз: медиана и p95 относятся к пакету строк, не API-SLA.
Память — выборочный peak RSS всего процесса, не точная память только модели; дополнительный RSS оценён приближённо.
Время CV fit отражает финальное обучение фолда; total_fold_seconds включает внутреннюю раннюю остановку и оценку.
Размер — сериализованный объект. LightGBM использует 2 CPU-потока, GPU не используется.
monitoring.json содержит PSI, качество и предупреждения. PSI >0.2 и покрытие <85% — диагностические
ориентиры, не согласованные бизнес-пороги; изменение сезонного состава само может менять распределение.
Контроль дрейфа не доказывает причину ухудшения. Для действующего сервиса нужен плановый запуск мониторинга.

## Ограничения
Прогнозируем реализованные очищенные поездки, не отклонённые заказы и не весь спрос.
Задержка истории в эксперименте: {config.availability_delay_hours} ч. Это параметр сценария, не измеренная задержка TLC.
Стоимость/дистанция известны после завершения поездки, поэтому готовность лагов онлайн не подтверждена.
Федеральные праздники — календарь pandas с переносом выходных; городские события не представлены.
DST-часы и затронутые окна исключаются; годовая устойчивость не доказана.
Ни улучшение качества, ни покрытие интервалов не гарантируются заранее.
Для внедрения нужны подтверждённый поток данных, бизнес-KPI и эксплуатационная проверка.

## Файлы
cv_folds.csv, cv_summary.csv — сравнение и ablation; optuna_*.csv — подбор;
interval_cv*.csv — выбор интервалов; evaluation.csv — декабрь и новый тест;
predictions.csv — прогнозы; errors_*.csv — ошибки сегментов;
importance_*.csv — интерпретация; environment.json — версии и хеш модели;
model.joblib — доверенный локальный артефакт; data_audit.csv — происхождение и очистка.
'''
    (artifacts/'REPORT.md').write_text(report,encoding='utf-8')
