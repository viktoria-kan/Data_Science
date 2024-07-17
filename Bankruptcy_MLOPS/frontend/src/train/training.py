"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
import optuna.visualization as vis
from ..plotting.charts import customize_plot


def start_training(config: dict, endpoint: object) -> dict:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # последние сохраненные значения метрик
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0, "logloss": 0}

    with st.spinner("Модель подбирает параметры..."):
        try:
            response = requests.post(endpoint, timeout=8005)
            response.raise_for_status()  # Проверка на успешный статус ответа
            st.success("Успешно!")
        except requests.RequestException as e:
            st.error(f"An error occurred: {e}")
            return

    # Проверка содержимого ответа
    try:
        output = response.json()
    except ValueError:
        st.error("Response content is not valid JSON")
        return

    # Проверка наличия ключа "metrics"
    if "metrics" in output:
        new_metrics = output["metrics"]
    else:
        st.error("Metrics not found in the response")

    # изменения в значениях метрик
    roc_auc, precision, recall, f1_metric, logloss = st.columns(5)
    roc_auc.metric(
        "ROC-AUC",
        new_metrics["roc_auc"],
        f"{new_metrics['roc_auc']-old_metrics['roc_auc']:.3f}",
    )
    precision.metric(
        "Precision",
        new_metrics["precision"],
        f"{new_metrics['precision']-old_metrics['precision']:.3f}",
    )
    recall.metric(
        "Recall",
        new_metrics["recall"],
        f"{new_metrics['recall']-old_metrics['recall']:.3f}",
    )
    f1_metric.metric(
        "F1 score", new_metrics["f1"], f"{new_metrics['f1']-old_metrics['f1']:.3f}"
    )
    logloss.metric(
        "Logloss",
        new_metrics["logloss"],
        f"{new_metrics['logloss']-old_metrics['logloss']:.3f}",
    )

    study = joblib.load(os.path.join(config["train"]["study_path"]))

    fig_history = vis.plot_optimization_history(study)
    fig_history = customize_plot(fig_history)
    st.plotly_chart(fig_history, use_container_width=True)

    try:
        fig_param_imp = vis.plot_param_importances(study)
        fig_param_imp = customize_plot(fig_param_imp)
        st.plotly_chart(fig_param_imp, use_container_width=True)
    except ValueError as e:
        if "Cannot evaluate parameter importances with only a single trial" in str(e):
            print(
                "Недостаточно завершенных экспериментов для оценки важности параметров."
            )
        else:
            raise e

    return new_metrics
