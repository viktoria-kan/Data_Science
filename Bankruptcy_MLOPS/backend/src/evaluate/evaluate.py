"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import os
import yaml
import joblib
import pandas as pd
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess
import logging


def pipeline_evaluate(
    config_path: str, dataset: pd.DataFrame = None, data_path: str = None
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param dataset: датасет
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    :return: предсказания
    """

    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # preprocessing
    if data_path:
        dataset = get_dataset(dataset_path=data_path)
    logging.info("load test df successful")
    dataset = pipeline_preprocess(
        data=dataset, flg_evaluate=True, **preprocessing_config
    )
    logging.info("pipeline_preprocess completed")

    model = joblib.load(os.path.join(train_config["model_path"]))
    logging.info("prediction:")
    prediction = model.predict(dataset).tolist()

    return prediction
