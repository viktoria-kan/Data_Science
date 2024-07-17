"""
Программа: Пайплайн для тренировки модели
Версия: 1.0
"""

import os
import joblib
import yaml
import logging
from ..data.split_dataset import split_train_test
from ..train.train import find_optimal_params, train_model
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param config_path: путь до файла с конфигурациями
    :return: None
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    train_data = get_dataset(dataset_path=preprocessing_config["train_path_proc"])
    logging.info("get_dataset completed successfully")

    train_data = pipeline_preprocess(
        data=train_data, flg_evaluate=False, **preprocessing_config
    )
    logging.info("execute and preproc successfully")

    df_train, df_test = split_train_test(dataset=train_data, **preprocessing_config)
    logging.info("Split successfully")

    logging.info("Start find opt params")
    study = find_optimal_params(
        data_train=df_train,
        data_test=df_test,
        d_type=preprocessing_config["data_type"],
        **train_config
    )

    clf = train_model(
        data_train=df_train,
        data_test=df_test,
        study=study,
        d_type=preprocessing_config["data_type"],
        **train_config
    )
    logging.info("train completed")
    joblib.dump(clf, os.path.join(train_config["model_path"]))
    joblib.dump(study, os.path.join(train_config["study_path"]))
    logging.info("save study, model successful")
