"""
Программа: Разделение данных на train/test
Версия: 1.0
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ..transform.transform import features_selection
import logging


def split_train_test(dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Разделение данных на train/test с последующим сохранением
    :param dataset: датасет
    :return: train/test датасеты
    """
    try:
        logging.info(f"splitting")
        df_train, df_test = train_test_split(
            dataset,
            stratify=dataset[kwargs["target_column"]],
            test_size=kwargs["test_size"][0],
            random_state=kwargs["random_state"],
        )
        return df_train, df_test
    except Exception as e:
        logging.error(f"Error split {e}")
        raise e


def get_train_test_data(
    data_train: pd.DataFrame, data_test: pd.DataFrame, target: str, d_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/test данных разбитых по отдельности на объект-признаки и целевую переменную
    :param data_train: train датасет
    :param data_test: test датасет
    :param target: название целевой переменной
    :param d_type:
    :return: набор данных train/test
    """
    try:
        logging.info(
            f"splitting on get_train_test_data {data_train.shape}, {data_test.shape}"
        )
        x_train, x_test = (
            data_train.drop(target, axis=1),
            data_test.drop(target, axis=1),
        )
        y_train, y_test = (
            data_train.loc[:, target],
            data_test.loc[:, target],
        )

        num_selected_cols = features_selection(x_train, d_type)
        scaler = StandardScaler()
        x_train_std = scaler.fit_transform(x_train[num_selected_cols])
        x_test_std = scaler.transform(x_test[num_selected_cols])

        return x_train_std, x_test_std, y_train, y_test
    except Exception as e:
        logging.error(f"Error get train-test {e}")
        raise e
