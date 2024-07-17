"""
Программа: Получение данных из файла
Версия: 1.0
"""

from typing import Text
import pandas as pd


def get_dataset(dataset_path: Text, sep: str = None) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :param sep: разделитель в данных
    :return: датасет
    """
    if sep:
        return pd.read_csv(dataset_path, sep=sep)
    else:
        return pd.read_csv(dataset_path)
