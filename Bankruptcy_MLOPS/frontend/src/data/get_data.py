"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

from io import BytesIO
import io
from typing import Dict, Tuple
import streamlit as st
import pandas as pd


def get_dataset(dataset_path: str, sep: str = None) -> pd.DataFrame:
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


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_csv(data)
    st.write("Датасет успешно загружен ✔️")
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_csv(dataset_bytes_obj, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files
