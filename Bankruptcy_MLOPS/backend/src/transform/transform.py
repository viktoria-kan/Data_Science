"""
Программа: Предобработка данных
Версия: 1.0
"""

import json
import warnings
import pandas as pd
from typing import Union, Optional
import logging

warnings.filterwarnings("ignore")


def calculation_of_coeffs_div(
    data: pd.DataFrame, x: str, y: str, new_col: str, perc: int
) -> pd.DataFrame:
    """
    Расчёт финансовых коэффицицентов
    :param data: датасет
    :param x: признак в числителе
    :param y: признак в знаменателе
    :param new_col: новый признак(коэффициент)
    :param perc: процент, на который нужно умножить результат
    :return: датасет
    """
    # Если в формуле не нужно умножать результат на 100, то параметр perc=1
    try:
        result = data[x] / data[y] * perc
        result[data[y] == 0] = 0
        data[new_col] = result
    except KeyError as e:
        logging.error(f"KeyError: Column {e} not found in DataFrame.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
    return data


def calculation_of_coeffs_sum(
    data: pd.DataFrame, x: str, y: str, new_col: str
) -> pd.DataFrame:
    """
    Расчёт финансовых коэффицицентов
    :param data: датасет
    :param x: первое слагаемое
    :param y: второе слагаемое
    :param new_col: новый признак(коэффициент)
    :return: датасет
    """
    result = data[x] + data[y]
    data[new_col] = result
    return data


def calculation_of_coeffs_subtraction(
    data: pd.DataFrame, x: str, y: str, new_col: str
) -> pd.DataFrame:
    """
    Расчёт финансовых коэффицицентов
    :param data: датасет
    :param x: признак, из которого вычитаем (уменьшаемое)
    :param y: признак, который вычитаем (вычитаемое)
    :param new_col: новый признак(коэффициент)
    :return: датасет
    """
    result = data[x] - data[y]
    data[new_col] = result
    return data


def features_selection(data: pd.DataFrame, data_type: str) -> pd.Index:
    """
    Определение признаков, соответствующих указанному типу данных
    :param data: датасет
    :param data_type: тип данных
    :return: список признаков
    """
    num_cols = data.select_dtypes(include=[data_type]).columns
    return num_cols


def execute_tasks_from_file(data: pd.DataFrame, file_path: str) -> pd.DataFrame:

    with open(file_path, "r", encoding="utf-8") as file:
        tasks = json.load(file)

    functions = {
        "calculation_of_coeffs_div": calculation_of_coeffs_div,
        "calculation_of_coeffs_sum": calculation_of_coeffs_sum,
        "calculation_of_coeffs_subtraction": calculation_of_coeffs_subtraction,
    }

    logging.info("Go tasks")
    for task in tasks:
        func = functions[task["function"]]
        args = task["args"]

        try:
            data = func(data, *args)
        except Exception as e:
            logging.error(f"Error executing {task['function']} with args: {args} - {e}")
            raise e

    return data


def get_bins(
    data: Union[int, float],
    first_val: Union[int, float] = 0,
    second_val: Union[int, float] = 0,
) -> str:
    """
    Генерация бинов для разных признаков
    :param data: датасет
    :param first_val: первое пороговое значение
    :param second_val: второе пороговое значение
    :return: датасет
    """
    assert isinstance(data, (int, float)), "Неверный тип данных в признаке"
    result = (
        "Ниже нормы"
        if data < first_val
        else "Оптимально" if first_val <= data <= second_val else "Выше нормы"
    )
    return result


def features_selection(data: pd.DataFrame, data_type: str) -> pd.Index:
    """
    Определение признаков, соответствующих указанному типу данных
    :param data: датасет
    :param data_type: тип данных
    :return: список признаков
    """
    num_cols = data.select_dtypes(include=[data_type]).columns
    return num_cols


def check_columns_evaluate(
    data: pd.DataFrame, unique_values_path: str, flg_evaluate: bool
) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и добавление недостающих
    :param data: датасет
    :param unique_values_path: путь до списка с признаками train
    :param flg_evaluate: флаг для evaluate
    :return: список признаков
    """
    with open(unique_values_path, encoding="utf-8") as json_file:
        unique_values = json.load(json_file)

    column_sequence = list(unique_values.keys())

    # Добавление target в column_sequence, если это тренировочные данные
    if not flg_evaluate and "target" not in column_sequence:
        column_sequence.append("target")

    # Добавление недостающих признаков
    for col in column_sequence:
        if col not in data.columns:
            data[col] = 0

    # Удаление лишних признаков, кроме target, если это тренировочные данные
    if flg_evaluate:
        data = data[[col for col in column_sequence if col in data.columns]]
    else:
        data = data[
            [col for col in column_sequence if col in data.columns or col == "target"]
        ]

    return data[column_sequence]


def save_unique_train_data(
    data: pd.DataFrame, unique_values_path: str, target_column: Optional[str] = None
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """

    if target_column is not None:
        unique_df = data.drop(columns=[target_column], axis=1, errors="ignore")
    else:
        unique_df = data

    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def pipeline_preprocess(
    data: pd.DataFrame, flg_evaluate: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :param flg_evaluate: флаг для evaluate
    :return: датасет
    """

    logging.info("go to execute_tasks")
    data = execute_tasks_from_file(data, kwargs["tasks_path"])

    data = check_columns_evaluate(
        data=data,
        unique_values_path=kwargs["unique_values_path"],
        flg_evaluate=flg_evaluate,
    )

    # getbins
    assert isinstance(
        kwargs["bins_columns"], dict
    ), "Подайте тип данных для бинаризации в формате dict"
    for key in kwargs["bins_columns"].keys():
        data[f"{key}_bins"] = data[key].apply(
            lambda x: get_bins(
                x,
                first_val=kwargs["bins_columns"][key][0],
                second_val=kwargs["bins_columns"][key][1],
            )
        )

    cat_selected_cols = features_selection(data, kwargs["data_type"][1])

    data = pd.get_dummies(data, columns=cat_selected_cols, drop_first=True, dtype=int)

    data = check_columns_evaluate(
        data=data,
        unique_values_path=kwargs["uniq_val_path_with_binar"],
        flg_evaluate=flg_evaluate,
    )

    return data
