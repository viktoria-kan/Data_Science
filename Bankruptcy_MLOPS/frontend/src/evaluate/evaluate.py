"""
Программа: Отрисовка окон для ввода данных
с дальнейшим получением предсказания на основании
введенных/сгенерированных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st
from random import uniform, randint


def get_min_value(feature: str, unique_values: dict) -> float:
    """
    Получение минимального значения для признака
    :param feature: признак
    :param unique_values: уникальные значения признака
    :return: минимальное значение признака
    """
    return min(unique_values[feature])


def get_max_value(
    feature: str, unique_values: dict, params: any, user_inputs: dict
) -> float:
    """
    Получение максимального значения для признака с учетом условия
    :param feature: признак
    :param unique_values: уникальные значения признака
    :param params: параметры для признака
    :param user_inputs: значения признаков, введенные вручную
    :return: максимальное значение признака
    """
    if "condition" in params:
        if params["condition"]["type"] == "max_less_than_or_equal":
            condition_feature = params["condition"]["value"]
            condition_value = user_inputs.get(
                condition_feature, max(unique_values[condition_feature])
            )
            return min(max(unique_values[feature]), condition_value)
        elif params["condition"]["type"] == "equal":
            condition_feature = params["condition"]["value"]
            return user_inputs.get(
                condition_feature, unique_values[condition_feature][0]
            )
        elif params["condition"]["type"] == "or":
            min_val = min(unique_values[feature])
            max_val = max(unique_values[feature])
            st.text({min_val, max_val})
            return max_val
    if "max" in params:
        return max(unique_values[params["max"]])
    else:
        return max(unique_values[feature])


def generate_random_values(
    unique_values: dict, feature_params: dict, user_inputs: dict, other_features: list
) -> dict:
    """
    Генерация случайных значений для признаков
    :param unique_values: уникальные значения признаков
    :param feature_params: параметры признаков (допустимые значения и условия)
    :param user_inputs: значения признаков, введенные вручную
    :param other_features: остальные признаки
    :return: сгенерированные значения
    """
    random_values = {}
    for feature, params in feature_params.items():
        if feature in other_features:
            min_val = get_min_value(params["min"], unique_values)
            max_val = get_max_value(
                feature, unique_values, params["condition"], user_inputs
            )
            if isinstance(min_val, int) and isinstance(max_val, int):
                random_values[feature] = randint(min_val, max_val)
            else:
                random_values[feature] = uniform(min_val, max_val)
            user_inputs[feature] = random_values[feature]  # Обновляем user_inputs
    return random_values


def display_input_fields(
    unique_values: dict,
    feature_params: dict,
    user_inputs: dict,
    important_features: list = None,
    other_features: list = None,
    random_values: dict = None,
) -> None:
    """
    Вывод признаков на экран
    :param unique_values: уникальные значения признаков
    :param feature_params: параметры признаков (допустимые значения и условия)
    :param user_inputs: значения признаков, введенные вручную
    :param important_features: самые важные признаки, которые обязательно вводятся вручную
    :param other_features: остальные признаки
    :param random_values: значения признаков сгенерированные случайно
    """
    if important_features:
        for feature in important_features:
            params = feature_params[feature]
            min_val = get_min_value(params["min"], unique_values)
            max_val = get_max_value(feature, unique_values, params, user_inputs)

            input_key = f"{feature}_input"
            st.markdown(
                f'<h1 class="custom-title_6">Диапазон значений для {feature}: {min_val, max_val}</h1>',
                unsafe_allow_html=True,
            )
            if user_inputs.get(feature) is None:
                user_inputs[feature] = st.number_input(
                    label=params["label"],
                    min_value=min_val,
                    max_value=max_val,
                    key=input_key,
                )

    if other_features:
        for feature in other_features:
            params = feature_params[feature]
            min_val = get_min_value(params["min"], unique_values)
            max_val = get_max_value(feature, unique_values, params, user_inputs)

            # Уникальный ключ для виджета st.number_input
            input_key = f"{feature}_input"

            if random_values and feature in random_values:
                default_value = min(random_values[feature], max_val)
                user_inputs[feature] = st.number_input(
                    label=params["label"],
                    min_value=min_val,
                    max_value=max_val,
                    value=default_value,
                    key=input_key,
                )
            else:
                # Определяем тип данных признака и устанавливаем соответствующее значение по умолчанию
                feature_type = type(min_val)
                if feature_type == float:
                    default_value = 10.0  # Значение по умолчанию для int
                else:
                    default_value = 10
                st.markdown(
                    f'<h1 class="custom-title_6">Диапазон значений для {feature}: {min_val, max_val}</h1>',
                    unsafe_allow_html=True,
                )
                user_inputs[feature] = st.number_input(
                    label=params["label"],
                    min_value=min_val,
                    max_value=max_val,
                    key=input_key,
                    value=default_value,
                )


def evaluate_input(
    unique_data_path: str, param_features_path: str, endpoint: object
) -> str:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param param_features_path: путь до файла с параметрами для вводимых признаков
    :param endpoint: endpoint
    """
    with open(unique_data_path, encoding="utf-8") as file:
        unique_df = json.load(file)

    with open(param_features_path, encoding="utf-8") as file:
        feature_params = json.load(file)

    st.markdown("Заполните **основные** поля:")
    # Определение 10 самых важных признаков
    important_features = [
        "СебестПрод_Отч",
        "Актив_ВнеОбА_Пред",
        "Пассив_КапРез_НераспПриб_Пред",
        "ВаловаяПрибыль_Отч",
        "Актив_ОбА_Отч",
        "Актив_ОбА_ДенежнСр_Отч",
        "Актив_ВнеОбА_Отч",
        "Пассив_КапРез_УставКапитал_Отч",
        "Актив_ОбА_ДебЗад_Отч",
        "Актив_ВнеОбА_ОснСр_Отч",
        "Пассив_КапРез_НераспПриб_Отч",
        "Выруч_Отч",
        "ПрАудит",
    ]

    col1, col2 = st.columns(2)
    with col1:
        user_inputs = {}
        display_input_fields(
            unique_values=unique_df,
            feature_params=feature_params,
            user_inputs=user_inputs,
            important_features=important_features,
        )

    with col2:
        choice = st.radio(
            "Выберите режим ввода **остальных** признаков",
            ["Ручной ввод", "Сгенерировать значения рандомно"],
        )

        other_features = [
            feature for feature in feature_params if feature not in important_features
        ]

        if choice == "Ручной ввод":
            display_input_fields(
                unique_values=unique_df,
                feature_params=feature_params,
                user_inputs=user_inputs,
                other_features=other_features,
            )
        elif choice == "Сгенерировать значения рандомно":
            random_values = generate_random_values(
                unique_df, feature_params, user_inputs, other_features
            )
            st.warning(
                "Чтобы начать предсказание спуститесь в конец страницы", icon="💡"
            )
            st.markdown(
                '<h1 class="custom-title_4">Сгенерированные значения для остальных признаков:</h1>',
                unsafe_allow_html=True,
            )
            display_input_fields(
                unique_values=unique_df,
                feature_params=feature_params,
                user_inputs=user_inputs,
                other_features=other_features,
                random_values=random_values,
            )

    button_ok = st.button("Начать предсказание 🔮")
    if button_ok:
        with st.spinner("✨ делаем прогноз ✨"):
            result = requests.post(endpoint, timeout=8005, json=user_inputs)
            json_str = json.dumps(result.json())
            output = json.loads(json_str)
            st.success("Успешно!")
            st.write(f"{output}")
            return output


def evaluate_from_file(
    data: pd.DataFrame, endpoint: object, files: BytesIO
) -> pd.DataFrame:
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files: файл с данными
    :return: датасет
    """

    button_ok = st.button("Сделать предсказание")
    if button_ok:
        with st.spinner("происходит предсказание..."):
            # заглушка так как не выводим все предсказания
            data_ = data[:10]
            output = requests.post(endpoint, files=files, timeout=8000)
            data_["predict"] = output.json()["prediction"]
            col1, col2 = st.columns([8, 2])
            with col1:
                st.write(data_[:10])
            with col2:
                st.write(data_["predict"][:10])
            return data_[:10]
