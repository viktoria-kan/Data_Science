"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def barplot_group(
    data: pd.DataFrame, col_main: str, col_group: str, data_x: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param col_main: признак для анализа по col_group
    :param col_group: признак для нормализации/группировки
    :param data_x: название оси Ох графика
    :return: поле рисунка
    """

    fig = plt.figure(figsize=(20, 10))

    # Создание копии данных для преобразования типа данных внутри функции
    data_copy = data[[col_main, col_group]].copy()
    data_copy[col_group] = data_copy[col_group].astype(str)

    grouped_data = (
        data_copy.groupby([col_group])[col_main]
        .value_counts(normalize=True)
        .rename("percentage")
        .mul(100)
        .reset_index()
        .sort_values(col_group)
    )

    ax = sns.barplot(
        x=col_main,
        y="percentage",
        hue=col_group,
        data=grouped_data,
        palette=["#369933", "#8D6149"],
    )
    fig.set_facecolor("#D8D1D1")
    ax.set_facecolor("#D8D1D1")
    # Изменение цвета оси X
    ax.spines["bottom"].set_color("#63224F")
    ax.spines["top"].set_color("#63224F")
    # Изменение цвета оси Y
    ax.spines["left"].set_color("#63224F")
    ax.spines["right"].set_color("#63224F")

    for p in ax.patches:
        percentage = "{:.1f}%".format(p.get_height())
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),  # координата xy
            ha="center",  # центрирование
            va="center",
            xytext=(0, 7),
            textcoords="offset points",  # точка смещения относительно координаты
            fontsize=16,
        )

    plt.ylabel("Проценты", fontsize=16)
    plt.xlabel(data_x, fontsize=16)
    legend = ax.legend(title="TARGET", fontsize=16)
    legend.get_frame().set_alpha(0.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    return fig


def kdeplotting(
    data: pd.DataFrame, data_x: str, title: str = None
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика kdeplot
    :param data: датасет
    :param data_x: ось OX
    :param title: название графика
    :return: поле рисунка
    """
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.kdeplot(data=data, common_norm=False, palette=["#369933", "#D6248D"])

    fig.set_facecolor("#D8D1D1")
    ax.set_facecolor("#D8D1D1")
    # Изменение цвета оси X
    ax.spines["bottom"].set_color("#63224F")
    ax.spines["top"].set_color("#63224F")
    # Изменение цвета оси Y
    ax.spines["left"].set_color("#63224F")
    ax.spines["right"].set_color("#63224F")

    plt.title(title, fontsize=30)
    plt.grid(True, color="gray")
    plt.ylabel("Распределение", fontsize=20)
    plt.xlabel(data_x, fontsize=20)
    legend = plt.legend(data, fontsize=18)
    # Изменение фона для legend
    legend.get_frame().set_alpha(0.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


def boxplotting(
    data: pd.DataFrame, x: str, y: str, lim: tuple, data_y: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика kdeplot
    :param data: датасет
    :param x: значения по оси Ох
    :param y: значения по оси Oy
    :param lim: диапазон для отображения значений на оси
    :param data_y: ось Oy
    :return: поле рисунка
    """

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("#D8D1D1")

    sns.boxplot(x=x, y=y, data=data, ax=ax, palette=["#369933", "#8D6149"])

    plt.ylabel(data_y, fontsize=20)
    plt.xlabel(x, fontsize=20)
    plt.legend(fontsize=18)

    plt.ylim(lim)
    plt.grid(True, color="gray")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)

    # Настройка цвета осей и текста
    ax.spines["bottom"].set_color("#63224F")
    ax.spines["left"].set_color("#63224F")
    ax.spines["top"].set_color("#63224F")
    ax.spines["right"].set_color("#63224F")
    ax.set_facecolor("#D8D1D1")

    return fig


def customize_plot(fig: Figure) -> matplotlib.figure.Figure:
    """
    Настройка цветов и шрифтов для графиков
    :param fig: график
    :return: измененный график
    """

    fig.update_layout(
        font=dict(family="Candara", size=20, color="#369933"),
        xaxis=dict(
            title=dict(font=dict(family="Candara", size=20, color="black")),
            tickfont=dict(family="Candara", size=17, color="black"),
        ),
        yaxis=dict(
            title=dict(font=dict(family="Candara", size=20, color="black")),
            tickfont=dict(family="Candara", size=17, color="black"),
        ),
        legend=dict(font=dict(family="Candara", size=17, color="black")),
    )

    # Изменяем цвет линий и маркеров
    for trace in fig["data"]:
        trace["marker"]["color"] = "#8D6149"

    return fig
