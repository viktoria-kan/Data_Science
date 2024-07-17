"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import barplot_group, kdeplotting, boxplotting
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file
from src.styles.customization import custom_css, load_global_styles, color_panel


CONFIG_PATH = "../config/params.yml"
st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)
st.markdown(custom_css(), unsafe_allow_html=True)
st.markdown(load_global_styles(), unsafe_allow_html=True)


with st.sidebar:
    st.image("logo_3.png", width=400)
    st.image("cat_4.png", width=400)


def data_page():
    """
    Страница с описанием данных
    """
    st.write(
        "Данные включают в себя финансовые показатели компаний, а также дополнительно рассчитанные         "
        "финансовые коэффициенты (например, коэффициенты ликвидности, рентабельность и др.). "
        "При прогнозировании не учитывается такая информация, как сфера деятельности компании, "
        "экономическая обстановка в стране/мире, расположение, и т.д."
    )
    st.markdown(
        '<h1 class="custom-title_3">Описание полей:</h1>', unsafe_allow_html=True
    )
    st.markdown(
        """
        ### Target
        - Целевая переменная: 1 - компания является банкротом; 0 - компания не является банкротом"""
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        ### Активы:
        - Актив_Отч - активы компании за *отчетный* период
        - Актив_ОбА_Отч - оборотные активы за отчетный период
        - Актив_ОбА_ДебЗад_Отч - дебиторская задолженность на отчетную дату отчетного периода
        - Актив_ОбА_Запасы_Отч - запасы на отчетную дату отчетного периода
        - Актив_ОбА_ДенежнСр_Отч - денежные средства
        - Актив_ОбА_ПрочОбА_Отч - прочие оборотные активы
        - Актив_ВнеОбА_Отч - внеоборотные активы за отчетный период
        - Актив_ВнеОбА_ОснСр_Отч - основные средства"""
        )
    with col2:
        st.markdown(
            """
        ### Пассивы:
        - Пассивы_Отч - пассивы компании за отчетный период
        - Пассив_КраткосрОбяз/ДолгосрОбяз.. - краткосрочные/долгосрочные обязательства
        - Пассив_КраткосрОбяз_КредитЗадолж_Отч - кредиторская задолженность
        - Пассив_КапРез.. - капитал и резервы
        - Пассив_КапРез_НераспПриб_Отч - нераспредлеленная прибыль
        - Пассив_КапРез_УставКапитал_Отч - уставный капитал"""
        )
    st.markdown(
        """
        ### Остальные признаки:
        - ПрАудит - необходимость аудита: 1 - компания должна пройти аудит; 0 - компания не должна проходить аудит
        - Выруч_Отч - выручка
        - ПрибПрод_Отч - прибыль(убыток) от продаж
        - ПрочДоход_Отч - прочие доходы
        - СовФинРез_Отч - совокупный финансовый результат
        - ПрочРасход_Отч - прочие расходы
        - СебестПрод_Отч - себестоимость продаж
        - ПрибУбДоНал_Отч - прибыль(убыток) до налогообложения
        - ВаловаяПрибыль_Отч - валовая прибыль(убыток)
        - НалПриб_Отч - налог на прибыль
        - ТекНалПриб_Отч - текущий налог на прибыль
        - ЧистАктив_Отч - чистые активы
        - ЧистПрибУб_Отч - чистая прибыль(убыток)
        - ДвижКап_Итог_Отч - величина капитала на 31 декабря отчетного года
    """
    )
    st.warning(
        "Данные описаны на примере **отчётного** периода. В датасете также присутствуют "
        "данные за предыдущий и позапрошлые годы (по отношению к отчетному). Они имеют аналогичное наименование "
        "за исключением окончаний (вместо Отч - **Пред** и **ПредПред** соответственно)",
        icon="ℹ️",
    )
    st.markdown(
        """
        ### Финансовые коэффициенты (рассчитываются и добавляются к данным автоматически):
        - ЧистНормПриб - чистая норма прибыли
        - ВаловаяРент - валовая рентабельность
        - РентабОперДеят - рентабельность по операционной деятельности
        - СовокупДолг - совокупный долг
        - Коэф_ТекущЛиквид - коэффициент текущей ликвидности
        - Рентаб_СобствКап - рентабельность собственного капитала
        - Мультиплик_СобствКап - мультипликатор собственного капитала
        - Эффект_Использ_Актив - эффективность использования активов
    """
    )


def home_page():
    """
    Главная страница с выбором раздела
    """
    st.markdown(
        """
            <ul class="custom-title_4">
            Банкротство компании — это неспособность предприятия в полном объеме удовлетворить требования кредиторов
            по денежным обязательствам и/или исполнить обязанности по уплате обязательных платежей.
            Прогнозирование банкротства поможет снизить потенциальные риски для инвесторов, клиентов, партнеров.
            </ul>
            """,
        unsafe_allow_html=True,
    )
    st.image("cat_3.jpg", width=600)


def exploratory():
    """
    Страница с анализом данных
    """
    st.markdown('<h1 class="custom-title_2">Анализ данных</h1>', unsafe_allow_html=True)

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    graphs = [
        "Влияние изменения доля дебиторской задолженности и запасов",
        "Влияние изменения доли кредиторской задолженности",
        "Влияние отношения дебиторской задолженности к кредиторской",
        "Влияние коэффициента текущей ликвидности",
    ]

    data = get_dataset(dataset_path=config["preprocessing"]["train_path_proc_2"])
    choice = st.selectbox("**Выберите нужный раздел ниже**", graphs)
    st.markdown("-----")

    st.write(data.head())
    st.warning(
        "Графики ниже представлены с целью оценки закономерностей в имеющихся данных. Отдельные финансовые "
        "показатели не являются индикатором статуса компании."
    )

    if choice == "Влияние изменения доля дебиторской задолженности и запасов":
        st.markdown(
            '<h1 class="custom-title_3">Дебиторская задолженность</h1>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<h1 class="custom-title_5">Распределение значений доли дебит. задолженности</h1>',
                unsafe_allow_html=True,
            )
            st.pyplot(
                kdeplotting(
                    data={
                        "банкрот": data[data.target == 1].Доля_ДебЗадолж,
                        "действующая": data[data.target == 0].Доля_ДебЗадолж,
                    },
                    data_x="Доля дебит. задолженности",
                )
            )

        with col2:
            st.markdown(
                '<h1 class="custom-title_5">Доля дебиторской задолженности - boxplot</h1>',
                unsafe_allow_html=True,
            )
            st.pyplot(
                boxplotting(
                    data=data,
                    x="target",
                    y="Доля_ДебЗадолж",
                    lim=None,
                    data_y="Доля дебит. задолженности",
                )
            )
        st.markdown(
            """
            <ul class="custom-title_4"></ul>
                <li>Распределение значений доли деб. задолженности отличается для банкротов и действующих компаний. 
                Для банкротов не характерна доля деб. задолженности меньше нуля, но больше компаний имеют значения
                в  промежутке от 0 до 75 процентов</ul>
                <li>Так как для действующих компаний характерна либо низкая доля деб. задолженности (ближе к 0), 
                либо очень высокая (от 75%) можно предположить, что действующие компании имеют эффективное управление 
                деб. задолженностью и/или могут позволить себе больше рисковать за счёт более устойчивого 
                финансового положения</ul>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<h1 class="custom-title_3">Запасы</h1>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                '<h1 class="custom-title_5">Распределение значений доли запасов</h1>',
                unsafe_allow_html=True,
            )
            st.pyplot(
                kdeplotting(
                    data={
                        "банкрот": data[data.target == 1].Доля_Запасов,
                        "действующая": data[data.target == 0].Доля_Запасов,
                    },
                    data_x="Доля запасов",
                )
            )
        with col4:
            st.markdown(
                '<h1 class="custom-title_5">Доля запасов - boxplot</h1>',
                unsafe_allow_html=True,
            )
            st.pyplot(
                boxplotting(
                    data=data,
                    x="target",
                    y="Доля_Запасов",
                    lim=None,
                    data_y="Доля запасов",
                )
            )
        st.markdown('<h1 class="custom-title_4"></h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="custom-title_4"></ul>
            <li>Для компаний банкротов зачастую характерна минимальная доля запасов (но присутствует много выбросов)</ul>
            <li>Действующие компании зачастую имеют доли запасов в размере 20-60% </ul>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    if choice == "Влияние изменения доли кредиторской задолженности":
        st.markdown(
            '<h1 class="custom-title_3">Динамика кредиторской задолженности</h1>',
            unsafe_allow_html=True,
        )
        col5, col6 = st.columns(2)
        with col5:
            st.markdown(
                '<h1 class="custom-title_5">Динамика за предыдущий и позапрошлый гг.</h1>',
                unsafe_allow_html=True,
            )
            st.pyplot(
                boxplotting(
                    data=data,
                    x="target",
                    y="Динам_КрЗадолж_Пред_ПредПред",
                    lim=(-1e2, 1e2),
                    data_y="Динамика кредиторской задолженности",
                )
            )
        with col6:
            st.markdown(
                '<h1 class="custom-title_5">Динамика за отчетный и предыдущий гг.</h1>',
                unsafe_allow_html=True,
            )
            st.pyplot(
                boxplotting(
                    data=data,
                    x="target",
                    y="Динам_КрЗадолж_Отч_Пред",
                    lim=(-1e2, 1e2),
                    data_y="Динамика кредиторской задолженности",
                )
            )
        st.markdown(
            """
            <ul class="custom-title_4"></ul>
            <li>Для действующих компаний характерно очень большое количество выбросов</ul> 
            <li>Медиана находится на одном уровне для target 1 и 0</ul>
            <li>У компаний банкротов бОльший разброс среди значений выше медианы, особенно в динамике</ul> 
            "отчётный-предыдущий гг"</ul>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        col5_1, col6_1 = st.columns([2, 8])
        with col5_1:
            st.markdown(
                '<h1 class="custom-title_4">Средние значения также искажены из-за компаний '
                "с большими выбросами:</h1>",
                unsafe_allow_html=True,
            )
            grouped_data = data.groupby("target")["Динам_КрЗадолж_Отч_Пред"].mean()
            output_text = grouped_data.to_string()
            st.text(output_text)
        with col6_1:
            st.markdown(
                """
                <ul class="custom-title_4"></ul>
                <li>Несмотря на то, что некоторые компании имеют высокий рост кредит. задолженности, 
                они не являются банкротами</ul>
                <li>Однако такие компании могут быть потенциальными банкротами</ul>
                </ul>
                """,
                unsafe_allow_html=True,
            )

    if choice == "Влияние отношения дебиторской задолженности к кредиторской":
        st.markdown(
            '<h1 class="custom-title_3">Отношение дебиторской задолженности к кредиторской</h1>',
            unsafe_allow_html=True,
        )
        color_panel(
            "pink", "gray", "black", "Оптимальное значение отношения: [0.9:1.1]"
        )
        st.markdown(
            '<h1 class="custom-title_5">Отношение дебит. задолженности к кредиторской - target</h1>',
            unsafe_allow_html=True,
        )
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Отнош_ДебитКредит_bins",
                col_group="target",
                data_x="Отношение дебиторской задолженности к кредиторской (бины)",
            )
        )
        st.markdown("---")
        st.markdown(
            """
            <ul class="custom-title_4"></ul>
            <li>Из графика видно, что оптимальное значение отношения дебит. задолженности к кредиторской 
            имеет маленькое кол-во компаний в целом</ul>
            <li>У компаний банкротов больше доля значений ниже нормы, что означает превышение кредит. задолженности 
            над дебиторской</ul>
            <li>Действующие компании имеют чуть больший процент оптимальных значений</ul>
            <li>Для действующих компаний значение ниже нормы может повысить шанс того, что компания не сможет 
            погасить своих обязательства</ul>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    if choice == "Влияние коэффициента текущей ликвидности":
        # st.markdown(
        #     """
        #     - Слишком низкий показатель указывает на то, что компания может не оплатить все обязательства в срок
        #     (коэфф-т 1 означает, что для оплаты текущих обязательств компании нужно продать все оборотные активы)
        #     - Оптимальные значения: [1.5:2.0]
        #     - но коэф-т текущей ликвидности рассчитывался в процентах, поэтому диапазон [150:200]
        #     """
        # )
        st.markdown(
            '<h1 class="custom-title_3">Коэффициент текущей ликвидности</h1>',
            unsafe_allow_html=True,
        )
        color_panel("pink", "gray", "black", "Оптимальные значения: [1.5:2.0]")
        st.markdown(
            """
            <ul class="custom-title_4"></ul>
            <li>Так как коэффициент текущей ликвидности рассчитывался в процентах, диапазон имет вид [150:200]</ul>
            <li>Слишком низкий показатель указывает на то, что компания может не оплатить все обязательства в срок</ul>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h1 class="custom-title_5">Коэффициент текущей ликвидности - target</h1>',
            unsafe_allow_html=True,
        )
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Коэф_ТекущЛиквид_Отч_bins",
                col_group="target",
                data_x="Коэффициент текущей ликвидности (бины)",
            )
        )
        st.markdown("---")
        st.markdown(
            """
            <ul class="custom-title_4"></ul>
            <li>Для большинства банкротов характерно значение коэффициента ликвидности ниже нормы</ul>
            <li>Оптимальное значение коэффициента ликвидности имеет небольшая доля компаний в целом, в том числе 
                среди действующих (однако среди действующих процент в два с половиной раза выше)</ul>
            <li>Для компаний банкротов характерен коэффициент ликвидности ниже нормы</ul>
            """,
            unsafe_allow_html=True,
        )
        # st.markdown('<h1 class="custom-title_5">Динамика коэффициента  за отчетный и предыдущий гг. - target</h1>',
        #             unsafe_allow_html=True)
        # st.pyplot(
        #     barplot_group(
        #         data=data,
        #         col_main='Динам_КоэфТекЛиквид_ОтчПред_bins',
        #         col_group='target',
        #         data_x='динамика коэффициента текущей ликвидности (бины)'
        #     )
        # )
        # st.markdown('---')
        # st.markdown(
        #     """
        #     <ul class="custom-title_4">
        #         <li>Для большинства банкротов характерно значение коэф-та ликвидности ниже нормы
        #         <li>Оптимальное значение коэффициента ликвидности имеет небольшая доля компаний в целом, в том числе
        #         среди действующих (однако среди действующих процент в два раза выше)</ul>
        #     """,
        #     unsafe_allow_html=True
        # )
        # st.markdown('<h1 class="custom-title_4">Нельзя сделать вывод о том, что динамика коэф-та ликвидности в одну '
        #             'или другую сторону влияет на статус компании</h1>', unsafe_allow_html=True)
        # st.markdown('<h1 class="custom-title_4">Финансовые данные компаний нужно смотреть в совокупности. '
        #             'Анализ отдельных признаков зачастую не показывает прямого влияния на целевую переменную. '
        #             'Но нужно учитывать, что изменение этих признаков может быть одним из множества факторов, '
        #             'влияющих на статус компании</h1>', unsafe_allow_html=True)


def training():
    """
    Тренировка модели
    """
    st.markdown(
        '<h1 class="custom-title_2">Тренировка модели</h1>', unsafe_allow_html=True
    )

    if "train_result" not in st.session_state:
        st.session_state.train_result = None
    if st.session_state.train_result:
        st.markdown("Прошлый результат:")
        st.markdown(st.session_state.train_result)

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["train"]

    if st.button("Начать тренировку ⌛"):
        st.session_state.train_result = start_training(config=config, endpoint=endpoint)

    if st.button("Очистить результат 🗑️"):
        st.session_state.train_result = None
        st.experimental_rerun()


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown('<h1 class="custom-title_2">Предсказание</h1>', unsafe_allow_html=True)

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["prediction_input"]

    # unique_data_path = config["preprocessing"]["uniq_val_path_with_binar"]
    unique_data_path = config["preprocessing"]["unique_values_before_preproc"]
    param_features_path = config["preprocessing"]["input_features_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        st.session_state.prediction_result = evaluate_input(
            unique_data_path=unique_data_path,
            param_features_path=param_features_path,
            endpoint=endpoint,
        )
    else:
        st.error("Сначала обучите модель")

    if st.button("Очистить результат"):
        st.session_state.prediction_result = None
        st.experimental_rerun()


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown(
        '<h1 class="custom-title_2">Предсказание по данным из файла</h1>',
        unsafe_allow_html=True,
    )

    if "predict_result" not in st.session_state:
        st.session_state.predict_result = None

    if "clear_result" not in st.session_state:
        st.session_state.clear_result = False

    if st.session_state.predict_result is not None:
        st.markdown("Предсказанные данные (прошлые):")
        st.write(st.session_state.predict_result)

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            st.session_state.predict_result = evaluate_from_file(
                data=dataset_csv_df, endpoint=endpoint, files=files
            )
        else:
            st.error("Сначала обучите модель")

    if st.button("Очистить результат 🗑️"):
        st.session_state.predict_result = None
        st.session_state.clear_result = True

    if st.session_state.clear_result:
        st.session_state.clear_result = False
        st.experimental_rerun()


def main():
    """
    Сборка пайплайна в одном блоке
    """
    st.markdown(custom_css(), unsafe_allow_html=True)
    st.markdown(
        "<h1 class='custom-title'>Прогнозирование банкротства компаний</h1>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(
        [
            "⚙️ **ГЛАВНОЕ МЕНЮ**",
            "📚 **ОПИСАНИЕ ДАННЫХ**",
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        with tab1:
            # main_page()
            page_names_to_funcs = {
                "Ожидание": home_page,
                "Анализ данных": exploratory,
                "Тренировка модели": training,
                "Предсказание по данным введенным вручную": prediction,
                "Предсказание по данным из файла": prediction_from_file,
            }
            choice_two = st.radio("**Выберите раздел:**", page_names_to_funcs)
            page_names_to_funcs[choice_two]()

    with col2:
        with tab2:
            data_page()


if __name__ == "__main__":
    main()
