"""
Программа: Модель для прогнозирования банкротства компании
Версия: 1.0
"""

import warnings
import optuna
import pandas as pd
import uvicorn

from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

from fastapi import FastAPI, HTTPException
import logging


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"
logging.basicConfig(level=logging.INFO)


class Bankruptcy(BaseModel):
    """
    Признаки для получения результатов модели
    """

    ПрАудит: int
    Актив_ОбА_ДебЗад_Отч: float
    Актив_ОбА_ДебЗад_Пред: float
    Актив_ОбА_Запасы_Отч: float
    Актив_ОбА_Запасы_Пред: float
    Актив_ОбА_Запасы_ПредПред: float
    Актив_ОбА_Отч: float
    Актив_ОбА_Пред: float
    Актив_ОбА_ДенежнСр_Отч: float
    Актив_ОбА_ДенежнСр_Пред: float
    Актив_ОбА_ПредПред: float
    Актив_ВнеОбА_ОснСр_Отч: float
    Актив_ВнеОбА_ОснСр_Пред: float
    Актив_ВнеОбА_ОснСр_ПредПред: float
    Актив_ВнеОбА_Отч: float
    Актив_ВнеОбА_Пред: float
    Актив_ВнеОбА_ПредПред: float
    Актив_Отч: float
    Актив_Пред: float
    Актив_ПредПред: float
    Пассив_КапРез_Отч: float
    Пассив_КапРез_Пред: float
    Пассив_КапРез_ПредПред: float
    Пассив_КапРез_НераспПриб_Отч: float
    Пассив_КапРез_НераспПриб_Пред: float
    Пассив_КапРез_НераспПриб_ПредПред: float
    Пассив_КапРез_УставКапитал_Отч: float
    Пассив_КапРез_УставКапитал_Пред: float
    Пассив_КапРез_УставКапитал_ПредПред: float
    Пассив_Отч: float
    Пассив_Пред: float
    Пассив_ПредПред: float
    Пассив_ДолгосрОбяз_Отч: float
    Пассив_КраткосрОбяз_Отч: float
    Пассив_КраткосрОбяз_Пред: float
    Пассив_КраткосрОбяз_ПредПред: float
    Пассив_КраткосрОбяз_КредитЗадолж_Отч: float
    Пассив_КраткосрОбяз_КредитЗадолж_Пред: float
    Пассив_КраткосрОбяз_КредитЗадолж_ПредПред: float
    Актив_ОбА_ДебЗад_ПредПред: float
    Актив_ОбА_ДенежнСр_ПредПред: float
    Пассив_ДолгосрОбяз_Пред: float
    Пассив_ДолгосрОбяз_ПредПред: float
    Актив_ОбА_ПрочОбА_Отч: float
    Актив_ОбА_ПрочОбА_Пред: float
    Актив_ОбА_ПрочОбА_ПредПред: float
    Выруч_Отч: float
    ПрибПрод_Отч: float
    ПрочДоход_Отч: float
    СовФинРез_Отч: float
    ПрочРасход_Отч: float
    СебестПрод_Отч: float
    ЧистПрибУб_Отч: float
    ПрибУбДоНал_Отч: float
    ВаловаяПрибыль_Отч: float
    НалПриб_Отч: float
    ТекНалПриб_Отч: float
    ДвижКап_Итог_Пред: float
    ДвижКап_Итог_Отч: float
    ДвижКап_Итог_ПредПред: int
    ЧистАктив_Отч: float
    ЧистАктив_Пред: float
    ЧистАктив_ПредПред: float


@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {"message": "Hello Data Scientist!"}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    try:
        pipeline_training(config_path=CONFIG_PATH)
        logging.info("start load metrics")
        metrics = load_metrics(config_path=CONFIG_PATH)
        return {"metrics": metrics}
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result[:10]}


@app.post("/predict_input")
def prediction_input(bankruptcy: Bankruptcy):
    """
    Предсказание модели по введенным данным
    """

    features = [
        [
            bankruptcy.ПрАудит,
            bankruptcy.Актив_ОбА_ДебЗад_Отч,
            bankruptcy.Актив_ОбА_ДебЗад_Пред,
            bankruptcy.Актив_ОбА_Запасы_Отч,
            bankruptcy.Актив_ОбА_Запасы_Пред,
            bankruptcy.Актив_ОбА_Запасы_ПредПред,
            bankruptcy.Актив_ОбА_Отч,
            bankruptcy.Актив_ОбА_Пред,
            bankruptcy.Актив_ОбА_ДенежнСр_Отч,
            bankruptcy.Актив_ОбА_ДенежнСр_Пред,
            bankruptcy.Актив_ОбА_ПредПред,
            bankruptcy.Актив_ВнеОбА_ОснСр_Отч,
            bankruptcy.Актив_ВнеОбА_ОснСр_Пред,
            bankruptcy.Актив_ВнеОбА_ОснСр_ПредПред,
            bankruptcy.Актив_ВнеОбА_Отч,
            bankruptcy.Актив_ВнеОбА_Пред,
            bankruptcy.Актив_ВнеОбА_ПредПред,
            bankruptcy.Актив_Отч,
            bankruptcy.Актив_Пред,
            bankruptcy.Актив_ПредПред,
            bankruptcy.Пассив_КапРез_Отч,
            bankruptcy.Пассив_КапРез_Пред,
            bankruptcy.Пассив_КапРез_ПредПред,
            bankruptcy.Пассив_КапРез_НераспПриб_Отч,
            bankruptcy.Пассив_КапРез_НераспПриб_Пред,
            bankruptcy.Пассив_КапРез_НераспПриб_ПредПред,
            bankruptcy.Пассив_КапРез_УставКапитал_Отч,
            bankruptcy.Пассив_КапРез_УставКапитал_Пред,
            bankruptcy.Пассив_КапРез_УставКапитал_ПредПред,
            bankruptcy.Пассив_Отч,
            bankruptcy.Пассив_Пред,
            bankruptcy.Пассив_ПредПред,
            bankruptcy.Пассив_ДолгосрОбяз_Отч,
            bankruptcy.Пассив_КраткосрОбяз_Отч,
            bankruptcy.Пассив_КраткосрОбяз_Пред,
            bankruptcy.Пассив_КраткосрОбяз_ПредПред,
            bankruptcy.Пассив_КраткосрОбяз_КредитЗадолж_Отч,
            bankruptcy.Пассив_КраткосрОбяз_КредитЗадолж_Пред,
            bankruptcy.Пассив_КраткосрОбяз_КредитЗадолж_ПредПред,
            bankruptcy.Актив_ОбА_ДебЗад_ПредПред,
            bankruptcy.Актив_ОбА_ДенежнСр_ПредПред,
            bankruptcy.Пассив_ДолгосрОбяз_Пред,
            bankruptcy.Пассив_ДолгосрОбяз_ПредПред,
            bankruptcy.Актив_ОбА_ПрочОбА_Отч,
            bankruptcy.Актив_ОбА_ПрочОбА_Пред,
            bankruptcy.Актив_ОбА_ПрочОбА_ПредПред,
            bankruptcy.Выруч_Отч,
            bankruptcy.ПрибПрод_Отч,
            bankruptcy.ПрочДоход_Отч,
            bankruptcy.СовФинРез_Отч,
            bankruptcy.ПрочРасход_Отч,
            bankruptcy.СебестПрод_Отч,
            bankruptcy.ЧистПрибУб_Отч,
            bankruptcy.ПрибУбДоНал_Отч,
            bankruptcy.ВаловаяПрибыль_Отч,
            bankruptcy.НалПриб_Отч,
            bankruptcy.ТекНалПриб_Отч,
            bankruptcy.ДвижКап_Итог_Пред,
            bankruptcy.ДвижКап_Итог_Отч,
            bankruptcy.ДвижКап_Итог_ПредПред,
            bankruptcy.ЧистАктив_Отч,
            bankruptcy.ЧистАктив_Пред,
            bankruptcy.ЧистАктив_ПредПред,
        ]
    ]

    cols = [
        "ПрАудит",
        "Актив_ОбА_ДебЗад_Отч",
        "Актив_ОбА_ДебЗад_Пред",
        "Актив_ОбА_Запасы_Отч",
        "Актив_ОбА_Запасы_Пред",
        "Актив_ОбА_Запасы_ПредПред",
        "Актив_ОбА_Отч",
        "Актив_ОбА_Пред",
        "Актив_ОбА_ДенежнСр_Отч",
        "Актив_ОбА_ДенежнСр_Пред",
        "Актив_ОбА_ПредПред",
        "Актив_ВнеОбА_ОснСр_Отч",
        "Актив_ВнеОбА_ОснСр_Пред",
        "Актив_ВнеОбА_ОснСр_ПредПред",
        "Актив_ВнеОбА_Отч",
        "Актив_ВнеОбА_Пред",
        "Актив_ВнеОбА_ПредПред",
        "Актив_Отч",
        "Актив_Пред",
        "Актив_ПредПред",
        "Пассив_КапРез_Отч",
        "Пассив_КапРез_Пред",
        "Пассив_КапРез_ПредПред",
        "Пассив_КапРез_НераспПриб_Отч",
        "Пассив_КапРез_НераспПриб_Пред",
        "Пассив_КапРез_НераспПриб_ПредПред",
        "Пассив_КапРез_УставКапитал_Отч",
        "Пассив_КапРез_УставКапитал_Пред",
        "Пассив_КапРез_УставКапитал_ПредПред",
        "Пассив_Отч",
        "Пассив_Пред",
        "Пассив_ПредПред",
        "Пассив_ДолгосрОбяз_Отч",
        "Пассив_КраткосрОбяз_Отч",
        "Пассив_КраткосрОбяз_Пред",
        "Пассив_КраткосрОбяз_ПредПред",
        "Пассив_КраткосрОбяз_КредитЗадолж_Отч",
        "Пассив_КраткосрОбяз_КредитЗадолж_Пред",
        "Пассив_КраткосрОбяз_КредитЗадолж_ПредПред",
        "Актив_ОбА_ДебЗад_ПредПред",
        "Актив_ОбА_ДенежнСр_ПредПред",
        "Пассив_ДолгосрОбяз_Пред",
        "Пассив_ДолгосрОбяз_ПредПред",
        "Актив_ОбА_ПрочОбА_Отч",
        "Актив_ОбА_ПрочОбА_Пред",
        "Актив_ОбА_ПрочОбА_ПредПред",
        "Выруч_Отч",
        "ПрибПрод_Отч",
        "ПрочДоход_Отч",
        "СовФинРез_Отч",
        "ПрочРасход_Отч",
        "СебестПрод_Отч",
        "ЧистПрибУб_Отч",
        "ПрибУбДоНал_Отч",
        "ВаловаяПрибыль_Отч",
        "НалПриб_Отч",
        "ТекНалПриб_Отч",
        "ДвижКап_Итог_Пред",
        "ДвижКап_Итог_Отч",
        "ДвижКап_Итог_ПредПред",
        "ЧистАктив_Отч",
        "ЧистАктив_Пред",
        "ЧистАктив_ПредПред",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    result = (
        "Результат: Компания является потенциальным банкротом"
        if predictions == 1
        else (
            "Результат: Компания не является потенциальным банкротом"
            if predictions == 0
            else "Ошибка"
        )
    )
    return result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
