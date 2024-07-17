"""
Программа: Тренировка данных
Версия: 1.0
"""

import optuna
from optuna import Study

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics
import logging


def objective(
    trial, data_x: np.array, data_y: np.array, n_folds: int, random_state: int
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """

    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    solver = trial.suggest_categorical(
        "solver", ["liblinear", "saga", "lbfgs", "newton-cg", "sag"]
    )
    tol = trial.suggest_loguniform("tol", 1e-6, 1e-2)
    class_weight = trial.suggest_categorical("class_weight", ["balanced"])
    random_state = trial.suggest_categorical("random_state", [random_state])

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_uniform("l1_ratio", 0, 1)

    # Исключение несовместимых комбинаций
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        raise optuna.exceptions.TrialPruned()
    if penalty == "l2" and solver == "liblinear":
        raise optuna.exceptions.TrialPruned()
    if penalty == "elasticnet" and solver != "saga":
        raise optuna.exceptions.TrialPruned()
    if penalty == None and solver in ["liblinear", "saga"]:
        raise optuna.exceptions.TrialPruned()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_predicts = np.empty(n_folds)

    x_train_df = pd.DataFrame(data_x)
    logging.info(f"dtypes x_train: {type(x_train_df)}, y_train:{type(data_y)}")

    for idx, (train_idx, test_idx) in enumerate(skf.split(x_train_df, data_y)):
        x_train_fold, x_valid_fold = (
            x_train_df.iloc[train_idx],
            x_train_df.iloc[test_idx],
        )
        y_train_fold, y_valid_fold = data_y.iloc[train_idx], data_y.iloc[test_idx]

        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            tol=tol,
            l1_ratio=l1_ratio,
            class_weight=class_weight,
            random_state=random_state,
        )

        model.fit(
            x_train_fold,
            y_train_fold,
        )
        predict = model.predict_proba(x_valid_fold)[:, 1]
        cv_predicts[idx] = roc_auc_score(y_valid_fold, predict)

    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, d_type: list, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :param d_type: тип данных
    :return: [LogisticRegression tuning, Study]
    """
    x_train_std, x_test_std, y_train, y_test = get_train_test_data(
        data_train=data_train,
        data_test=data_test,
        target=kwargs["target_column"],
        d_type=d_type[0],
    )

    study = optuna.create_study(direction="maximize", study_name="LogReg")
    logging.info(f"dtypes x_train: {type(x_train_std)}, y_train:{type(y_train)}")

    function = lambda trial: objective(
        trial, x_train_std, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)

    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    d_type: list,
    **kwargs,
) -> LogisticRegression:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param d_type: тип данных
    :return: LogisticRegression
    """

    logging.info("start train model")
    x_train_std, x_test_std, y_train, y_test = get_train_test_data(
        data_train=data_train,
        data_test=data_test,
        target=kwargs["target_column"],
        d_type=d_type[0],
    )

    finish_test_preds = []
    finish_test_preds_proba = []

    cv = StratifiedKFold(
        n_splits=kwargs["n_folds"], shuffle=True, random_state=kwargs["random_state"]
    )
    cv_predicts_val = np.empty(kwargs["n_folds"])

    x_train_df = pd.DataFrame(x_train_std)

    for idx, (train_idx, test_idx) in enumerate(cv.split(x_train_df, y_train)):
        x_train_, x_val = x_train_df.iloc[train_idx], x_train_df.iloc[test_idx]
        y_train_, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

        clf = LogisticRegression(**study.best_params)
        clf.fit(x_train_, y_train_)

        # OOF
        preds_val_proba = clf.predict_proba(x_val)[:, 1]
        cv_predicts_val[idx] = roc_auc_score(y_val, preds_val_proba)

        # holdout
        preds_test = clf.predict(x_test_std)
        preds_test_proba = clf.predict_proba(x_test_std)

        finish_test_preds.append(preds_test)
        finish_test_preds_proba.append(preds_test_proba)

    # сохранение метрик
    logging.info("try save metrics")
    save_metrics(
        data_x=x_test_std, data_y=y_test, model=clf, metric_path=kwargs["metrics_path"]
    )
    logging.info("save successful")
    return clf
