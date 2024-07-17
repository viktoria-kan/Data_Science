# Данные:
## Папки:
    data:
        raw - сырые данные из базы данных (с разметкой и без)
        processed - обработанные данные:
                         - train_ver1 - после файла 1.1_Preprocessing
                         - train_ver2 - после части EDA в файле 2_EDA_and_TRAIN
        check - обработанные (в файле 1.2_Preprocessing) данные без разметки
## Конфигурационный файл:
    - train_path_proc: сохранение после файла 1.1_Preprocessing
    - train_path_proc_2: сохранение после обработки в файле 2_EDA_and_train
    - unique_values_before_preproc: уникальные значения признаков после первичной обработки
    - unique_values_path: уникальные значения признаков
    - uniq_val_path_with_binar: уникальные значения признаков после бинаризации
    - input_features_path: файл содержит название признака, его минимальное, максимальное значения и условия для ввода вручную
    (например - значение признака X не должно быть больше чем признак Y)

# Полезные команды:
## Backend:
- Запуск приложения из папки backend, где --reload - указывает на автоматическое обновление при изменении кода:

    `cd backend`

    `uvicorn main:app --host=0.0.0.0 --port=8005 --reload`

Доступ к сервису FastAPI, при условии, что прописали ранее 8005 порт: http://localhost:8005/docs

### Убить все процессы:
    Get-Process | Where-Object {$_.ProcessName -like "uvicorn main:app*"} | Stop-Process -Force
ИЛИ найдите PID процесса:

- В отдельном терминальном окне выполните команду для поиска PID процесса, который слушает порт 8005 (для Windows):

    `netstat -ano | findstr :8005`

- После того как вы найдете PID процесса, используйте команду для его завершения (powershell):

`tasklist | findstr pycharm`

`taskkill /PID <PID> /F`

## Streamlit:
- Команды в отдельном терминале для запуска приложения Streamlit:

    `cd frontend`

    `streamlit run main.py` И приложение будет доступно по адресу: http://localhost:8501

Если запустить по конкретному порту, например 8080, то:

    `streamlit run main.py --server.port 8080`

### Убить процессы:
    Get-Process | Where-Object {$_.ProcessName -like "streamlit run"} | Stop-Process -Force