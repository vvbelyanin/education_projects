# Подготовка виртуальной машины
В командной строке:
 - клонирование репозитория
```
git clone https://github.com/vvbelyanin/mle-sprint5-project
```
- обновление установленных пакетов
```
sudo apt-get update
```
- установка пакета виртуального окружения Python
```
sudo apt-get install python3.10-venv
```
- создание виртуальной среды
```
python3.10 -m venv env_recsys_start
```
- активирование окружения 
```
. env_recsys_start/bin/activate
```
- фиксация конкретных версий пакетов
```
pip install -r requirements.txt
```
- в текущей папке должен быть файл .env со следующими кредами
```
S3_BUCKET_NAME=<...>
AWS_ACCESS_KEY_ID=<...>
AWS_SECRET_ACCESS_KEY=<...>
S3_ENDPOINT_URL=<...>
```
- загрузка переменных окружения из файла .env
```
export $(cat .env | xargs)
```
# Загрузка файлов данных в каталог /parquet
```
mkdir -p ./parquet
wget -P ./parquet https://storage.yandexcloud.net/mle-data/ym/tracks.parquet
wget -P ./parquet https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet
wget -P ./parquet https://storage.yandexcloud.net/mle-data/ym/interactions.parquet
```
# Код части 1 проекта в Jupyter notebook  

[recommendations.ipynb](https://github.com/vvbelyanin/mle-sprint5-project/blob/main/recommendations.ipynb)

- код выполнялся в VS Code 1.91.1
- при необходимости - запуск Jupyter Lab
```
jupyter lab --ip=0.0.0.0 --no-browser
```
# Код части 2 проекта в скриптах Python

Для корректной работы сервиса необходимы данные, полученные в части 1:
- parquet/recommendations.parquet
- parquet/similar.parquet
- parquet/top_popular.parquet
- parquet/items.parquet
- parquet/catalog_names.parquet
  
Если такого каталога нет, то он будет создан, и данные будут загружены из 
S3-бакета:  
```
s3-student-mle-20240325-4062b25c06
``` 
Cервис рекомендаций:
[recommendation_service.py](https://github.com/vvbelyanin/mle-sprint5-project/blob/main/recommendation_service.py).  

Тестирование сервиса:
[test_service.py](https://github.com/vvbelyanin/mle-sprint5-project/blob/main/test_service.py).  

Запуск и тестирование сервиса рекомендаций
- активирование окружения, если не было сделано
```
. env_recsys_start/bin/activate
```
- запуск сервиса
```
python3 recommendation_service.py
```
- проверка порта
```
lsof -i:8000
```
- проверка (вывод топ-5 популярных треков)
```
curl http://127.0.0.1:8000/top_popular/5
```
- проверка эндпоинтов Swagger UI и ReDoc
```
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```
- тестирование сервиса средствами unittest (в другом терминале)
```
python3 test_service.py
```
Cкрипт генерирует [test_service.log](https://github.com/vvbelyanin/mle-sprint5-project/blob/main/test_service.log), который выводится на экран после окончания тестирования.  

Остановка сервиса (в терминале, где производился запуск) по Ctrl+C.