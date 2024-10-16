Тема: Продакшн машинного обучения  
Название: Деплой и интеграция модели для предсказания стоимости объектов недвижимости  
Описание: Заказчик является компанией-маркетплейсом для аренды и покупки жилой и коммерческой недвижимости. Был разработан MVP алгоритма на основе машинного обучения для оценки рыночной стоимости квартиры по её характеристикам. Был проведена оптимизация модели. Необходимо вывести в продакшн разработанную модель. 


Этапы решения задачи    
| # | Описание | Инструменты/Ключевые слова |
|:----:|:---------------------------|:-----------------------------------------------------------|
| 1 | Разработка FastAPI микросервиса | `FastAPI` `uvicorn`|
| 2 | Контейнеризация микросервиса | `Docker` `Docker Compose` `unittest`|
| 3 | Развертывание системы мониторинга | `Prometheus`|
| 4 | Настройка дашборда для мониторинга в Grafana | `Grafana`|


## Проект на тему "Релиз модели в продакшн"

##### Описание задачи
Вывод в продакшн разработанной на предыдущих этапах модели для Яндекс Недвижимости, маркетплейсе для аренды и покупки жилой и коммерческой недвижимости. 

##### Бизнес-задачи
- Разработка онлайн-сервиса для предсказания модели цены на недвижимось    
- Обеспечение кроссплатформенности сервиса для работы с другими сервисами компании    
- Настройка мониторинга работы сервиса для своевременного извещения о потенциальных рисках

##### Задачи инженера машинного обучения
- Разработка FastAPI микросервиса
- Контейнеризация микросервиса с помощью Docker
- Развертывание системы мониторинга с использованием Prometheus и Grafana
- Настройка дашборда для мониторинга в Grafana

##### Используемые инструменты
- Visual Studio Code
- FastAPI, uvicorn
- Docker и Docker Compose
- Prometheus
- Grafana
- Python-библиотеки для экспортёров: prometheus_client, prometheus_fastapi_instrumentator.

### Выполнение проекта

#### Файловая структура проекта
:open_file_folder: mle-project-sprint-3-v001/    
├── :green_book: Instructions.md (этот файл)    
├── :green_book: Monitoring.md (файл с описанием дашборда и метрик)    
├── :green_book: README.md (содержание проекта: описание этапов и выводы)    
├── :page_facing_up: requirements.txt (библиотеки для начального этапа)    
└── :file_folder: services/    
    ├── :file_folder: app/ (папка с кодом микросервиса)    
    │   ├── :page_facing_up: app_stage_3.py (код для этапа 3)    
    │   ├── :page_facing_up: app_stage_4.py (код для этапа 4)    
    │   ├── :page_facing_up: app.py (код для этапов 1, 2)    
    │   ├── :page_facing_up: fastapi_handler.py (класс-обработчик)    
    │   ├── :page_facing_up: load_test.py (скрипт для имитации трафика)    
    │   └── :page_facing_up: tests.py (тестирование микросервиса)    
    ├── :whale2: docker-compose-stage-3-4.yaml (настройки для Docker)    
    ├── :whale2: docker-compose.yaml    
    ├── :whale2: Dockerfile    
    ├── :file_folder: grafana/ (сервис для дашборда)    
    │   └── :file_folder: provisioning/ (конфигурация сервиса)    
    │       ├── :file_folder: dashboards/    
    │       │   ├── :page_facing_up: dashboard.json (сохраненный дашборд)    
    │       │   ├── :page_facing_up: dashboard.yaml (инициализация дашборда)    
    │       │   └── :chart_with_upwards_trend: screenshot.jpg (скриншот)    
    │       └── :file_folder: datasources/    
    │           └── :page_facing_up: datasource.yaml (настройка источника данных)    
    ├── :file_folder: models/ (папка с моделью)    
    │   ├── :page_facing_up: create_model.py (код для генерации модели)    
    │   ├── :page_facing_up: fitted_model.pkl (ML-модель)    
    │   ├── :page_facing_up: load_model.py (код для загрузки модели)    
    │   ├── :page_facing_up: loaded_model.pkl (загруженная ML-модель)    
    │   └── :page_facing_up: model_pipeline.ipynb (Jupyter Notebook c feature engineering pipeline)    
    ├── :file_folder: prometheus/ (сервис метрик)    
    │   └── :page_facing_up: prometheus.yml (конфигурация)    
    └── :page_facing_up: requirements.txt (библиотеки для сборки Docker)    


#### Этап 1. Написание FastAPI-микросервиса
- Программирование микросервиса, принимающего входные данные и выдающего предсказания модели в формате JSON, используя FastAPI и uvicorn
- Разработка класса-обработчика с функциями валидации входных данных, и возвращающего предсказания
- Заполнение файла requirements.txt
- Написание инструкции в Instructions.md для установки и запуска микросервиса с использованием виртуального окружения
- Добавление примера curl-запросов

Есть возможность использования двух вариантов модели:    
- генерация модели "из коробки" - скрипт create_model.py
- загрузка модели из предыдущего спринта (по сути это пайплайн обработки данных + оптимизированная модель), скрипт load_model.py
Приложена часть Jupyter Notebook с предыдущего проекта с генерацией фичей и обучением в файле [model_pipeline.ipynb](./services/models/model_pipeline.ipynb).    
Код класса-обработчика `FastApiHandler` был реализован в файле services/app/[fastapi_handler.py](./services/app/fastapi_handler.py).    
Метод `validate_params()` класса проверяет параметры модели на соответствие списку требуемых полей.     
Функция `gen_random_data()` генерирует случайные параметры модели.    
     
Код микросервиса реализован в services/app/[app.py](./services/app/app.py).
Определено три эндпоинта:    
/ - возвращение статуса "Alive"    
/predict - возврат значения предсказания модели для заданных входных параметров    
/random - возврат значения предсказания модели для случайных параметров    

По адресу http://127.0.0.1:8000/docs  - тестирование микросервиса в Swagger    
GET - ресурс "/" выдает "Alive"    
GET - ресурс "/random" выдает случайный score    
POST - ресурс "/predict" выдает score модели либо сообщения об ошибках в зависимости от входных данных:
 - {"Error": "Problem with parameters"}
 - {"Error": "Problem with request"}
 - 422 Error: Unprocessable Entity в случае ошибки в JSON и т.д.

Для проверки работы микросервиса создан скрипт sevices/app/[tests.py](./services/app/tests.py), тестирующий все эндпоинты и различные варианты ошибок входных параметров модели.


##### Результаты этапа 1
- Код микросервиса в app, модель в models, requirements.txt
- Instructions.md по запуску FastAPI-микросервиса
- Пример запроса Swagger UI на странице /docs

#### Этап 2. Контейнеризация микросервиса
- Настройка Dockerfile для сборки образа сервиса
- Настройка docker-compose.yaml для запуска сервиса в режиме Docker Compose
- Написание инструкции по запуску микросервиса с помощью команд docker image и docker container
- Написание инструкции по запуску микросервиса в режиме Docker Compose
- Добавление примера запроса, который микросервис обработает корректно

##### Результаты этапа 2
- Dockerfile для сборки образа FastAPI-микросервиса
- docker-compose.yaml для сборки контейнера микросервиса
- Instructions.md с командами `docker image`+`docker container` и `docker compose` с примером запроса, который микросервис обработает корректно

#### Этап 3. Запуск сервисов для системы мониторинга
- Добавление в `docker compose` описание сервиса Prometheus
- Добавление файла конфигурации [prometheus.yml](./services/prometheus/prometheus.yml), который  подключается в качестве тома в Docker
- Добавление в `docker compose` описание сервиса Grafana
- Добавьте в микросервис экспортёр с помощью `prometheus_fastapi_instrumentator`
- Написание инструкции по запуску микросервиса и системы мониторинга в режиме `docker compose`

На этом этапе в [docker-compose.yaml](./services/docker-compose.yaml) добавлено описание сервисов prometheus и grafana, 
файл с дополнением кода: services/[docker-compose-stage-3-4.yaml](./services/docker-compose-stage-3-4.yaml).    
Обновленный код микросервиса находится в файле services/app/[app_stage_3.py](./services/app/app_stage_3.py) с добавленным экспортёром с помощью `prometheus_fastapi_instrumentator`.
Создан файл конфигурации prometheus services/prometheus/prometheus.yml, предусмотрено его сохранение в томе Docker.
Создан каталог services/grafana/ (см. этап 4).
В файл services/.env помещены значения переменных `GRAFANA_USER`, `GRAFANA_PASS`

##### Результаты этапа 3
- Файл [docker-compose-stage-3-4.yaml](./services/docker-compose-stage-3-4.yaml), в котором теперь присутствует описание сервисов Prometheus и Grafana
- Заполненный [prometheus.yml](./services/prometheus/prometheus.yml)
- Файл с микросервисом [app-stage-3.py](./services/app/app_stage_3.py), в котором теперь присутствует запуск экспортёра с помощью `prometheus_fastapi_instrumentator`
- [Instructions.md](./Instructions.md) с указанием адресов микросервиса, Prometheus, Grafana

#### Этап 4. Построение дашборда для мониторинга
- Описание в файле [Monitoring.md](./Monitoring.md) используемых ML-метрик реального времени,  также инфраструктурного и прикладного уровней для мониторинга приложения
- Добавление метрик разного типа с помощью `prometheus_client` в FastAPI-приложение
- Напиcание скрипта для симуляции нагрузки на сервис [load_test.py](./services/app/load_test.py) и добавление инструкции по его запуску     
- Построение дашборда в Grafana
- Сохранение дашборда в JSON-файл с названием [dashboard.json](./services/grafana/provisioning/dashboards/dashboard.json) и его загрузка в Git
- Добавление скриншота дашборда и его описания в [Monitoring.md](./Monitoring.md)


В файле [Monitoring.md](./Monitoring.md) приведены набор и описание ML-метрики реального времени, используемые для мониторинга работы микросервиса.    
В файл с кодом микросервиса добавлены метрики prometheus_client Gauge, Histogram, обновления сохранены в services/app/[app_stage_4.py](./services/app/app_stage_4.py).    
В файле services/app/[load_test.py](./services/app/load_test.py) содержится код, имитирующий нагрузку на сервис, а именно обращения к ресурсам /metrics и /random с интервалами до 1 с.    
В код микросервиса добавлен код для имитации ошибок с заданной вероятностью, а также задержки до 1 с:
```
raise HTTPException(status_code=500, detail="Random failure for testing purposes")
time.sleep(random.random())
```
Для гарантии сохранения информации при каждом перестроении контейнера, информация об источнике данных prometheus и дашборде сохраняется средствами Grafana provisioning. Для этого созданы файлы в соответствующих каталогах: /services/grafana/provisioning/dashboards/[dashboard.yaml](./services/grafana/provisioning/dashboards/dashboard.yaml) и /services/garfana/provisioning/datasources/[datasource.yaml](./services/grafana/provisioning/datasources/datasource.yaml).    
Сохраненный дашборд находится в файле /services/garfana/provisioning/dashboards/[dashboard.json](./services/grafana/provisioning/dashboards/dashboard.json).    
Скриншот дашборда находится в файле /services/garfana/provisioning/dashboards/[screenshot.jpg](./services/grafana/provisioning/dashboards/screenshot.jpg).

Описание дашборда см. в [Monitoring.md](./Monitoring.md).

##### Результаты этапа 4
- Описание в файле Monitoring.md выбранных метрик и обоснование их выбора
- Инструкция в файле Instructions.md запуску скрипта, симулирующего нагрузку на сервис
- Описание дашборда в файле Monitoring.md
- Файл с дашбордом dashboard.json

### Итоги проекта и общие выводы
- Была поставлена задача разработки онлайн-сервиса для предсказания модели цены на недвижимось    
- Были определены основные характеристики микросервиса: кроссплатформенность, мониторинг, вывод в продакшн
- Основной код - Python, основной формат конфигураций - YAML
- Для достижения кроссплатформенности использованы инструменты FastAPI и Docker
- Системы мониторинга выстроена с помощью Prometheus и Grafana
- Выбранные инструменты и платформы позволяют выполнить поставленные задачи без избыточного кода, настройки сервисов задаются в простых переносимых форматах
- Проведены валидационные тесты работы микросервиса
- Настроенный мониторинг позволяет отслеживать основные метрики на прикладном и инфраструктурном уровнях
- Дальнейшее усовершенствование микросервиса зависит от масштабов и параметров ввода сервиса в продакшн, а также от аппаратных ресурсов, которые все вместе, возможно, потребуют донастройки используемых инструментов, но на базе данного проекта это является выполнимой рабочей задачей