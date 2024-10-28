## Инструкции по запуску микросервиса
#### Этап 1. Написание FastAPI-микросервиса

```
# апгрейд менеджера пакетов
python3 -m pip install --upgrade pip

# создание виртуального окружения в папке .mle-sprint3-venv
python3 -m venv .mle-sprint3-venv

# активация виртуального окружения
source .mle-sprint3-venv/bin/activate

# установка необходимых пакетов
pip install -r requirements.txt

# перед запуском создайте/проверьте наличие /.env файла 
# с параметрами подключения к object storage

# указание корневой папки для корректной работы импорта
export PYTHONPATH=$(pwd)

# генерация и обучение модели "из коробки" (не используется)
python services/models/create_model.py

# загрузка обученной модели (используется в микросервисе)
python services/models/load_model.py

# проверка кода микросервиса с возвратом тестового предсказания
python services/app/fastapi_handler.py

# запуск ASGI-сервера со значениями по умолчанию --host 127.0.0.1 --port 8000
uvicorn services.app.app:app

# в другом терминале:
# отправка тестового GET-запроса, получение ответа {"status":"Alive"}
curl "http://127.0.0.1:8000/"

# отправка тестового POST-запроса, получение ответа {"score": <...>}
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
         "floor": 1,
         "is_apartment": 0,
         "kitchen_area": 7.0,
         "living_area": 27.0,
         "rooms": 2,
         "total_area": 40.0,
         "building_id": 764,
         "build_year": 1936,
         "building_type_int": 1,
         "latitude": 55.74044418334961,
         "longitude": 37.52492141723633,
         "ceiling_height": 3.0,
         "flats_count": 63,
         "floors_total": 7,
         "has_elevator": 1
     }'

# отправка запроса для получение случайного предсказания {"score": <...>}
curl "http://127.0.0.1:8000/random" 

# тестирование микросервиса в Swagger    
http://127.0.0.1:8000/docs

# остановка ASGI-сервера в терминале, где он запущен, по Ctrl-C

# запуск тестов, получение OK в случае успеха
python services/app/tests.py
```

#### Этап 2. Контейнеризация микросервиса

```
# переход в каталог с Docker
cd services

# загрузка образа ОС с Python
docker pull python:3.10-slim

# создание контейнера с использованием файла Dockerfile
docker build -t my-fastapi-app:latest .

# и вывод списка образов для проверки
docker image ls

# запуск контейнера
docker run -d -e APP_MODULE=services.app.app:app -p 8000:8000 my-fastapi-app:latest

# вывод списка запущенных контейнеров
docker container ps

# проверка работы микросервиса
curl "http://127.0.0.1:8000/"
curl "http://127.0.0.1:8000/random"

# остановка всех запущенных контейнеров
docker stop $(docker ps -a -q)

# создайте/проверьте наличие файла services/.env
# с заданной переменной окружения APP_PORT
# создание и запуск контейнера в режиме docker compose в фоновом (detached) режиме
APP_MODULE=services.app.app:app docker compose up --build -d

# проверка работы микросервиса
curl "http://127.0.0.1:8000/"
curl "http://127.0.0.1:8000/random"

# остановка и удаление контейнера
docker compose down
```

#### Этап 3. Запуск сервисов для системы мониторинга

```
# сборка и запуск контейнера с конфигурацией для этапа 3 в фоновом режиме
APP_MODULE=services.app.app_stage_3:app \
    docker compose \
        -f docker-compose.yaml \
        -f docker-compose-stage-3-4.yaml \
    up --build -d

# проверка работы сервисов:    
# при необходимости (например, в VS Code) настройте перенаправление портов
# FastAPI-микросервис: 
http://127.0.0.1:8000/    

#Prometheus: 
http://127.0.0.1:9090/targets    

#Grafana: 
http://127.0.0.1:3000    

# остановка и удаление контейнера
docker compose -f docker-compose.yaml -f docker-compose-stage-3-4.yaml down
```

#### Этап 4. Построение дашборда для мониторинга

```
# сборка и запуск контейнера с конфигурацией для этапа 4 в фоновом режиме
APP_MODULE=services.app.app_stage_4:app \
    docker compose \
        -f docker-compose.yaml \
        -f docker-compose-stage-3-4.yaml \
    up --build -d

# запуск скрипта с имитацией нагрузки (1000 запросов)
python app/load_test.py

# Тестирование Grafana, Логин, пароль, Dashboards → My dashboard
http://127.0.0.1:3000    

# Прервать работу скрипта load_test.py можно по Ctrl-C

# окончание работы с сервисом:
# остановка и удаление контейнера
docker compose -f docker-compose.yaml -f docker-compose-stage-3-4.yaml down

# проверка дискового пространства
docker system df

# очистка всех неиспользуемых ресурсов
docker system prune -f
```