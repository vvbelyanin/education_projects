**Численные методы. Определение стоимости автомобилей**

Сервис по продаже автомобилей с пробегом разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости.    
    
Рабочий процесс и инструменты    
| # | Описание | Инструменты/Ключевые слова |
|:----:|:---------------------------|:-----------------------------------------------------------|
| 1 | Анализ датасета и предобработка данных | `pandas` `sklearn` |
| 2 | Подбор моделей для регрессии | `sklearn` `CatBoostRegressor` `XGBRegressor` `LGBMRegressor` |
| 3 | Поиск оптимальных гиперпараметров | `n_estimators` `learning_rate` `num_leaves` |
| 4 | Анализ заданных метрик модели | `RMSE` |

Описание и выполнение см. в [Jupyter NB](./workflow.ipynb)