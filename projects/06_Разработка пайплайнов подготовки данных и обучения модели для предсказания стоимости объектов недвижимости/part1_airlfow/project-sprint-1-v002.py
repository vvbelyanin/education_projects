# part1_airflow/project-sprint-1-v002.py

import pendulum
from airflow.decorators import dag, task
from steps.messages import send_telegram_success_message, send_telegram_failure_message

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    tags=["ETL", 'sprint-1', 'create_extract_transform_clean_load'],
    catchup=False,
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)
def clean_flat_dataset():
    
    import pandas as pd
    import numpy as np
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    
    @task()
    def create_table():
        from sqlalchemy import Table, Column, BigInteger, Float, Integer, MetaData, UniqueConstraint, inspect
        hook = PostgresHook('destination_db')
        db_engine = hook.get_sqlalchemy_engine()
        metadata = MetaData()
        clean_flats = Table('clean_flats', metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('floor', Integer),
            Column('is_apartment', Integer),
            Column('kitchen_area', Float),
            Column('living_area', Float),
            Column('rooms', Integer),
            Column('total_area', Float),
            Column('price', BigInteger),
            Column('building_id', Integer),
            Column('build_year', Integer),
            Column('building_type_int', Integer),
            Column('latitude', Float),
            Column('longitude', Float),
            Column('ceiling_height', Float),
            Column('flats_count', Integer),
            Column('floors_total', Integer),
            Column('has_elevator', Integer),
            Column('flat_id', Integer),
            UniqueConstraint('flat_id', name='unique_clean_flat_id_constraint')
        )
        metadata.drop_all(db_engine)
        metadata.create_all(db_engine)
    
    @task()
    def extract():
        hook = PostgresHook('destination_db')
        conn = hook.get_conn()
        data = pd.read_sql('select * from combined_flats', conn)
        conn.close()
        return data

    @task()
    def transform(data: pd.DataFrame):
        data = data.drop(['id', 'flat_id'], axis=1)

        data.loc[data['rooms'] > 5, 'rooms'] = 5
        data['building_type_int'].replace({5: 4}, inplace=True)

        df = data.query('kitchen_area !=0 and living_area!=0')
        living_area_ratio_mean = np.mean(df.living_area/df.total_area)
        kitchen_area_ratio_mean = np.mean(df.kitchen_area/df.total_area)
        data.loc[data['kitchen_area'] == 0.0, 'kitchen_area'] = data['total_area'] * kitchen_area_ratio_mean # ~0.17
        data.loc[data['living_area'] == 0.0, 'living_area'] = data['total_area'] * living_area_ratio_mean # ~0.59

        return data

    @task()
    def clean_outliers(data: pd.DataFrame):
        exclude_columns = (['longitude', 'latitude', 
                                        'building_id', 'build_year',
                                        'rooms', 'building_type_int',
                                        'is_apartment', 'has_elevator'])
        columns = [x for x in data.columns if x not in exclude_columns]
        quantile_value = 0.001
        for col in data[columns].columns:
            dq = data[col].quantile([quantile_value, 1 - quantile_value]).to_list()
            data = data[(dq[0] <= data[col]) & (data[col] <= dq[1])]
        
        # keep='first' предусмотрен по умолчанию, для ясности кода можно указать явно
        data = data.drop_duplicates(keep='first')
        data = data.reset_index(names='flat_id')

        return data


    @task()
    def load(data: pd.DataFrame):
        hook = PostgresHook('destination_db')
        hook.insert_rows(
            table= 'clean_flats',
            replace=True,
            target_fields=data.columns.tolist(),
            replace_index=['flat_id'],
            rows=data.values.tolist()
    )
    
    create_table()
    data = extract()
    transformed_data = transform(data)
    cleaned_data = clean_outliers(transformed_data)
    load(cleaned_data)

clean_flat_dataset()