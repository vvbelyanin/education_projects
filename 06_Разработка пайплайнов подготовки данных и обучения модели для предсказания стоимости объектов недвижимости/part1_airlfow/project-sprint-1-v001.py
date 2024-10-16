# part1_airflow/project-sprint-1-v001.py

import pendulum
from airflow.decorators import dag, task
from steps.messages import send_telegram_success_message, send_telegram_failure_message

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message,
    tags=["ETL", "sprint-1", "create_extract_transform_load"]

)
def prepare_flat_dataset():
    import pandas as pd
    import numpy as np
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    
    @task()
    def create_table():
        from sqlalchemy import MetaData, Table, Column, Integer, Float, UniqueConstraint
        from sqlalchemy import inspect
        hook = PostgresHook('destination_db')
        db_conn = hook.get_sqlalchemy_engine()
        metadata = MetaData()
        combined_flats = Table(
            'combined_flats',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('floor', Integer),
            Column('is_apartment', Integer),
            Column('kitchen_area', Float),
            Column('living_area', Float),
            Column('rooms', Integer),
            Column('total_area', Float),
            Column('price', Float),
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
            UniqueConstraint('flat_id', name='unique_flat_id_constraint')
            )
        if not inspect(db_conn).has_table(combined_flats.name):
            metadata.create_all(db_conn)

    @task()
    def extract(**kwargs):

        hook = PostgresHook('destination_db')
        conn = hook.get_conn()
        sql = f'''
            select * 
            from flats 
            join buildings ON buildings.id=flats.building_id
            '''
        data = pd.read_sql(sql, conn)

        data = data.drop(['id','studio'], axis=1)
        conn.close()

        return data

    @task()
    def transform(data: pd.DataFrame):
        data[['has_elevator','is_apartment']] = data[['has_elevator','is_apartment']].astype('int')
        data['price'] = data['price'].astype('float')
        data = data.reset_index(names='flat_id')
        return data

    @task()
    def load(data: pd.DataFrame):
        hook = PostgresHook('destination_db')
        hook.insert_rows(
            table="combined_flats",
            replace=True,
            target_fields=data.columns.tolist(),
            replace_index=['flat_id'],
            rows=data.values.tolist()
    )

    create_table()
    data = extract()
    transformed_data = transform(data)
    load(transformed_data)

prepare_flat_dataset()