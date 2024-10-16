# part1_airflow/steps/messages.py

from airflow.providers.telegram.hooks.telegram import TelegramHook

def send_telegram_success_message(context):
    hook = TelegramHook(telegram_conn_id=None,
                        token='6716887876:AAHausJ_Vj3jUgkhSuImo3rm0-XJIw2VNXw',
                        chat_id='-4123911992')
    dag = context['task_instance_key_str']
    run_id = context['run_id']
    message = f'Исполнение DAG {dag} с id={run_id} прошло успешно!'
    hook.send_message({
        'chat_id': '-4123911992',
        'text': message
    }) 

def send_telegram_failure_message(context):
    hook = TelegramHook(telegram_conn_id=None,
                        token='6716887876:AAHausJ_Vj3jUgkhSuImo3rm0-XJIw2VNXw',
                        chat_id='-4123911992')
    dag = context['task_instance_key_str']
    run_id = context['run_id']
    message = f'Исполнение DAG {dag} с id={run_id} прошло неудачно.'
    hook.send_message({
        'chat_id': '-4123911992',
        'text': message
    })
