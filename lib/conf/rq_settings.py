from os import environ

REDIS_URL = 'redis://'+environ['REDIS_PORT_6379_TCP_ADDR']+':6379'
