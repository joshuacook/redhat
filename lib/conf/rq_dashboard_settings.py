from os import environ

REDIS_URL = 'redis://'+environ['REDIS_PORT_6379_TCP_ADDR']+':6379'
RQ_POLL_INTERVAL = 2500  #: Web interface poll period for updates in ms
