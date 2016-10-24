from numpy import fromstring
from os import environ
from redis import StrictRedis

REDIS = StrictRedis(host=environ['REDIS_PORT_6379_TCP_ADDR'])

def get_best_loss():
    return int(REDIS.get('best_loss'))/1e7

def get_training_count():
    return int(REDIS.get('training_count')) 

def get_weights_matrix():
    from_redis_weights = REDIS.get('weights_matrix')
    from_redis_weights = (fromstring(from_redis_weights)
                          .reshape(2,7326))
    
    return from_redis_weights

def init_training_count():
    return REDIS.set('training_count', 0)

def incr_training_count():
    return REDIS.incr('training_count')

def set_best_loss(best_loss):
    return REDIS.set('best_loss', int(round(best_loss,7)*1E7))

def set_weights_matrix(weights_matrix):    
    encoded_weights = weights_matrix.ravel().tostring()
    return REDIS.set('weights_matrix', encoded_weights)
