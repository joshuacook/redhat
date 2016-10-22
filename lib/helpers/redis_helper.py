from numpy import fromstring
from os import environ
from redis import StrictRedis

REDIS = StrictRedis(host=environ['REDIS_PORT_6379_TCP_ADDR'])

def read_best_loss():
    return int(REDIS.get('best_loss'))/1e7
    return fromstring(best_loss, dtype='int64') 

def read_weights_matrix(l,w):
    from_redis_weights = REDIS.get('weights_matrix')
    from_redis_weights = (fromstring(from_redis_weights, dtype='int64')
                          .reshape(l,w))
    
    return from_redis_weights

def write_best_loss(best_loss):
    return REDIS.set('best_loss', int(round(best_loss,7)*1E7))

def write_weights_matrix(weights_matrix):    
    l,w = weights_matrix.shape
    encoded_weights = weights_matrix.ravel().tostring()
    result = REDIS.set('weights_matrix', encoded_weights)
    return result, (l,w)

