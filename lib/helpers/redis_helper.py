from redis import StrictRedis


REDIS = StrictRedis(host='redis')

def write_weights_matrix_to_redis(weights_matrix):    
    l,w = weights_matrix.shape
    encoded_weights = weights_matrix.ravel().tostring()
    result = REDIS.set('weights_matrix', encoded_weights)
    return result, (l,w)

def read_weights_matrix_from_redis(l,w):
    from_redis_weights = REDIS.get('weights_matrix')
    from_redis_weights = (np
                      .fromstring(from_redis_weights,dtype='int64')
                      .reshape(l,w))
    
    return from_redis_weights