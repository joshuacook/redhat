from time import sleep
from lib.helpers.database_helper import pull_and_shape_batch
from lib.helpers.redis_helper import get_best_loss, get_weights_matrix
from lib.helpers.redis_helper import get_training_count
from lib.helpers.redis_helper import set_best_loss, set_weights_matrix
from lib.helpers.redis_helper import incr_training_count, init_training_count
from lib.nn.functions import gradient_loss_function, loss_function
from lib.nn.functions import random_matrix, scores

def initialize_training_session():

    # initialize training count
    init_training_count()

    # initialize the loss to be arbitrarily high
    bestloss = int(1e10) 
    set_best_loss(bestloss)

    # initialize a random_weights matrix
    random_weights = random_matrix(2, 7326)
    set_weights_matrix(random_weights)

def prepare_plot_of_loss_function(length=750):
    import matplotlib.pyplot as plt
    import seaborn
    training_counts = []
    loss_values = []
    training_count = get_training_count()
    while training_count < length:
        training_counts.append(training_count)
        loss_values.append(get_best_loss())
        sleep(1)
        training_count = get_training_count()
    
    plt.plot(training_counts[1:], loss_values[1:])  

    return training_counts, loss_values
 
def train_via_random_search(n=100,offset=0, action_ids=None, gamma=0.001):    
    incr_training_count()
    batch_features, \
        batch_outcomes = pull_and_shape_batch(n=n,
                                              offset=offset*n,
                                              action_ids=action_ids)
    random_weights = random_matrix(2, 7326)
    loss = loss_function(random_weights, 
                         batch_features, 
                         batch_outcomes,
                         gamma=gamma)
    
    if loss < get_best_loss(): 
        set_best_loss(loss)
        set_weights_matrix(random_weights)  

def train_via_random_local_search(n=100, offset=0, action_ids=None, gamma=0.001):
    incr_training_count()
    step_size = 0.0001
    weight_matrix = get_weights_matrix()
    batch_features, \
        batch_outcomes = pull_and_shape_batch(n=n,
                                              offset=offset*n,
                                              action_ids=action_ids)
    weight_matrix_try = weight_matrix + random_matrix(2, 7326) * step_size
    loss = loss_function(weight_matrix_try,
                         batch_features, 
                         batch_outcomes,
                         gamma=gamma)
    
    if loss < get_best_loss(): 
        set_best_loss(loss)
        set_weights_matrix(weight_matrix_try)  

def train_via_gradient_descent(n=100, offset=0, action_ids=None, delta=1.0, gamma=0.001):
    training_count = incr_training_count()
    step_size = 0.0001
    weight_matrix = get_weights_matrix()
    batch_features, \
        batch_outcomes = pull_and_shape_batch(n=n,
                                              offset=offset*n,
                                              action_ids=action_ids)

    grad = gradient_loss_function(weight_matrix, 
                                  batch_features,
                                  batch_outcomes,
                                  delta=delta)   
    weight_matrix = weight_matrix + grad * step_size
    loss = loss_function(weight_matrix,
                         batch_features, 
                         batch_outcomes,
                         gamma=gamma)
    
    set_best_loss(loss)
    set_weights_matrix(weight_matrix)  