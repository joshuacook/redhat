from lib.helpers.database_helper import pull_and_shape_batch
from lib.nn.functions import loss_function, random_matrix, scores

def train_with_random_on_n_rows_from_offset(offset,n=100):    
    batch_features, batch_outcomes = pull_and_shape_batch(n,offset*n)
    print("pulled batch")
    random_weights = random_matrix(2, 7326)
    print("generated random weight")
    loss = loss_function(random_weights, batch_features, batch_outcomes)
    print("loss: {}".format(loss))
    
    if loss < read_best_loss(): 
        write_best_loss(loss)
        write_weights_matrix(random_weights)  
