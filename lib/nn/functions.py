from numpy import abs, arange, argmax, greater, maximum, mean, outer, sum, zeros
from numpy.linalg import norm
from numpy.random import randn

def loss_function(weight_matrix, X_batch, y_batch, delta=1.0, gamma=0.001, rtn_mean=True):
    these_scores = scores(weight_matrix, X_batch) 
    correct_scores = these_scores[arange(len(these_scores)), 
                                  y_batch]
    margins = maximum(0, these_scores.T - correct_scores.T + delta).T
    margins[arange(len(margins)), y_batch] = 0
    if rtn_mean:
        return mean(sum(margins,axis=1)  + gamma*norm(weight_matrix))
    else:
        return sum(margins,axis=1)  + gamma*norm(weight_matrix)  

def gradient_loss_function(weight_matrix, X_batch, y_batch, delta=1.0):
    these_scores = scores(weight_matrix, X_batch) 
    incorrect_indices = abs(y_batch - 1)
    correct_scores = these_scores[arange(len(these_scores)), 
                                  y_batch]
    grads_mult = greater( 
                      (these_scores.T - correct_scores.T + delta),
                      zeros(these_scores.shape).T).astype(int).T
    grads_mult[arange(len(grads_mult)), y_batch] = 0
    grads_mult[arange(len(grads_mult)), y_batch] = - sum(grads_mult, axis=1)
    grads = grads_mult.T.dot(X_batch) 
    return grads

def random_matrix(m,n):
    return randn(m, n) 

def scores(weight_matrix, batch):
    return weight_matrix.dot(batch.T).T

def predict(weight_matrix, X_batch):
    these_scores = scores(weight_matrix, X_batch)
    return argmax(these_scores, axis=1)

def measure_accuracy(weight_matrix, features, outcomes):
    return mean(predict(weight_matrix, features) == outcomes)
