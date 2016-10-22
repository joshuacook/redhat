from numpy import arange, maximum, mean, sum
from numpy.linalg import norm
from numpy.random import randn

def loss_function(weight_matrix, X_batch, y_batch, delta=1.0, gamma=0.1):
    these_scores = scores(weight_matrix, X_batch) 
    correct_scores = these_scores[arange(len(these_scores)), 
                                  y_batch]
    margins = maximum(0, these_scores.T - correct_scores.T + delta).T
    margins[arange(len(margins)), y_batch] = 0
    return mean(sum(margins,axis=1)/2  + gamma*norm(weight_matrix))

def random_matrix(m,n):
    return randn(m, n) * 0.0001

def scores(weight_matrix, batch):
    return weight_matrix.dot(batch.T).T
