import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linspace, maximum, sum, zeros 
from numpy.linalg import norm

def loss_function_i(correct_class,x,W,delta=1.0,gamma=0.1):
    scores = W.dot(x)
    correct_score = scores[correct_class]
    margins = maximum(0, scores - correct_score + delta)
    margins[correct_class] = 0
    return sum(margins) + gamma*norm(W)

def loss_function_in_a_direction(variable_weight, param, correct_class, x, W):
    delta_W = zeros(W.shape)
    delta_W[:,param] += int(variable_weight)*W[:,param]
    return loss_function_i(correct_class,x,W+delta_W)

def plot_loss_function_for_a_single_parameter(plot_axis,param,correct_class,x,W):
    dependent_vector = [loss_function_in_a_direction(variable_weight,
                                                     param,
                                                     correct_class,
                                                     x,W) 
                        for variable_weight in linspace(-20,20,200)]
    plot_axis.plot(linspace(-20,20,200),dependent_vector)
    
def render_all_plots_1d(correct_class,x,W):
    figure, axes = plt.subplots(1,7, sharex=True, sharey=True, figsize=(20,6))
    for param, axis in zip(range(7),axes):
        plot_loss_function_for_a_single_parameter(axis,param,correct_class,x,W)    

def loss_function_in_two_directions(a,p_1,b,p_2,correct_class,x,W):
    delta_W = zeros(W.shape)
    delta_W[:,p_1] += int(a)*W[:,p_1]
    delta_W[:,p_2] += int(b)*W[:,p_2]
    return loss_function_i(correct_class,x,W+delta_W)

def build_heat_map_for_two_parameters(p_1,p_2,min_val,max_val,nx,correct,x,W):
    X = linspace(min_val, max_val, nx)
    Y = linspace(min_val, max_val, nx)
    return [[loss_function_in_two_directions(xv,p_1,yv,p_2,correct,x,W)
             for xv in X]
            for yv in Y]

def plot_heatmap(plot_axis,p_1,p_2,correct,x,W):
    this_heat_map = build_heat_map_for_two_parameters(p_1,p_2,-100,100,200,correct,x,W)
    sns.heatmap(this_heat_map,
                cmap='autumn', 
                cbar=False, 
                xticklabels=False, 
                yticklabels=False, 
                vmin=0,vmax=5,
                ax=plot_axis)
    
def render_all_plots_2d(correct_class,x,W):
    plt.figure(figsize=(18,21))
    figure, axes = plt.subplots(7,6, sharex=True, sharey=True, figsize=(18,21))
    for p_1, axes_i in zip(range(7),axes):        
        p_2s = [i for i in range(7)]
        p_2s.remove(p_1) 
        for p_2, axis in zip(p_2s, axes_i):
            plot_heatmap(axis,p_1,p_2,correct_class,x,W)
            axis.set_title("p {} v p {}".format(str(p_1),str(p_2)))

