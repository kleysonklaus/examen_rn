import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import skimage
from dnn_utils_v2 import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


dimesion = [12288,..., 1] 

def modelo(X, Y, dimesion, learning_rate = ..., num_iterations = 1800, print_cost=False):#lr was 0.009
   
    np.random.seed(1)
    costs = []                         
    parameters = initialize_parameters_deep(layers_dims)
   
    for i in range(0, num_iterations):
      
        AL, caches = L_model_forward(X, parameters)
      
        cost = compute_cost(AL, Y)
      
        grads = L_model_backward(AL, Y, caches)
      
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
     
        if print_cost and i % 100 == 0:
            print ("costo %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    
    return parameters

parameters = modelo(train_x, train_y, dimesion, num_iterations = ..., print_cost = True)

print("acuraccy train:")
pred_train = predict(train_x, train_y, parameters)

print("acuraccy test:")
pred_test = predict(test_x, test_y, parameters)




