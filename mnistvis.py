#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:36:42 2018

@author: sghos003
"""

from keras.models import load_model
model = load_model('mnist.h5')
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (3, 3)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'preds')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx =  0
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    
    filter_idx =  1
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx =  2
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx =  3
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx =  4
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx =  5
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx = 6
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx =  7
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    filter_idx =  8
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    
    filter_idx =  9
for tv_weight in np.arange(1):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])
    
#img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
#plt.imshow(img[..., 0])
#img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.))
#plt.imshow(img[..., 0])
#
#
#img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), verbose=True)
#plt.imshow(img[..., 0])
#img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), 
#                           tv_weight=0., lp_norm_weight=0., verbose=True)
#plt.imshow(img[..., 0])





#for output_idx in np.arange(10):
#    # Lets turn off verbose output this time to avoid clutter and just see the output.
#    img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))
#    plt.figure()
#    plt.title('Networks perception of {}'.format(output_idx))
#    plt.imshow(img[..., 0])