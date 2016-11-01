#!/usr/bin/env python
#coding=utf-8
###############################################
# File Name: DeconvNet2D.py
# Author: Liang Jiang
# mail: jiangliang0811@gmail.com
# Created Time: Sun 30 Oct 2016 09:52:15 PM CST
# Description: Code for Deconvnet based on keras
###############################################

import argparse
import numpy as np
import sys
import time
from PIL import Image
from matplotlib import pyplot as plt
from keras.layers import (
        Input,
        InputLayer,
        Flatten,
        Dense)
from keras.layers.convolutional import (
        Convolution2D,
        MaxPooling2D)
from keras.activations import *
from keras.models import Model, Sequential
from keras.applications import vgg16, imagenet_utils
import keras.backend as K


class DConvolution2D(object):
    def __init__(self, layer):
        self.layer = layer

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DConvolution2D
        nb_up_filter = W.shape[0]
        nb_up_row = W.shape[2]
        nb_up_col = W.shape[3]
        input = Input(shape = layer.input_shape[1:])
        output = Convolution2D(
                nb_filter = nb_up_filter, 
                nb_row = nb_up_row, 
                nb_col = nb_up_col, 
                border_mode = 'same',
                weights = [W, b]
                )(input)
        self.up_func = K.function([input, K.learning_phase()], output)

        # Flip W horizontally and vertically, 
        # and set down_func for DConvolution2D
        W = np.transpose(W, (1, 0, 2, 3))
        W = W[:, :, ::-1, ::-1]
        nb_down_filter = W.shape[0]
        nb_down_row = W.shape[2]
        nb_down_col = W.shape[3]
        b = np.zeros(nb_down_filter)
        input = Input(shape = layer.output_shape[1:])
        output = Convolution2D(
                nb_filter = nb_down_filter, 
                nb_row = nb_down_row, 
                nb_col = nb_down_col, 
                border_mode = 'same',
                weights = [W, b]
                )(input)
        self.down_func = K.function([input, K.learning_phase()], output)

    # function to compute Convolution output in forward pass
    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    # function to compute Deconvolution output in forward pass
    def down(self, data, learning_phase = 0):
        self.down_data= self.down_func([data, learning_phase])
        return self.down_data
    

class DDense(object):
    def __init__(self, layer):
        self.layer = layer
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]
        
        #Set up_func for DDense
        input = Input(shape = layer.input_shape[1:])
        output = DDensese(output_dim = layer.output_shape[1],
                weights = [W, b])(input)
        self.up_func = K.function([input, K.learning_phase()], output)
        
        #Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape = self.output_shape[1:])
        output = DDensese(
                output_dim = self.input_shape[1], 
                weights = flipped_weights)(input)
        self.down_func = K.function([input, K.learning_phase()], output)
    

    # function to compute dense output in forward pass
    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data
        
    # function to compute reverse dense output in forward pass
    def down(self, data, learning_phase = 0):
        # data = data - self.bias
        self.down_data = self.down_func([data, learning_phase])
        return self.down_data

class DPooling(object):
    def __init__(self, layer):
        self.layer = layer
        self.poolsize = layer.pool_size
        # self.poolsize = layer.poolsize
    
    def up(self, data, learning_phase = 0):
        [self.up_data, self.switch] = \
                self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data, learning_phase = 0):
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data
    
    # Compute pooled output and switch in forward pass, 
    # switch stores location of the maximum value in each poolsize * poolsize block
    def __max_pooling_with_switch(self, input, poolsize):
        switch = np.zeros(input.shape)
        out_shape = list(input.shape)
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        out_shape[2] = out_shape[2] / poolsize[0]
        out_shape[3] = out_shape[3] / poolsize[1]
        pooled = np.zeros(out_shape)
        
        for sample in range(input.shape[0]):
            for dim in range(input.shape[1]):
                for row in range(out_shape[2]):
                    for col in range(out_shape[3]):
                        patch = input[sample, 
                                dim, 
                                row * row_poolsize : (row + 1) * row_poolsize,
                                col * col_poolsize : (col + 1) * col_poolsize]
                        max_value = patch.max()
                        pooled[sample, dim, row, col] = max_value
                        max_col_index = patch.argmax(axis = 1)
                        max_cols = patch.max(axis = 1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample, 
                                dim, 
                                row * row_poolsize + max_row, 
                                col * col_poolsize + max_col]  = 1
        return [pooled, switch]
    
    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):
        tile = np.ones((switch.shape[2] / input.shape[2], 
            switch.shape[3] / input.shape[3]))
        # print('tile: ', tile)
        out = np.kron(input, tile)
        unpooled = out * switch
        return unpooled


class DActivation(object):
    def __init__(self, layer, linear = False):
        self.layer = layer
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape = layer.output_shape)

        output = self.activation(input)
        # According to the original paper, 
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function(
                [input, K.learning_phase()], output)
        self.down_func = K.function(
                [input, K.learning_phase()], output)

    # Compute activation in forward pass
    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    # Compute activation in backward pass
    def down(self, data, learning_phase = 0):
        self.down_data = self.down_func([data, learning_phase])
        return self.down_data
    
    
class DFlatten(object):
    def __init__(self, layer):
        self.layer = layer
        self.shape = layer.input_shape[1:]
        self.up_func = K.function(
                [layer.input, K.learning_phase()], layer.output)

    # Flatten 2D input into 1D output
    def up(self, data, learning_phase = 0):
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    # Reshape 1D input into 2D output
    def down(self, data, learning_phase = 0):
        # print(self.shape)
        # print(data.shape[1:])
        new_shape = [data.shape[0]] + list(self.shape)
        # print('new_shape: ', new_shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return self.down_data

class DInput(object):
    def __init__(self, layer):
        self.layer = layer
    
    # input and output of Inputl layer are the same
    def up(self, data, learning_phase = 0):
        self.up_data = data
        return self.up_data
    
    def down(self, data, learning_phase = 0):
        self.down_data = data
        return self.down_data
    
def visualize(model, data, layer_name, feature_to_visualize, visualize_mode):
    deconv_layers = []
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], Convolution2D):
            deconv_layers.append(DConvolution2D(model.layers[i]))
            deconv_layers.append(
                    DActivation(model.layers[i]))
        elif isinstance(model.layers[i], MaxPooling2D):
            deconv_layers.append(DPooling(model.layers[i]))
        elif isinstance(model.layers[i], Dense):
            deconv_layers.append(DDense(model.layers[i]))
            deconv_layers.append(
                    DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Flatten):
            deconv_layers.append(DFlatten(model.layers[i]))
        elif isinstance(model.layers[i], InputLayer):
            deconv_layers.append(DInput(model.layers[i]))
        else:
            print(model.layers[i].get_config())
        if layer_name == model.layers[i].name:
            break
    

    deconv_layers[0].up(data)
    
    for i in range(1, len(deconv_layers)):
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    output = deconv_layers[-1].up_data
    print('shape of output', output.shape)
    print('max of output: ', output.max())
    print('min of output: ', output.min())

    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:, feature_to_visualize, :, :]
    if 'max' == visualize_mode:
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp
    elif 'all' != visualize_mode:
        print('Illegal visualize mode')
        sys.exit()
    output = np.zeros_like(output)
    if 2 == output.ndim:
        output[:, feature_to_visualize] = feature_map
    else:
        output[:, feature_to_visualize, :, :] = feature_map

    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    
    deconv = deconv_layers[0].down_data
    deconv = deconv.squeeze()
    
    return deconv

    
def argparser():
    parser = argparse.ArgumentParser()
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    np.random.seed(1000)
    model = vgg16.VGG16(weights = 'imagenet', include_top = True)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    img = Image.open('./husky.jpg')
    img_array = np.array(img)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = img_array[np.newaxis, :]
    img_array = img_array.astype(np.float)
    img_array = imagenet_utils.preprocess_input(img_array)
    

    layer_name = 'block5_conv3'
    feature_to_visualize = 99
    visualize_mode = 'max'
    
    deconv = visualize(model, img_array, 
            layer_name, feature_to_visualize, visualize_mode)

    deconv = np.transpose(deconv, (1, 2, 0))
    print('shape of deconv: ', deconv.shape)
    print('max of deconv: ', deconv.max())
    print('min of deconv: ', deconv.min())
    # sys.exit()
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    deconv = deconv[:, :, ::-1]
    # deconv = deconv.astype(np.uint8)
    print('shape of deconv: ', deconv.shape)
    print('dtype of deconv: ', deconv.dtype)
    print('max of deconv: ', deconv.max())
    print('min of deconv: ', deconv.min())
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    img.save('husky_{}_{}_feature.png'.format(layer_name, feature_to_visualize))
    plt.imshow(deconv)
    plt.show()

if "__main__" == __name__:
    main()
