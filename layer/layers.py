import tensorflow as tf
import sys
import os

# weight initialization
def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

# bias initialization
def bias_variable(shape, name = None):
    initial = tf.constant(0.1, shape=shape) #  positive bias
    return tf.Variable(initial, name = name)

# 2D convolution
def conv2d(x, W, name = None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

# max pooling
def max_pool_2x2(x, name = None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name = name)

class DNN:
    '''
    A self define combination of tf.nn module
    mainly integrate 
    layer, actu class
    use to define higher abstract level of dnn architecture

    '''
    def __init__(self):
        self.weights = {}
        self.biases = {}
        

    def conv2d(self,data,kernel ,strides=[1,1,1,1], padding='SAME', actu = 'relu',
                batchnorm=False, pooling = None, 
                var_type = None ,num=None):
        
        # Every weights has an unique number
        # num == None
        x = data
        k = kernel
        # init parameter
        kw,kh,kc,kn = k
        w_name, b_name, conv_name = 'w%d' % num, 'b%d' % num, 'conv%d' % num  
        a_name, bn_name, pool_name = 'a%d' % num, 'bn%d' % num, 'p%d' % num
        s = strides
        p = padding
        if var_type is None:
            var_type = tf.float32

        # get variables
        w_init = tf.truncated_normal(k, stddev=0.1)
        w = tf.Variable(w_init, dtype = var_type, name = w_name)
        b_init = tf.constant(0.1, shape=[kn])
        b = tf.Variable(b_init, dtype = var_type, name = b_name)        
        wbn, bbn = None, None

        # conv here
        x = Layer('conv2d', data=x,opts=['weights',w,'strides',s,'padding',p],name=conv_name) + b

        # activation here
        x = Layer('relu', data=x,opts=[],name=a_name)

        # batchnorm here
        if batchnorm == True:
            x = Layer('batchnorm', data=x, name = bn_name)

        if pooling is not None:
            x = Layer(pooling, data=x, opts=['strides',s,'padding',p],name=pool_name)

        self.weights[w_name] = w
        self.biases[b_name] = b
        return x
        


def Layer(layer_type=None,data=None,opts=None,name=None):

    
    if layer_type == 'conv2d':
        '''def conv2d(data=x,weights=W,strides=s,padding = 'SAME',name=None)'''
        s, p = [1,1,1,1], 'SAME'
        
        for idx in range(len(opts)):
            if opts[idx] == 'weights':
                w = opts[idx+1]
            elif opts[idx] == 'strides':
                s = opts[idx+1]
            elif opts[idx] == 'padding':
                p = opts[idx+1]     

        return layer.conv2d(data,w,strides=s,padding = p,name=name)

    elif layer_type == 'batchnorm':
        pass

    elif layer_type == 'dropout':
        pass
    elif layer_type == 'maxp':
        #def maxp(data=x,kszie=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=None):
        ksize, s, p = [1,2,2,1], [1,2,2,1], 'SAME'
        
        for idx in range(len(opts)):
            if opts[idx] == 'kszie':
                ksize = opts[idx+1]
            elif opts[idx] == 'strides':
                s = opts[idx+1]
            elif opts[idx] == 'padding':
                p = opts[idx+1] 

        return layer.maxp(data, ksize=ksize, strides=s, )

    elif layer_type == 'avgp':
        pass
    elif layer_type == 'globalavg':
        pass

    elif layer_type == 'relu':
        '''def relu(x,name=None)'''
        return actu.relu(data,name=name)

    elif layer_type == 'nrelu':
        pass

    elif layer_type == 'elu':
        pass

    elif layer_type == 'leaky':
        pass
    elif layer_type == 'sigmoid':
        pass
    elif layer_type == 'tanh':
        pass
    
    return None


#def conv2d(data,weights,strides,padding = 'SAME',name=None):
#    return tf.nn.conv2d(data, weights, strides=strides, padding=padding, name = name)

def batchnorm(data,name=None):
    return tf.nn.batchnorm(data,name=name)

def dropout(data,drop_prob=0.3,name=None):
    return tf.nn.dropout(data,dropout=drop_prob,name=name)      

def maxp(data,kszie=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=None):
    return tf.nn.max_pool(data,ksize= kszie,srides = strides,padding = padding,name = name)

def avgp():
    pass

def globalavg():
    pass

def relu(x,name=None):
    return tf.maximum(x,0,name)

def nrelu(x,name=None):
    return tf.minimum(x,0,name)

def elu(x,a=1,name=None):
    return tf.nn.elu(x,a,name)

def leaky(x,a=0.9,name=None):
    return tf.nn.leaky(x,a,name)

def sigmoid(x,name=None):
    return tf.nn.sigmoid(x,name)

def tanh(x,name=None):
    return tf.nn.tanh(x,name)

     