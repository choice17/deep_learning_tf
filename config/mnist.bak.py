import tensorflow as tf
import sys
import os
sys.path.insert(0,os.path.dirname(os.getcwd()))
import layer.layers as layers
import config.template as template
import numpy as np
#DNN = layers.DNN
class mnist(template.TFgraph):

    def __init__(self):        

        super(mnist,self).__init__()
        self.weights = {}
        self.biases = {}
        self.name = 'mnist'
        



    def create_graph(self,xsize=[None,28,28,1],ysize=[None,10]):
        """
        # alias  
        """              
        tp = tf.placeholder
        tn = tf.nn                
        tfsoftmax = tf.nn.softmax_cross_entropy_with_logits
        tfAdam = tf.train.AdamOptimizer
        self._graph['x_data'] = tp(dtype=tf.float32, shape=xsize, name='graph_x_data')
        self._graph['y_data'] = tp(dtype=tf.float32, shape=ysize, name='graph_y_data')
        # input : N,28,28,3
        """
        dnn = DNN()

        # layer1 : N,14,14,16
        x = dnn.conv2d(self.graph['x_data'], kernel=[3,3,1,16] ,strides=[1,1,1,1], padding='SAME', actu = 'relu', pooling = 'mapx', var_type = tf.float32 ,num=1)
        # layer2 : N,14,14,32
        x = dnn.conv2d(x, [3,3,16,32],[1,1,1,1], 'SAME','relu',None, var_type = tf.float32 ,num=2)

        # layer3 : N,7,7,64
        x = dnn.conv2d(x, [3,3,32,64],[1,1,1,1], 'SAME','relu','mapx', var_type = tf.float32 ,num=3)

        # layer4 : N,7,7,64
        x = dnn.conv2d(x, [1,1,64,32],[1,1,1,1], 'SAME','relu',None, var_type = tf.float32 ,num=4)

        # layer5 : N,1,1,10
        x = dnn.conv2d(x, [3,3,32,10],[1,1,1,1], 'SAME','relu','globalavg', var_type = tf.float32 ,num=5)
        """
        # output : N,1,1,10
        

        # 1.layer: convolution + max pooling

        
        self.weights['w1'] = tf.Variable( tf.truncated_normal([3,3,1,16], stddev=0.1))
        self.biases['b1'] = tf.Variable( tf.constant(0.1, shape=[16]))
        x = tf.nn.conv2d(self._graph['x_data'], self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b1']
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        self.weights['w2'] = tf.Variable( tf.truncated_normal([3,3,16,32], stddev=0.1))
        self.biases['b2'] = tf.Variable( tf.constant(0.1, shape=[32]))
        x = tf.nn.conv2d(x, self.weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b2']
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        self.weights['w3'] = tf.Variable( tf.truncated_normal([3,3,32,64], stddev=0.1))
        self.biases['b3'] = tf.Variable( tf.constant(0.1, shape=[64]))
        x = tf.nn.conv2d(x, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        x = tf.nn.relu(x)
        
        self.weights['w4'] = tf.Variable( tf.truncated_normal([5*5*64,10], stddev=0.1))
        self.biases['b4'] = tf.Variable( tf.constant(0.1, shape=[10]))
        x = tf.reshape(x, [-1,5*5*64]) # (.,1024)
        x = tf.nn.relu(tf.matmul(x,self.weights['w4']) + self.biases['b4'] ) # (.,1024)

        #self._params['dropkr'] = tf.placeholder(dtype=tf.float32)
        #x = tf.nn.dropout(x, self._params['dropkr'])

        # 5.layer: fully connected
        self.weights['w5'] = tf.Variable( tf.truncated_normal([10,10], stddev=0.1))
        self.biases['b5'] = tf.Variable( tf.constant(0.1, shape=[10]))
        x = tf.add(tf.matmul(x, self.weights['w5']),self.biases['b5'], name = 'z_pred_tf')# => (.,10)

        self._graph['pred'] = x

        # cost function
        self._graph['cost'] = self.cost(self._graph['y_data'],self._graph['pred'])
        
        self.optim_config(scene = 'cross_entropy')           

        # tensors to save intermediate accuracies and losses during training
        self.create_log_acc_loss()
                     
        # number of weights and biases
        num_weights, num_biases = self.weights_count()                

        print('num_weights =', num_weights)
        print('num_biases =', num_biases)
        
        return None  


    def cost(self,y_data,pred):
        cost_val =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_data, logits=pred), name = 'cross_entropy')
        return cost_val

    def create_log_acc_loss(self):

        self._graph['train_loss'] = tf.Variable(np.array([]), dtype=tf.float32, 
                                                     name='graph_train_loss', validate_shape = False)
        self._graph['valid_loss'] = tf.Variable(np.array([]), dtype=tf.float32, 
                                             name='graph_valid_loss', validate_shape = False)
        self._graph['train_acc'] = tf.Variable(np.array([]), dtype=tf.float32, 
                                            name='graph_train_acc', validate_shape = False)
        self._graph['valid_acc'] = tf.Variable(np.array([]), dtype=tf.float32, 
                                            name='graph_valid_acc', validate_shape = False)
    def optim_config(self,scene='cross_entropy'):
            
        tp = tf.placeholder
        tn = tf.nn                
        tfsoftmax = tf.nn.softmax_cross_entropy_with_logits
        tfAdam = tf.train.AdamOptimizer

        if scene == 'cross_entropy':                

            # optimisation function
            self._graph['lr_tf'] = tp(dtype=tf.float32)
            self._graph['train_step'] = tfAdam(self._graph['lr_tf']).minimize(
                    self._graph['cost'])

            # predicted probabilities in one-hot encoding
            self._graph['pred_prob'] = tn.softmax(self._graph['pred'], name='graph_pred_prob') 
            
            # tensor of correct predictions
            self._graph['pred_correct'] = tf.equal(tf.argmax(self._graph['pred_prob'], 1),
                tf.argmax(self._graph['y_data'], 1),
                name = 'graph_pred_correct')  
            
            # accuracy 
            self._graph['acc'] = tf.reduce_mean(tf.cast(self._graph['pred_correct'], dtype=tf.float32),
                                             name = 'graph_accuracy')


