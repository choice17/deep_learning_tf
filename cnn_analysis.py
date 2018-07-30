from nn_graph import MNIST
from dataprocessing import MNIST_Loader
import sklearn.model_selection
import datetime
import tensorflow as tf
import numpy as np

def main():
           
        mnist = MNIST(mnist_name = '0') # instance of nn_class
        mnist.create_graph() # create graph
        #mnist.attach_saver() # attach saver tensors

        mnist.load_session_from_file('0') 
        '''
        with tf.Session() as sess:
        
            # attach summaries
            mnist.attach_summary(sess) 
            
            # variable initialization of the default graph
            sess.run(tf.global_variables_initializer()) 
        
            # training on original data
            mnist.train_graph(sess, x_train, y_train, x_valid, y_valid, n_epoch = 1.0)
            
            # training on augmented data
            mnist.train_graph(sess, x_train, y_train, x_valid, y_valid, n_epoch = 5.0,
                                train_on_augmented_data = True)

            # save tensors and summaries of model
            mnist.save_model(sess)
        '''
   
   

    
    
if __name__ == '__main__':
    main()