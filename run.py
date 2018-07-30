from nn_graph import MNIST
from dataprocessing import MNIST_Loader
import sklearn.model_selection
import datetime
import tensorflow as tf
import numpy as np

def main():
    train_file = 'data/MNIST/train.csv'
    test_file = 'data/MNIST/test.csv'
    mnist_loader = MNIST_Loader(train_file, test_file)
    x_train_valid, y_train_valid, _ = mnist_loader.get_data()

    # cross validations
    cv_num = 10 # cross validations default = 20 => 5% validation set
    nn_name = list(map(str,np.arange(10)))
    kfold = sklearn.model_selection.KFold(cv_num, shuffle=True, random_state=123)
    
    for i,(train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
        if i > 0:
            continue
        # start timer
        start = datetime.datetime.now()
        
        # train and validation data of original images
        x_train = x_train_valid[train_index]
        y_train = y_train_valid[train_index]
        x_valid = x_train_valid[valid_index]
        y_valid = y_train_valid[valid_index]
        
        # create neural network graph
        
        
        mnist = MNIST(mnist_name = nn_name[i]) # instance of nn_class
        mnist.create_graph() # create graph
        mnist.attach_saver() # attach saver tensors

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
    
   
   

    
    
if __name__ == '__main__':
    main()