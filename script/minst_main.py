## train the neural network graph
"""
for training mnist data demo
"""
import sklearn
import tensorflow as tf

def main():
    nn_name = ['tmp']

    # cross validations
    cv_num = 3 # cross validations default = 20 => 5% validation set
    kfold = sklearn.model_selection.KFold(cv_num, shuffle=True, random_state=123)

    for i,(train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
        
        # start timer
        start = datetime.datetime.now();
        
        # train and validation data of original images
        x_train = x_train_valid[train_index]
        y_train = y_train_valid[train_index]
        x_valid = x_train_valid[valid_index]
        y_valid = y_train_valid[valid_index]
        
        # create neural network graph
        nn_graph = nn_class(nn_name = nn_name[i]) # instance of nn_class
        nn_graph.create_graph() # create graph
        nn_graph.attach_saver() # attach saver tensors
        
        # start tensorflow session
        with tf.Session() as sess:
            
            # attach summaries
            nn_graph.attach_summary(sess) 
            
            # variable initialization of the default graph
            sess.run(tf.global_variables_initializer()) 
        
            # training on original data
            nn_graph.train_graph(sess, x_train, y_train, x_valid, y_valid, n_epoch = 1.0)
            
            # training on augmented data
            nn_graph.train_graph(sess, x_train, y_train, x_valid, y_valid, n_epoch = 14.0,
                                train_on_augmented_data = True)

            # save tensors and summaries of model
            nn_graph.save_model(sess)
            
        # only one iteration
        if True:
            break;
            
        
    print('total running time for training: ', datetime.datetime.now() - start)