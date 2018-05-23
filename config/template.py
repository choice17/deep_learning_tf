import tensorflow as tf 
from functools import reduce
import datetime
import sys
import os
sys.path.insert(0,os.path.dirname(os.getcwd()))
import layer.layers as layers
import numpy as np
import dataprocessing.utils as dutils

class TFgraph:

    def __init__(self):
        self.weight = {}
        self.biases = {}
        # learning rate
        # log step
        # saver
        # name
        # minibatch size
        # global parmas 
        self._params = {}
        self._params['mb_size'] = 32
        self._params['lr'] = {0: 0.1, 3000: 0.05, 5000: 0.01}
        self._params['log_step'] = 100
        self._params['log_idx'] = 0
        self._params['batch_idx'] = 0
        self._params['epoch_idx'] = 0
        self._params['idx_in_epoch'] = 0
        self._params['use_tb_summary'] = False
        self._params['use_tf_saver'] = False

        # embeds into graph only
        # graph parmas/variable to be implemented
        self._graph ={}
        self._graph['cost'] = None
        self._graph['acc']  = None
        self._graph['lr'] = None
        self._graph['train_step'] = None
        self._graph['train_loss'] = None
        self._graph['valid_loss'] = None
        self._graph['train_acc'] = None
        self._graph['valid_acc'] = None
        # implement in config model
        self._graph['lr_tf'] = None
        self._graph['x_data'] = None
        self._graph['y_data'] = None
        self._graph['perm_array'] = np.array([])


    def set_params(opts):
        for idx in range(len(opts))[::2]:
            if opts[idx] == 'mb_size':
                self._params['mb_size'] = opts[idx+1]
            elif opts[idx] == 'lr':
                self._params['lr'] = opts[idx+1]
            elif opts[idx] == 'data_size':
                self._params['data_size'] = opts[idx+1]


    def create_graph(self):
        pass

    def cost(self,y_data,z_pred):
        pass

    def nextbatch(self):
        self._params['batch_idx'] += 1
        self._params['idx_in_epoch'] += self._params['mb_size']

        if self._params['idx_in_epoch'] >= self._params['data_size']['train_data']:
            self._params['batch_idx'] = 0
            self._params['idx_in_epoch'] = 0

        self.train_data =[]
        
        pass


    def generate_image(self,img):
        return dutils.generate_images(img)

    def summary_variable(self, var, var_name):
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def attach_summary(self, sess):
        
        # create summary tensors for tensorboard
        self.use_tb_summary = True

        for name in self.weights.keys():
            self.summary_variable(self.weights[name],name)

        for name in self.biases.keys():
            self.summary_variable(self.biases[name], name)
        
        tf.summary.scalar('graph_cross_entropy', self._graph['cost'])
        tf.summary.scalar('graph_accuracy', self._graph['acc'])

        # merge all summaries for tensorboard
        self.merged = tf.summary.merge_all()

        # initialize summary writer 
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(os.getcwd(), 'logs', (self.name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)

    def attach_saver(self):
        # initialize tensorflow saver
        self.use_tf_saver = True
        self.saver_tf = tf.train.Saver()

    def train_graph(self, sess, x_train, y_train, x_valid, y_valid, n_epoch = 1, 
                    train_on_augmented_data = False):


        # train on original or augmented data
        self.train_on_augmented_data = train_on_augmented_data
        
        # training and validation data
        self._graph['x_train'] = x_train
        self._graph['y_train'] = y_train
        self._graph['x_valid'] = x_valid
        self._graph['y_valid'] = y_valid
        
        # use augmented data
        if self.train_on_augmented_data:
            print('generate new set of images')
            self._graph['x_train_aug'] = dutils.normalize_data(dutils.generate_images(self._graph['x_train']))
            self._graph['y_train_aug'] = self._graph['y_train']
        
        # parameters
        print(self._params)
        mb_per_epoch = self._graph['x_train'].shape[0]/self._params['mb_size']
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        
        # start timer
        start = datetime.datetime.now();
        print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),': start training')
        print('learnrate = ',self._graph['lr'],', n_epoch = ', n_epoch,
              ', mb_size = ', self._params['mb_size'])
        # looping over mini batches
        for i in range(int(n_epoch*mb_per_epoch)+1):

            # adapt learn_rate
            #self._params['batch_idx'] = 0
            #self._params['idx_in_epoch'] = 0
            if i in self._params['lr'].keys():
                self._graph['lr'] = self._params['lr'][i]
                #deprecated self._graph['lr'] = int(self._params['idx_in_epoch'] // self.learn_rate_step_size)
                print(datetime.datetime.now()-start,': set learn rate to %.6f'%self._graph['lr'])            
            
            # get new batch
            x_batch, y_batch = self.next_mini_batch() 
            #print('x data - shape %s, y data - shape %s, graph lr : %s' % (x_batch.shape,y_batch.shape,self._graph['lr']))
            # run the graph
            sess.run(self._graph['train_step'], feed_dict={self._graph['x_data']: x_batch, 
                                                    self._graph['y_data']: y_batch,                                                      
                                                    self._graph['lr_tf']: self._graph['lr']})
             
            
            # store losses and accuracies
            if i%int(self._params['log_step']*mb_per_epoch) == 0 or i == int(n_epoch*mb_per_epoch):
             
                self._params['log_idx'] += 1 # for logging the results
                
                feed_dict_train = {
                    self._graph['x_data']: self._graph['x_train'][self._graph['perm_array'][:len(self._graph['x_valid'])]], 
                    self._graph['y_data']: self._graph['y_train'][self._graph['perm_array'][:len(self._graph['y_valid'])]]
                    }
                
                feed_dict_valid = {self._graph['x_data']: self._graph['x_valid'], 
                                   self._graph['y_data']: self._graph['y_valid'] 
                                   }
                
                # summary for tensorboard
                if self._params['use_tb_summary']:
                    train_summary = sess.run(self.merged, feed_dict = feed_dict_train)
                    valid_summary = sess.run(self.merged, feed_dict = feed_dict_valid)
                    self.train_writer.add_summary(train_summary, self.n_log_step)
                    self.valid_writer.add_summary(valid_summary, self.n_log_step)
                
                train_loss.append(sess.run(self._graph['cost'],
                                           feed_dict = feed_dict_train))

                train_acc.append(self._graph['acc'].eval(session = sess, 
                                                       feed_dict = feed_dict_train))
                
                valid_loss.append(sess.run(self._graph['cost'],
                                           feed_dict = feed_dict_valid))

                valid_acc.append(self._graph['acc'].eval(session = sess, 
                                                       feed_dict = feed_dict_valid))

                print('%.2f epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(
                    self._params['epoch_idx'], train_loss[-1], valid_loss[-1],
                    train_acc[-1], valid_acc[-1]))
     
        # concatenate losses and accuracies and assign to tensor variables
        tl_c = np.concatenate([self._graph['train_loss'].eval(session=sess), train_loss], axis = 0)
        vl_c = np.concatenate([self._graph['valid_loss'].eval(session=sess), valid_loss], axis = 0)
        ta_c = np.concatenate([self._graph['train_acc'].eval(session=sess), train_acc], axis = 0)
        va_c = np.concatenate([self._graph['valid_acc'].eval(session=sess), valid_acc], axis = 0)
   
        sess.run(tf.assign(self._graph['train_loss'], tl_c, validate_shape = False))
        sess.run(tf.assign(self._graph['valid_loss'], vl_c , validate_shape = False))
        sess.run(tf.assign(self._graph['train_acc'], ta_c , validate_shape = False))
        sess.run(tf.assign(self._graph['valid_acc'], va_c , validate_shape = False))
        
        print('running time for training: ', datetime.datetime.now() - start)
        return None



    def save_model(self, sess):
        
        # tf saver
        if self.use_tf_saver:
            #filepath = os.path.join(os.getcwd(), 'logs' , self.nn_name)
            filepath = os.path.join(os.getcwd(), self.nn_name)
            self.saver_tf.save(sess, filepath)
        
        # tb summary
        if self.use_tb_summary:
            self.train_writer.close()
            self.valid_writer.close()
        
        return None

    def forward(self, sess, x_data):
        return self

    def load_tensors(self, graph):
        pass

    def get_loss(self, sess):
        train_loss = self.train_loss_tf.eval(session = sess)
        valid_loss = self.valid_loss_tf.eval(session = sess)
        return train_loss, valid_loss 

    def get_accuracy(self, sess):
        train_acc = self.train_acc_tf.eval(session = sess)
        valid_acc = self.valid_acc_tf.eval(session = sess)
        return train_acc, valid_acc 

    def get_weights(self, sess):
        pass
    def get_biases(self, sess):
        pass
    def load_session_from_file(self, filename):
        pass
    def weights_count(self):
        
        wnum = lambda p: reduce((lambda x,y: x*y),p)
        weights = [w.get_shape().as_list() for w in self.weights.values()]
        biases = [b.get_shape().as_list() for b in self.biases.values()]
        return np.sum(list(map(wnum,list(weights)))), np.sum(list(biases))

    def next_mini_batch(self):

        start = self._params['idx_in_epoch']
        self._params['idx_in_epoch'] += self._params['mb_size']
        self._params['epoch_idx'] += self._params['mb_size']/len(self._graph['x_train'])  
        
        # adapt length of permutation array
        if not len(self._graph['perm_array']) == len(self._graph['x_train']):
            self._graph['perm_array'] = np.arange(len(self._graph['x_train']))
        
        # shuffle once at the start of epoch
        if start == 0:
            np.random.shuffle(self._graph['perm_array'])

        # at the end of the epoch
        if self._params['idx_in_epoch'] > self._graph['x_train'].shape[0]:
            np.random.shuffle(self._graph['perm_array']) # shuffle data
            start = 0 # start next epoch
            self._params['idx_in_epoch'] = self._params['mb_size'] # set index to mini batch size
            
            if self.train_on_augmented_data:
                # use augmented data for the next epoch
                self._graph['x_train_aug'] = dutils.normalize_data(dutils.generate_images(self.x_train))
                self._graph['y_train_aug'] = self._graph['y_train']
                
        end = self._params['idx_in_epoch']
        
        if self.train_on_augmented_data:
            # use augmented data
            x_tr = self._graph['x_train_aug'][self._graph['perm_array'][start:end]]
            y_tr = self._graph['y_train_aug'][self._graph['perm_array'][start:end]]
        else:
            # use original data
            x_tr = self._graph['x_train'][self._graph['perm_array'][start:end]]
            y_tr = self._graph['y_train'][self._graph['perm_array'][start:end]]
        
        return x_tr, y_tr

class test:
    def __init__(self):
        print('testing')