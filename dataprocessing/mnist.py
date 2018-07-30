import pandas as pd
import numpy as np
import utils
class MNIST_Loader(object):
    def __init__(self, train_data_file=None, test_data_file=None):
        
        self.train_data = 'data/MNIST/train.csv'
        self.test_data = 'data/MNIST/test.csv'
        if train_data_file is not None:
            self.train_data = train_data_file
            self.test_data = test_data_file
        self.img_w = 28
        self.img_h = 28

    def load(self, train_data_file, test_data_file):
        self.train_data = train_data_file
        self.test_data = test_data_file

    def get_data(self):
        self.read()
        x_data_valid, y_data_valid, x_test = self.extract_normalize_images()
        return x_data_valid, y_data_valid, x_test

    def read(self):
        self.data_df = pd.read_csv(self.train_data)
        self.test_df = pd.read_csv(self.test_data)
        print('')
        print(self.data_df.isnull().any().describe())

        # 10 different labels ranging from 0 to 9
        print()
        print('distinct labels ', self.data_df['label'].unique())

        # data are approximately balanced (less often occurs 5, most often 1)
        print()
        print(self.data_df['label'].value_counts())

    # extract and normalize images
    def extract_normalize_images(self):
        x_train_valid = self.data_df.iloc[:,1:].values.reshape(-1,self.img_h,self.img_w,1) # (42000,28,28,1) array
        x_train_valid = x_train_valid.astype(np.float) # convert from int64 to float32
        x_train_valid = utils.normalize_data(x_train_valid)
        
        x_test = self.test_df.iloc[:,0:].values.reshape(-1,self.img_h,self.img_w,1) # (28000,28,28,1) array
        x_test = x_test.astype(np.float)
        x_test = utils.normalize_data(x_test)  
        
        image_size = 784

        # extract image labels
        y_train_valid_labels = self.data_df.iloc[:,0].values # (42000,1) array
        labels_count = np.unique(y_train_valid_labels).shape[0]; # number of different labels = 10

        #plot some images and labels
        #plt.figure(figsize=(15,9))
        #for i in range(50):
        #    plt.subplot(5,10,1+i)
        #    plt.title(y_train_valid_labels[i])
        #    plt.imshow(x_train_valid[i].reshape(28,28), cmap=cm.binary)
        
        # labels in one hot representation
        y_train_valid = utils.dense_to_one_hot(y_train_valid_labels, labels_count).astype(np.uint8)
        return (x_train_valid, y_train_valid, x_test)
