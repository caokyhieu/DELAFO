import numpy as np
import keras
import h5py
from tensorflow.python.keras.utils.data_utils import Sequence
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,path ='',batch_size=32,n_fold=10,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.data = h5py.File(path, 'r')
        self.X = self.data['X']
        self.length = self.X.shape[0]
        self.y = self.data['y']
        self.date_X = self.data['date_X']
        self.date_y = self.data['date_y']
        self.tickers = self.data['ticker'][:]
        self.shuffle = shuffle
        # self.tickers = [i.decode('UTF-8') for i in self.tickers]
        self.n_fold = n_fold + 1
        self.current_fold = 1
        # self.indexes = np.arange(self.length)
        self.on_epoch_end()
        
    
    def update_current_fold(self,n):
        self.current_fold = n 

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def finish_train(self):
        self.data.close()

    def gen_data(self,list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        X = self.X[list_IDs_temp]
        y = self.y[list_IDs_temp]

        # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')

        #     # Store class
        #     y[i] = self.labels[ID]

        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes.sort()
        # Generate data
        X, y = self.gen_data(indexes)

        return X, y
   
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(self.length//self.n_fold * self.current_fold + self.length % self.n_fold ))
        self.val_indexes = np.arange(len(self.indexes),len(self.indexes) + self.length//self.n_fold)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
