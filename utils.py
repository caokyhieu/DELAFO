import pickle
from tensorflow.keras.models import load_model
import json
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import pdb
def loss_with_constraint(mean=0,variance=1):
    eps = 1e-6
    if isinstance(mean,tf.Tensor) and isinstance(variance,tf.Tensor):
        c = K.sum(K.pow(mean,2) + variance - K.log(variance+ eps) - 1)
    else:
        c = 0

    def sharpe_ratio_loss_l(y_true, y_pred):
        epsilon = 1e-6
        max_bound = 1 - epsilon
        n_tickers = K.cast(K.shape(y_true)[1],K.floatx())
        constraint_value = 3e-3
        y_pred_reshape = K.expand_dims(y_pred, axis=-1)
        z = y_true * y_pred_reshape
        z = K.sum(z, axis=1)
        ## test line
        # sum_w = K.clip(K.sum(y_pred_reshape, axis=1), epsilon, n_tickers* max_bound)
        sum_w = K.minimum(K.maximum(K.sum(y_pred_reshape, axis=1),K.constant(epsilon)),n_tickers*max_bound)

        ## constraint for number of tickers
    #     num_constraint = C * (K.sum(y_pred) - 50)*(K.sum(y_pred)-10)
        rate = z/sum_w
        sharpeRatio = K.mean(rate, axis = 1)/K.maximum(K.std(rate, axis=1),epsilon)
        constraint =  K.sum((1.6 - y_pred) * y_pred,axis=1)
        return K.mean(constraint_value* constraint - sharpeRatio) +  c
    return sharpe_ratio_loss_l
def sharpe_ratio_loss(constraint_value=3e-3,C=1.6,p=1.0,m=1.0):
    def sharpe_ratio_loss_(y_true, y_pred):
        # y_true = K.print_tensor(y_true)
        epsilon = 1e-6
        max_bound = 1 - epsilon
        n_tickers = K.cast(K.shape(y_true)[1],K.floatx())
        # constraint_value = 3e-3
        y_pred_reshape = K.expand_dims(y_pred, axis=-1)
        z = y_true * y_pred_reshape
        z = K.sum(z, axis=1)
        ## test line
        # sum_w = K.clip(K.sum(y_pred_reshape, axis=1), epsilon, n_tickers* max_bound)
        sum_w = K.minimum(K.maximum(K.sum(y_pred_reshape, axis=1),K.constant(epsilon)),n_tickers*max_bound)

        ## constraint for number of tickers
    #     num_constraint = C * (K.sum(y_pred) - 50)*(K.sum(y_pred)-10)
        rate = z/sum_w
        sharpeRatio = K.mean(rate, axis = 1)/K.maximum(K.std(rate, axis=1),epsilon)
        constraint =  K.sum(((C - y_pred) ** p) * (y_pred ** m),axis=1)
        return K.mean(constraint_value* constraint - sharpeRatio)
    return sharpe_ratio_loss_

def sharpe_ratio(y_true, y_pred):
    ## y_true shape (Batch,N_ticker,Time)
    ## y_pred shape (Batch,N_ticker) 
    epsilon = 1e-6
    n_tickers = K.cast(K.shape(y_true)[1],K.floatx())
    y_pred_reshape = K.expand_dims(y_pred, axis=-1) ## shape (Batch,N_ticker,1) 
    y_pred_reshape= K.cast(K.greater(K.clip(y_pred_reshape, 0, 1), 0.5), K.floatx())

    z = y_true * y_pred_reshape
    z = K.sum(z, axis=1)

    # sum_w = K.clip(K.sum(y_pred_reshape, axis=1), epsilon, n_tickers)
    sum_w = K.minimum(K.maximum(K.sum(y_pred_reshape, axis=1),K.constant(epsilon)),n_tickers)
    rate = z/sum_w
    sharpeRatio = K.mean(rate, axis = 1)/K.maximum(K.std(rate, axis=1),epsilon)

    return K.mean(sharpeRatio)



def load_config_file(path):
    with open(path,'r') as file:
        f = json.load(file)
    return f

import json
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



def load_model_by_name(name):
    PATH_MODEL ='/content/gdrive/My Drive/Research/entropy/model'
    with open(PATH_MODEL +'/' + name +'_params.pkl','rb') as f:
        hyper_params,custom_objects = pickle.load(f)

    model = load_model(PATH_MODEL+'/'+name+'.h5',custom_objects=custom_objects)
    return model,hyper_params,custom_objects


def data():
    with open('/content/gdrive/My Drive/Research/entropy/data/data_used.pkl','rb') as f:
        X_train,X_test,y_train,y_test = pickle.load(f)
    return X_train,y_train,X_test,y_test
