import argparse
from preprocess_data import *
from utils import *
from models.addatt_RNN import *
from models.attention_layer import *
from models.RNN_models import *
from models.selfatt_RNN import *
from models.resnet import *
from models.RNN_AddAtt import *
from models.new_model import build_self_att_add_att_model,build_vae_add_att_lstm_model,build_self_att_self_att_model,build_2_SA_AA_model
from models.RNN_AA_SA import build_gru_sa_model,build_lstm_sa_model
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.optimizers import Adam,SGD
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pdb
from models.norm_layer import normalize_layer
from models.tcn_network import compiled_tcn
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import math
from utils import sharpe_ratio_loss,sharpe_ratio
from data_loader import DataGenerator
import yaml
from models.tcn_network import TCN
def step_decay(epoch,lr):
    # drop = 0.9
    # epochs_drop = 10
    if epoch%10==9:
        lr = lr * math.exp(-0.1)
    return lr



class DELAFO:
    
    def __init__(self,model_name,model,data_gen):
        self.model_name = model_name
        self.model = model
        # self.X,self.y,self.tickers = X,y,tickers
        # self.timesteps_input = timesteps_input
        # self.timesteps_output = timesteps_output
        lrate = LearningRateScheduler(step_decay)
        # learning schedule callback
        self.callbacks_list = [lrate]
        # self.opt = Adam(lr=0.1)
        self.data_gen = data_gen
        self.tickers = [i.decode('utf-8') for i in self.data_gen.tickers.squeeze()]


    @classmethod
    def from_existing_config(cls,path_data,model_name,model_config_path,timesteps_input=64,timesteps_output=19,type_norm='batch',batch_size=16,n_fold=10):

        # X,y,tickers,_,_ = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
        if path_data.endswith('.csv'):
            path = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
        else:
            path = path_data
        data_gen = DataGenerator(path =path,batch_size=batch_size,n_fold=n_fold,shuffle=True)
        n_ticker = data_gen.tickers.shape[0]
        

        if model_name == "ResNet":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm

            model = build_resnet_model(hyper_params)
        elif model_name == "GRU":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm

            model = build_gru_model(hyper_params)
        elif model_name == "LSTM":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm

            model = build_lstm_model(hyper_params)
        elif model_name == "AA_GRU":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            print(hyper_params)
            model = build_add_att_gru_model(hyper_params)
        elif model_name == "AA_LSTM":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_add_att_lstm_model(hyper_params)
        elif model_name == "SA_GRU":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_selfatt_gru_model(hyper_params)
        elif model_name == "SA_LSTM":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_selfatt_lstm_model(hyper_params)
        elif model_name == "LSTM_AA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_lstm_add_att_model(hyper_params)
        elif model_name == "GRU_AA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_gru_add_att_model(hyper_params)
        elif model_name == "SA_AA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_self_att_add_att_model(hyper_params)
        elif model_name == "AA_LSTM_VAE":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_vae_add_att_lstm_model(hyper_params)
        elif model_name == "SA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_selfatt_model(hyper_params)
        
        elif model_name == "2_SA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_self_att_self_att_model(hyper_params)
        
        elif model_name == "2_SA_AA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_2_SA_AA_model(hyper_params)

        elif model_name == "LSTM_SA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_lstm_sa_model(hyper_params)

        elif model_name == "GRU_SA":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            model = build_gru_sa_model(hyper_params)
        elif model_name =='TCN':
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (n_ticker,timesteps_input,2)
            hyper_params['type_norm'] = type_norm
            print(hyper_params)
            model = compiled_tcn(**hyper_params)


        model_name = model_name + '_%s'%(type_norm)
        #### scale X by max value
        return cls(model_name,model,data_gen)
 

    @classmethod
    def from_saved_model(cls,path_data,model_path,timesteps_output,batch_size=16,n_fold=10):
        '''  If you load pretrain model with new custom layer, you should put it in custom_objects
            below.
        '''
        model = load_model(model_path,custom_objects={"AdditiveAttentionLayer":AdditiveAttentionLayer,
                                                        "SelfAttentionLayer":SelfAttentionLayer,
                                                        "sharpe_ratio_loss":sharpe_ratio_loss,
                                                        "sharpe_ratio":sharpe_ratio,
                                                        "normalize_layer":normalize_layer,
                                                        "TCN":TCN})
        model_name = model.name
        input_shape = K.int_shape(model.input)
        timesteps_input = input_shape[2]
        # X,y,tickers,_,_ = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
        if path_data.endswith('.csv'):
            path = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
        else:
            path = path_data
        data_gen = DataGenerator(path =path,batch_size=batch_size,n_fold=n_fold,shuffle=True)
        return cls(model_name,model,data_gen)

    def reset_params(self):
        session = K.get_session()
        for layer in self.model.layers: 
            for v in layer.__dict__:
                v_arg = getattr(layer,v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(layer.name, v))
            
    def write_log(self,history,path_dir,name_file):
        his = history.history
        if os.path.exists(path_dir)==False:
            os.makedirs(path_dir)
        with open(os.path.join(path_dir,name_file), 'w') as outfile:
            json.dump(his, outfile,cls=MyEncoder, indent=2)
        print("write file log at %s"%(os.path.join(path_dir,name_file)))

    
    def train_model(self,epochs,path_dir,start_index=3,early_stopping=True,ld=3e-3,C=1.6,p=1.0,m=1.0):
        # tscv = TimeSeriesSplit(n_splits=n_fold)
        # s = 0
        lr = self.model.optimizer.get_config()['learning_rate']
        
        self.tickers = [i.decode("utf-8") for i in self.data_gen.tickers.squeeze()]
        # print(self.tickers)
        # pdb.set_trace()
        # for train_index, test_index in tscv.split(self.X):
        for s in range(self.data_gen.n_fold - 1 ):
            if s> start_index:
                self.data_gen.update_current_fold(s)
                # X_tr, X_val = self.X[train_index], self.X[test_index[range(self.timesteps_output-1,len(test_index),self.timesteps_output)]]
                # y_tr, y_val = self.y[train_index], self.y[test_index[range(self.timesteps_output-1,len(test_index),self.timesteps_output)]]

                X_val = self.data_gen.X[self.data_gen.val_indexes]
                y_val = self.data_gen.y[self.data_gen.val_indexes]
                # pdb.set_trace()
                ## reset model
                self.model.compile(loss=sharpe_ratio_loss(constraint_value=ld,C=C,p=p,m=m), optimizer=Adam(lr=lr), metrics = [sharpe_ratio])
                self.reset_params()
                # X_max_train = X_tr.max(axis=2)[:,:,np.newaxis,:]
                # X_max_val = X_val.max(axis=2)[:,:,np.newaxis,:]
                if early_stopping:
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20,restore_best_weights=True)
                    his = self.model.fit_generator(self.data_gen, epochs= epochs,validation_data=(X_val,y_val),callbacks= [es],shuffle=False)
                else:
                    his = self.model.fit_generator(self.data_gen, epochs= epochs,validation_data=(X_val,y_val),shuffle=False)
                mask_tickers = self.predict_portfolio(X_val)
                print('Sharpe ratio of this portfolio: %s' % str([self.calc_sharpe_ratio(mask_tickers[i],y_val[i]) for i in range(len(y_val))]))
                self.save_model(path_dir,s)

                self.write_log(his,'./logs/%s' % self.model_name,"log_%d.txt"%(self.data_gen.val_indexes[-1]))
            s+=1
        ### train with whole data
        self.data_gen.finish_train()

        self.visualize_log('./logs',self.model_name)


    def save_model(self,path_dir="pretrain_model_2",ver=0):
        if os.path.exists(os.path.join(path_dir,self.model_name))==False:
            os.makedirs(os.path.join(path_dir,self.model_name))
        
        self.model.save(os.path.join(path_dir,self.model_name,str(ver) + '.h5'))
        print("Model saved at %s" % os.path.join(path_dir,self.model_name))

    def predict_portfolio(self,X,mask=True):
        results = self.model.predict(X)
        if mask:
            mask_tickers = results>0.5
            print("There are total %d samples to predict" % len(results))
            for i in range(len(mask_tickers)):
                print('Sample %d : [ %s ]' % (i, ' '.join([self.tickers[j] for j in range(len(self.tickers)) if mask_tickers[i][j]==1])))
        else:
            mask_tickers = results>0.5
            mask_tickers = mask_tickers * results
        return mask_tickers

    def calc_sharpe_ratio(self,weight,y):
        """Here y is the daily return have the shape (tickers,days)
        weight have the shape (tickers,)"""
        n_day = y.shape[1]
        epsilon = 1e-6
        weights = np.round(weight)
        sum_w = np.clip(weights.sum(),epsilon,y.shape[0])
        norm_weight = weights/sum_w
        port_return = norm_weight.dot(y).squeeze()
        mean = np.mean(port_return)
        std = np.maximum(np.std(port_return),epsilon)
        return np.sqrt(n_day) * mean/std

    def visualize_log(self,path_folder,model_name):
        n_cols = 6
        n_rows = 1
        fig,axes = plt.subplots(ncols = n_cols,figsize=(20,3))
        path_files = [os.path.join(path_folder,model_name,file) for file in os.listdir(os.path.join(path_folder,model_name)) if os.path.isfile(os.path.join(path_folder,model_name,file))]
        for i,path in enumerate(path_files[-6:]):

            with open(path) as f:
                history = json.loads(f.read())

            axes[i].plot(history['sharpe_ratio'][50:])
            axes[i].plot(history['val_sharpe_ratio'][50:])
            axes[i].set_ylabel('Sharpe_ratio')
            axes[i].set_xlabel('Epoch')
            axes[i].legend(['Train', 'Test'], loc='upper left')
        new_path = os.path.join('/'.join(path_folder.split('/')[:-1]),'plot',model_name)
        if os.path.exists(new_path)==False:
            os.makedirs(new_path)
        plt.savefig(os.path.join(new_path,'1.png'))


if __name__ =="__main__":
    
    ## path for config_file of each model
    # model_config_path = {'ResNet':"./config/resnet_hyper_params.json",
    #                     'GRU': "./config/gru_hyper_params.json",
    #                     'LSTM':"./config/lstm_hyper_params.json",
    #                     'AA_GRU':"./config/gru_hyper_params.json",
    #                     'AA_LSTM':"./config/lstm_hyper_params.json",
    #                     'SA_GRU':"./config/gru_hyper_params.json",
    #                     'SA_LSTM':"./config/lstm_hyper_params.json",
    #                     'LSTM_AA':"./config/lstm_hyper_params.json",
    #                     'GRU_AA':"./config/gru_hyper_params.json",
    #                     'SA_AA':"./config/lstm_hyper_params.json",
    #                     'AA_LSTM_VAE':"./config/lstm_hyper_params.json",
    #                     'SA':"./config/lstm_hyper_params.json",
    #                     '2_SA':"./config/lstm_hyper_params.json",
    #                     '2_SA_AA':"./config/lstm_hyper_params.json",
    #                     'GRU_SA':"./config/gru_hyper_params.json",
    #                     'LSTM_SA':"./config/lstm_hyper_params.json"}
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, help='Input dir for data')
    # parser.add_argument('--model', choices=[m for m in model_config_path], default='AA_GRU')
    # parser.add_argument('--load_pretrained', type=bool, default=False,help='Load pretrain model')
    # parser.add_argument('--model_path', type=str, default='',help='Path to pretrain model')
    # parser.add_argument('--n_fold', type=int, default=10,help='Number of fold you want to train and eval ')
    # parser.add_argument('--n_epoch', type=int, default=200,help='Number of fold you want to train and eval ')
    # parser.add_argument('--batchsize', type=int, default=16,help='Number of fold you want to train and eval ')
    # parser.add_argument('--timesteps_input', type=int, default=64,help='timesteps (days) for input data')
    # parser.add_argument('--timesteps_output', type=int, default=19,help='Timesteps (days) for output data ')
    # parser.add_argument('--save_model_path', type=str, default='',help='Where you save model')
    # parser.add_argument('--type_norm', choices=['batch','layer'],help='Type of norm')
    # parser.add_argument('--start_index', type=int, default=3,help='starting fold')
    # parser.add_argument('--early_stopping', type=int, default=1,help='starting fold')
    # parser.add_argument('--ld', type=float, default=3e-3,help='lambda coeff ')

    # args = parser.parse_args()
    with open('options/opt.yml', 'rb') as f:
        args = yaml.load(f.read())['training']
    

    if args['load_pretrained'] == False:
        delafo = DELAFO.from_existing_config(args['data_path'],args['model'],args['model_config_path'],args['timesteps_input'],args['timesteps_output'],type_norm=args['type_norm'],n_fold=args['n_fold'],batch_size=args['batchsize'])
        delafo.train_model(epochs=args['n_epoch'],path_dir=args['save_model_path'],start_index=args['start_index'],early_stopping=args['early_stopping'],ld= args['ld'],C=args['C'],p=args['p'],m=args['m'])
    else:
        delafo = DELAFO.from_saved_model(args['data_path'],args['model_path'],args['timesteps_output'],n_fold=args['n_fold'],batch_size=args['batchsize'])
        delafo.train_model(epochs=args['n_epoch'],path_dir=args['save_model_path'],start_index=args['start_index'],early_stopping=args['early_stopping'],ld= args['ld'],C=args['C'],p=args['p'],m=args['m'])
    
    ## save the config file

    with open(os.path.join(args['save_model_path'],args['model'] + '_' + args['type_norm'],'config.yml'), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)