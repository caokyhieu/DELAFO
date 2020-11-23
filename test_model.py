from main import DELAFO
from sklearn.model_selection import TimeSeriesSplit
from keras.models import load_model
import glob
from preprocess_data import *
from utils import *
from models.addatt_RNN import *
from models.attention_layer import *
from models.RNN_models import *
from models.selfatt_RNN import *
from models.resnet import *
from models.RNN_AddAtt import *
import numpy as np
import argparse
from models.norm_layer import LayerNormalization
from data_loader import DataGenerator
import pdb
import yaml
from models.tcn_network import TCN
# from models.new_model import build_self_att_add_att_model,build_vae_add_att_lstm_model
def check_path(paths,index):
    for path in paths:
        # print('Search %s'%(path.split('/')[-1][:-3]))
        if path.split('/')[-1][:-3] == str(index):
            # print('load model %s'%(path.split('/')[-1][:-3]))
            return path
        else:
            continue
    return None


def test_model(dir_model,path_data,timesteps_output=19,n_split=10,start_index = 3,ld=0.03,C=1.6,p=2,m=1.0):
    saved_models = glob.glob(dir_model + '/*.h5')
    assert len(saved_models)!=0
    model_name = saved_models[0].split('/')[-2]
    custom_objects={"AdditiveAttentionLayer":AdditiveAttentionLayer,
                                                        "SelfAttentionLayer":SelfAttentionLayer,
                                                        "sharpe_ratio_loss":sharpe_ratio_loss,
                                                        "sharpe_ratio_loss_":sharpe_ratio_loss(constraint_value=ld,C=C,p=p,m=m),
                                                        "sharpe_ratio":sharpe_ratio,
                                                        "LayerNormalization":LayerNormalization,
                                                        "TCN":TCN}

    model = load_model(saved_models[0],custom_objects=custom_objects)
    model.summary()
    input_shape = K.int_shape(model.input)
    timesteps_input = input_shape[2]
    # X,y,tickers,date_X,date_y = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
    data_gen = DataGenerator(path =path_data,batch_size=16,n_fold=n_split,shuffle=True)
    # ver = list(map(lambda x: int(x.split('.')[0]),[file for file in os.listdir(dir_model) if file.endswith('.h5')]))
    # if len(ver)>0:
    #     ver = np.min(ver)
    # else:
    #     print('Empty folder')
    # random_index = np.random.randint(low=0,high=18)
    random_index = 0

    delafo = DELAFO(model_name,model,data_gen)

    # tscv = TimeSeriesSplit(n_splits=n_split)
    sharpe = []
    num_tickers = []
    
    for i in range(n_split):
        print(i)
        if i>start_index:
            model_path = check_path(saved_models,i)
            if model_path == None:
                print('Check the saved models, do not have %d.h5 model'%(i))
                break
            print(saved_models)
            delafo.model = load_model(model_path,custom_objects=custom_objects)
            delafo.data_gen.update_current_fold(i)
            
            X_val = delafo.data_gen.X[delafo.data_gen.val_indexes[timesteps_output-1::timesteps_output]]
            y_val = delafo.data_gen.y[delafo.data_gen.val_indexes[timesteps_output-1::timesteps_output]]

            #### calculate 
            mask_tickers = delafo.predict_portfolio(X_val,mask=True)
            num_tickers.append((mask_tickers>0.5).sum(axis=1).mean())
            print('Sharpe ratio of this portfolio: %s' % str([delafo.calc_sharpe_ratio(mask_tickers[j],y_val[j]) for j in range(len(y_val))]))
            sharpe.append(np.mean([delafo.calc_sharpe_ratio(mask_tickers[j],y_val[j]) for j in range(len(y_val))]))
        
       
    
    print('Mean of Sharpe ratio %0.4f'%(np.mean(sharpe)))
    print('STD of Sharpe ratio %0.4f'%(np.std(sharpe)))
    # pdb.set_trace()
    print('Mean of num tickers %0.4f'%(np.mean(num_tickers)))


if __name__ =='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, help='Input dir for data')
    # parser.add_argument('--dir_model', type=str, default='',help='path to saved model')
    # parser.add_argument('--timesteps_output', type=int, default=19,help='Timesteps (days) for output data ')
    # parser.add_argument('--n_split', type=int, default=10,help='Time of split for forward test ')
    # parser.add_argument('--start_index', type=int, default=3,help='test starting fold')
    # args = parser.parse_args()
    with open('options/opt_test.yml', 'rb') as f:
        args = yaml.load(f.read())['testing']
    
    test_model(**args)
    





        
